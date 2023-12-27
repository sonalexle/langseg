from PIL import Image
import random
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
torch.backends.cuda.matmul.allow_tf32 = True
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

from methods.prompt_engineering import extract_class_embeddings


class CLIPMultilabelClassifier(nn.Module):
    def __init__(self, model_id, candidate_labels, device='cuda'):
        super().__init__()
        self.clip_processor = CLIPProcessor.from_pretrained(model_id)
        self.clip_model = CLIPModel.from_pretrained(model_id)
        self.clip_model.to(device)
        self.box_templates = init_boxes(init_choices())
        self.candidate_labels = candidate_labels
        self.device = device

    @torch.no_grad()
    def init_text_embeds(self):
        text_embeds = extract_class_embeddings(self.clip_model, self.clip_processor.tokenizer, self.candidate_labels)
        self.text_embeds = torch.stack(list(text_embeds.values()), dim=0).to(self.device)

    @torch.no_grad()
    def forward(self, image: Image.Image, choice=(8,8), clf_thresh=0.5):
        assert isinstance(image, Image.Image)
        image_crops, _ = obtain_image_crops(image, choice, self.box_templates)
        image_crops = [image] + image_crops
        pixel_values = self.clip_processor(images=image_crops, return_tensors="pt").pixel_values

        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values.to(self.device))
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, self.text_embeds.t()) * logit_scale

        probs = logits_per_image.softmax(dim=-1)
        # probs = probs.max(dim=0).values
        probs = (probs[0] + probs[1:].max(dim=0).values) / 2
        labels = (probs > clf_thresh).long()

        return labels, probs

    @torch.no_grad()
    def forward_v2(self, image: Image.Image, choice=(8,8), clf_thresh=0.5):
        assert isinstance(image, Image.Image)
        image_crops, _ = obtain_image_crops(image, choice, self.box_templates)
        image_crops = [image] + image_crops
        pixel_values = self.clip_processor(images=image_crops, return_tensors="pt").pixel_values

        image_embeds = self.clip_model.get_image_features(pixel_values=pixel_values.to(self.device))
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = torch.matmul(image_embeds, self.text_embeds.t()) * logit_scale

        probs = logits_per_image.softmax(dim=-1)
        # probs = probs.max(dim=0).values
        probs = (probs[0] + probs[1:].max(dim=0).values) / 2
        labels = (probs > clf_thresh).long()
    
        label_global = logits_per_image[0,:].argmax() # force a prediction on the original image (0th image)
        zeros = torch.zeros_like(labels)
        zeros[label_global] = 1
        label_global = zeros
        
        labels = torch.logical_or(labels, label_global)

        return labels, probs

    @torch.no_grad()
    def forward_v3(self, image: Image.Image, clf_thresh=0.5):
        assert isinstance(image, Image.Image)
        pixel_values = self.clip_processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        image_embeds = self.get_dense_image_features(self.clip_model, pixel_values) # (1+n_patches, feat_dim)
        image_embeds = image_embeds[:, 1:].squeeze() # dont take CLS image token
        image_embeds = F.normalize(image_embeds, dim=-1)

        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_patch = torch.matmul(image_embeds, self.text_embeds.t()) * logit_scale
        
        probs = logits_per_patch.softmax(dim=-1)
        probs = probs.max(dim=0).values
        labels = (probs > clf_thresh).long()

        return labels, probs     

    @staticmethod
    @torch.no_grad()
    def get_dense_image_features(
        clip_model: CLIPModel,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        output_attentions = output_attentions if output_attentions is not None else clip_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else clip_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else clip_model.config.use_return_dict

        vision_outputs = clip_model.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        x_ori, x = vision_outputs.hidden_states[-1]
        x[:, 0] = x_ori[:, 0]
        unpooled_output = x

        # unpooled_output = vision_outputs[0]  # last hidden state, with all image patches + [CLS] token (patch 0)
        unpooled_output = clip_model.vision_model.post_layernorm(unpooled_output)

        image_features = clip_model.visual_projection(unpooled_output)

        return image_features


def init_choices(M=16):
    choices = []
    for m in range(1, M+1):
        for n in range((m + 1)//2, min(m*2 + 1, M+1)):
            choices.append((m, n))
    return choices


def init_boxes(choices):
    box_templates = {}
    for choice in choices:
        M, N = choice
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, N + 1), torch.linspace(0, 1, M + 1),
                                        indexing='xy')
        x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
        x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
        pseudo_boxes = torch.cat([x0y0s, x1y1s],
                                 dim=-1).view(-1, 4)

        assert pseudo_boxes.shape[0] == M*N
        box_templates[choice] = pseudo_boxes

    return box_templates


def obtain_image_crops(image, choice, box_templates):
    image_crops = []
    img_w, img_h = image.size
    max_anns = -1#20
    crop_scale = 1
    normed_boxes = box_templates[choice]
    indices = list(range(len(normed_boxes)))
    random.shuffle(indices)
    indices = indices[:max_anns]
    boxes = normed_boxes * torch.tensor([img_w, img_h, img_w, img_h])
    for idx in indices:
        box = boxes[idx]
        x0, y0, x1, y1 = box.tolist()    # todo expand
        if crop_scale > 1.0:
            box_w, box_h = x1 - x0, y1 - y0
            cx, cy = (x1 + x0)/2, (y1 + y0)/2
            delta_factor = 0.5 * self.args.crop_scale
            x0, y0, x1, y1 = max(cx - box_w * delta_factor, 0), max(cy - box_h * delta_factor, 0), \
                min(cx + box_w * delta_factor, img_w), min(cy + box_h * delta_factor, img_h)
        image_crops.append(image.crop((x0, y0, x1, y1)))

    return image_crops, boxes[indices]


class BLIPMultilabelClassifier(nn.Module):
    def __init__(self, model_id, candidate_labels, device='cuda'):
        super().__init__()
        self.blip_processor = InstructBlipProcessor.from_pretrained(model_id)
        self.blip_model = InstructBlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
        self.inputs = self.get_input_ids(candidate_labels)
        self.init_answer_tokens()

    def init_answer_tokens(self):
        tokens_noyes = [150, 4273] # set 0, no - yes
        tokens_falsetrue = [6136, 1176] # set 1, false - true
        self.set_ind = torch.tensor([[0,1],[2,3]], device=self.blip_model.device)
        self.tokens_all = tokens_noyes + tokens_falsetrue

    def get_input_ids(self, candidate_labels):
        prompts = []
        question_template = 'Based on the image, is this statement true or false? "{}" Answer: '
        option_template = "there is {}"
        for l in candidate_labels:
            prompts.append(question_template.format(option_template.format(l)))
        input_ids = self.blip_processor(
            images=None, text=prompts, return_tensors="pt", padding=True
        ).to(self.blip_model.device)
        return input_ids

    @torch.no_grad()
    def forward(self, image: Image.Image, clf_thresh=0.5):
        assert isinstance(image, Image.Image)
        pixel_values = self.blip_processor(images=image, text=None, return_tensors="pt") \
            .to(self.blip_model.device, self.blip_model.dtype)["pixel_values"]
        pixel_values = pixel_values.expand(len(self.inputs["input_ids"]), *pixel_values.shape[1:])
        self.inputs["pixel_values"] = pixel_values

        outputs = self.blip_model.generate(
            **self.inputs, max_new_tokens=1,
            output_scores=True, return_dict_in_generate=True
        )

        outputs = outputs.scores[0][:, self.tokens_all]
        
        # for each prompt, check which of "no, yes" or "false, true" the max logit is
        which_set = (outputs.argmax(dim=-1) > 1).long()
        which_set = self.set_ind[which_set].view(-1,)

        # compute binary probs within either "no, yes" or "false, true"
        probs = outputs[torch.arange(outputs.shape[0]).repeat_interleave(2), which_set].reshape(-1, 2).softmax(dim=-1)[:, 1]
        labels = (probs > clf_thresh).long()

        del self.inputs["pixel_values"]

        return labels, probs