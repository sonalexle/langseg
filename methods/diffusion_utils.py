# https://github.com/google/prompt-to-prompt/
from PIL import Image
from typing import Tuple, List

import numpy as np
import torch
from cv2 import putText, getTextSize, FONT_HERSHEY_SIMPLEX
import matplotlib.pyplot as plt

from methods.diffusion_seg import normalize_ca
from methods.diffusion_patch import AttentionStore


def configure_ldm(ldm):
    ldm.vae.requires_grad_(False)
    ldm.unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    ldm.text_encoder.text_model.encoder.requires_grad_(False)
    ldm.text_encoder.text_model.final_layer_norm.requires_grad_(False)
    ldm.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)


def load_model_weights(pipe, weight_path, model_type):
    # https://github.com/chaofengc/TexForce/
    # load_model_weights(pipe, './TexForce/lora_weights/sd15_refl/', 'unet+lora')
    # load_model_weights(pipe, './TexForce/lora_weights/sd15_texforce/', 'text+lora')
    if model_type == 'text+lora':
        from peft import PeftModel
        text_encoder = pipe.text_encoder
        PeftModel.from_pretrained(text_encoder, weight_path)
    elif model_type == 'unet+lora':
        pipe.unet.load_attn_procs(weight_path)


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))
        if image.ndim == 3:
            image = image[:, :, :3]
        elif image.ndim == 2:
            image = image[..., None]
        else:
            raise ValueError("Unexpected image dimension: {}".format(image.shape))
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


def aggregate_attention(attention_store: AttentionStore, res: int, is_cross: bool, select: int, prompts):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for itemname, item in attention_maps.items():
        if is_cross and "attn1" in itemname: continue
        elif not is_cross and "attn2" in itemname: continue
        if item.shape[1] == num_pixels:
            cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]#.softmax(dim=-1)
            out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(attention_store: AttentionStore, res: int, prompts, tokenizer, select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    if isinstance(attention_store, torch.Tensor):
        attention_maps = attention_store
    else:
        attention_maps = aggregate_attention(attention_store, res, True, select, prompts)
        attention_maps = normalize_ca(attention_maps[None,...]).squeeze()
    max_score = attention_maps[...,1:].max()
    images = []
    for i in range(len(tokens)):
        try:
            image = attention_maps[...,i]
        except IndexError:
            break
        if i == 0:
            img_max = image.max()
        else:
            img_max = max_score
        image = 255 * image / img_max
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.float().numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    view_images(np.stack(images, axis=0))


def show_self_attention_comp(attention_store: AttentionStore, res: int, prompts,
                             max_com=10, select: int = 0, seq_first=True):
    if isinstance(attention_store, torch.Tensor):
        attention_maps = attention_store
    else:
        attention_maps = aggregate_attention(attention_store, res, False, select, prompts)
    attention_maps = attention_maps.float().numpy().reshape((res ** 2, res ** 2))
    if not seq_first:
        attention_maps = attention_maps.T # because originally dim 1 is the softmax dim, but here we got dim 0 as the softmax dim instead
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    # print(vh.shape) # (res ** 2, res ** 2)
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res) # or u[:,i]
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    view_images(np.concatenate(images, axis=1))


def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    display_img(pil_img)


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def display_img(image):
    global display_index
    plt.figure(figsize=(12,7))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
