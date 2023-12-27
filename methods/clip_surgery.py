# https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py
# https://github.com/xmed-lab/CLIP_Surgery
# https://github.com/WalBouss/GEM
from typing import Optional, Tuple, Union

from transformers.modeling_outputs import BaseModelOutput
import torch
from torch import nn
from torch.nn import functional as F


def patch_encoder_forward(model_self):
    self = model_self
    def forward(
        inputs_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )
            if isinstance(layer_outputs, list):
                hidden_states = [layer_outputs[0], layer_outputs[-1]]
            else:
                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        assert isinstance(hidden_states, list)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        hidden_states = hidden_states[0] # return the output from the original path

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
    return forward


def patch_encoderlayer_forward(model_self, is_last=False):
    self = model_self
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        if isinstance(hidden_states, list):
            x_ori, x = hidden_states
            assert isinstance(x_ori, torch.Tensor) and isinstance(x, torch.Tensor)
            attn_outputs = self.self_attn(
                hidden_states=self.layer_norm1(x_ori),
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            assert isinstance(attn_outputs, list) and len(attn_outputs) == 3
            x_ori_res, attn_weights, x_res = attn_outputs
            x_ori += x_ori_res
            x_ori = x_ori + self.mlp(self.layer_norm2(x_ori))
            x += x_res # skip ffn for the new path    
        else:
            x = hidden_states
            attn_outputs = self.self_attn(
                hidden_states=self.layer_norm1(x),
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
                output_attentions=output_attentions,
            )
            assert isinstance(attn_outputs, list) and len(attn_outputs) == 3
            x_ori_res, attn_weights, x_res = attn_outputs
            x_ori = x + x_ori_res
            x_ori = x_ori + self.mlp(self.layer_norm2(x_ori))
            x += x_res

        outputs = (x_ori,)

        if output_attentions:
            outputs += (attn_weights,)

        outputs += (x,)
        outputs = list(outputs)

        return outputs
    return forward


def patch_selfattn_forward(model_self):
    self = model_self
    def forward(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        # clip surgery v-v attention
        attn_weights = torch.bmm(value_states, value_states.transpose(1, 2)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        v = torch.bmm(attn_probs, value_states)
        v = v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        v = v.transpose(1, 2)
        v = v.reshape(bsz, tgt_len, embed_dim)
        v = self.out_proj(v)

        return [attn_output, attn_weights_reshaped, v]
    return forward


def patch_clip_for_maskclip(clip_model):
    clip_model.vision_model.encoder.forward = patch_encoder_forward(clip_model.vision_model.encoder)
    # patch the last 6 layers
    for i in range(1, 7): # surgery 7, maskclip 2
        clipvit_layer = clip_model.vision_model.encoder.layers[-i]
        clipvit_layer.self_attn.forward = patch_selfattn_forward(clipvit_layer.self_attn)
        clipvit_layer.forward = patch_encoderlayer_forward(clipvit_layer, is_last=i==1)
