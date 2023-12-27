# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
# https://github.com/orpatashnik/local-prompt-mixing
# https://github.com/google/prompt-to-prompt/
# https://github.com/aliasgharkhani/SLiMe
import abc
from typing import Optional, Union, Tuple, Dict, List

from diffusers.models.attention_processor import Attention, USE_PEFT_BACKEND
from diffusers import UNet2DConditionModel
import torch
from torch.nn import functional as F
import numpy as np

from methods.sd_attn_layers import *


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource and not self.no_uncond else 0

    @abc.abstractmethod
    def forward(self, attn, layer_key: str):
        raise NotImplementedError

    def __call__(self, attn, layer_key: str, n_heads=8):
        if self.no_uncond:
            attn = self.forward(attn, layer_key, n_heads=n_heads)
        elif self.low_resource:
            # idea is to collect self-attn for uncond pass, and cross-attn for cond
            if ("attn2" in layer_key and not self.uncond_fwpass) or \
                ("attn1" in layer_key and self.uncond_fwpass):
                attn = self.forward(attn, layer_key, n_heads=n_heads)
        else:
            h = attn.shape[0]
            if "attn2" in layer_key:
                attn[h // 2:] = self.forward(attn[h // 2:], layer_key, n_heads=n_heads)
            elif "attn1" in layer_key: # takes uncond embed pass for self-attn
                attn[:h // 2] = self.forward(attn[:h // 2], layer_key, n_heads=n_heads)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, low_resource, no_uncond=True):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.low_resource = low_resource
        self.no_uncond = no_uncond


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class DummyController:
    def __call__(self, *args):
        return args[0]

    def __init__(self):
        self.num_att_layers = 0

        
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store(layer_keys: List[str]) -> Dict[str, Optional[torch.Tensor]]:
        return {key: None for key in layer_keys}

    def forward(self, attn, layer_key: str, n_heads=8):
        if layer_key not in self.step_store:
            raise KeyError(f"Unknown layer key {layer_key} not found in step store.")
        if self.low_resource:
            attn = attn.cpu()
        attn = attn.view(-1, n_heads, *attn.shape[1:]).mean(dim=1)
        self.step_store[layer_key] = attn
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                self.attention_store[key] += self.step_store[key]
        self.step_store = self.get_empty_store(self.layer_keys)

    def get_average_attention(self):
        average_attention = {key: item / self.cur_step for key, item in
                             self.attention_store.items()}
        return average_attention

    def reset(self):
        super().reset()
        self.step_store = self.get_empty_store(self.layer_keys)
        self.attention_store = {}

    def __init__(self, low_resource=False, no_uncond=True, layer_keys: List[str] = None):
        super().__init__(low_resource, no_uncond)
        if layer_keys is None:
            self.layer_keys = ATTENTION_LAYERS
        else:
            self.layer_keys = layer_keys
        self.step_store = self.get_empty_store(self.layer_keys)
        self.attention_store = {}
        self.uncond_fwpass = False

    def __getitem__(self, key):
        return self.attention_store[key]


def modify_get_attention_scores(model_self):
    """
    Modify the get_attention_scores function of the Attention class to also return the attention scores.
    :param model_self: The Attention module instance to modify.
    """
    def get_attention_scores(query, key, attention_mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = query.dtype
        if model_self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=model_self.scale,
        )
        del baddbmm_input

        if model_self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        
        # del attention_scores

        # if query.shape[1] != key.shape[1]: # cross attention
        #     # cosine similarity for cross attention scores
        #     query = F.normalize(query, p=2., dim=-1)
        #     key = F.normalize(key, p=2., dim=-1)
        #     attention_scores = torch.bmm(query, key.transpose(-1, -2))

        attention_probs = attention_probs.to(dtype)

        return attention_probs, attention_scores
    return get_attention_scores


def modify_attention_forward(model_self):
    """
    Modify the forward function of the Attention class to also return the attention scores.
    :param model_self: The Attention module instance to modify.
    """
    def forward(
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # encoder_hidden_states shape: (batch_size, seq_len, embed_dim)
        # hidden_states shape: (batch_size, n_pixels, embed_dim)
        # embed_dim = head_dim * n_heads (n_heads = 8 usually)
        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if model_self.spatial_norm is not None:
            hidden_states = model_self.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = model_self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if model_self.group_norm is not None:
            hidden_states = model_self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = model_self.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif model_self.norm_cross:
            encoder_hidden_states = model_self.norm_encoder_hidden_states(encoder_hidden_states)

        key = model_self.to_k(encoder_hidden_states, *args)
        value = model_self.to_v(encoder_hidden_states, *args) # does not change shape

        query = model_self.head_to_batch_dim(query)
        key = model_self.head_to_batch_dim(key)
        value = model_self.head_to_batch_dim(value) # (batch_size * head_size, seq_len, embed_dim // head_size)

        # attention_probs, attention_scores = model_self.get_attention_scores(query, key, attention_mask) # (batch_size * head_size, n_pixels, seq_len)
        attention_probs = model_self.get_attention_scores(query, key, attention_mask) # (batch_size * head_size, n_pixels, seq_len)
        attention_scores = attention_probs

        hidden_states = torch.bmm(attention_probs, value) # (batch_size * head_size, n_pixels, embed_dim // head_size)
        hidden_states = model_self.batch_to_head_dim(hidden_states) # (batch_size, n_pixels, embed_dim)

        # linear proj
        hidden_states = model_self.to_out[0](hidden_states, *args)  # does not change shape
        # dropout
        hidden_states = model_self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if model_self.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / model_self.rescale_output_factor

        return hidden_states, attention_scores
    return forward


def register_attention_hooks(
        unet: UNet2DConditionModel, controller, attention_layers_to_use: List,
        hooks: Optional[Dict] = {}, detach: bool = False
    ):
    if len(hooks) > 0:
        remove_hooks(hooks, lambda layer: True)
        assert len(hooks) == 0
    for module_name in attention_layers_to_use:
        module: Attention = eval("unet." + module_name)
        # module.get_attention_scores = modify_get_attention_scores(module)
        module.forward = modify_attention_forward(module)
        hooks[module_name] = module.register_forward_hook(
            create_nested_hook_for_attention_modules(controller, module_name, detach=detach)
        )
    return hooks


def create_nested_hook_for_attention_modules(controller, n, detach: bool = True):
    """
    Store the attention maps of the attention module in the `controller` object.
    Also return only the hidden states of the attention module since `Attention.forward`
    is modified to return a tuple, and the model expects only the hidden states.
    """
    def hook(module, input, output):
        hidden_states, attention_scores = output
        if detach:
            attention_scores = attention_scores.detach()
        controller(attention_scores, n, n_heads=module.heads)
        return hidden_states
    return hook


def extract_original_attn_methods(unet):
    """
    Extract the original get_attention_scores and forward methods of the Attention class.
    Do this recursively for all submodules of the UNet2DConditionModel, whenever the class name of the submodule is Attention
    """
    def inner_extract(net_name, net, original_attn_methods, n_layers):
        """
        :param net_: The UNet2DConditionModel instance or one of its submodules.
        :return: A dictionary with the original methods.
        """
        if net.__class__.__name__ == "Attention":
            original_attn_methods[net_name] = {
                "get_attention_scores": net.get_attention_scores,
                "forward": net.forward
            }
            return n_layers + 1
        elif hasattr(net, 'named_children'):
            for net_ in net.named_children():
                if net_[0].isnumeric():
                    new_name = f"{net_name}[{net_[0]}]"
                else:
                    new_name = f"{net_name}.{net_[0]}"
                n_layers = inner_extract(new_name, net_[1], original_attn_methods, n_layers)
        return n_layers

    n_layers = 0
    original_attn_methods = {}
    sub_nets = unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name or "up" in net_name or "mid" in net_name:
            n_layers = inner_extract(net_name, net, original_attn_methods, n_layers)

    return original_attn_methods, n_layers


def restore_original_attn_methods(unet, original_attn_methods, layers_to_restore: Optional[List] = None, hooks: Optional[Dict] = None):
    """
    Restore the original get_attention_scores and forward methods of the Attention class.
    Do this recursively for all submodules of the UNet2DConditionModel, whenever the class name of the submodule is Attention
    """
    # the two optional args must be either both None or both not None
    assert (layers_to_restore is None and hooks is None) \
        or (layers_to_restore is not None and hooks is not None), \
        "If restoring specific attention layers, both `layers_to_restore` and `hooks` must be provided."

    def inner_restore(net_name, net, original_attn_methods, n_layers):
        """
        :param net_: The UNet2DConditionModel instance or one of its submodules.
        :return: the number of layers restored.
        """
        if net.__class__.__name__ == "Attention" and (layers_to_restore is None or net_name in layers_to_restore):
            net.get_attention_scores = original_attn_methods[net_name]["get_attention_scores"]
            net.forward = original_attn_methods[net_name]["forward"]
            return n_layers + 1
        elif hasattr(net, 'named_children'):
            for net_ in net.named_children():
                if net_[0].isnumeric():
                    new_name = f"{net_name}[{net_[0]}]"
                else:
                    new_name = f"{net_name}.{net_[0]}"
                n_layers = inner_restore(new_name, net_[1], original_attn_methods, n_layers)
        return n_layers

    n_layers = 0
    sub_nets = unet.named_children()
    for net_name, net in sub_nets:
        if "down" in net_name or "up" in net_name or "mid" in net_name:
            n_layers += inner_restore(net_name, net, original_attn_methods, 0)
    
    print(f"Restored {n_layers} attention layers.")
    if hooks is not None:
        remove_hooks(hooks, lambda layer: layer in layers_to_restore)


def remove_hooks(hooks, should_remove = lambda layer: True):
    layers = [l for l in hooks.keys()]
    for layer in layers:
        if should_remove(layer):
            hooks.pop(layer).remove()
