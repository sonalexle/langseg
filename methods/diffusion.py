import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
from PIL import Image


def reshape_attention_maps_to_batch(attn: torch.Tensor, batch_size: int):
    """
    Reshape attention maps from (batch_size * n_heads, ...) to (batch_size, n_heads, ...)
    """
    n_heads = attn.shape[0] // batch_size
    attn = attn.view(batch_size, n_heads, *attn.shape[1:])
    return attn


def get_attention_maps(
    raw_attention_maps,
    batch_size,
    label_indices=None,
    output_size=64,
    average_layers=True,
    simple_average=False,
    apply_softmax=True,
    softmax_dim=-1,
):
    ca_maps, sa_maps = [], []
    ca_resolutions = []
    sa_resolutions = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    for layer in raw_attention_maps.keys():
        assert layer.endswith("attn1") or layer.endswith("attn2"), "Unrecognized layer: {}".format(layer)
        attn_map = raw_attention_maps[layer]
        bsz_x_nheads, img_embed_len, seq_len = attn_map.shape # seq_len = 77 (cross) or HW (self)
        assert bsz_x_nheads == batch_size, "we should have averaged over heads in attention_store"
        if (img_embed_len > output_size ** 2) or (simple_average and img_embed_len != output_size ** 2):
            continue
        attn_map = attn_map.to(device)
        attn_res = int(img_embed_len**0.5) # attention resolution

        res_dict = sa_resolutions if layer.endswith("attn1") else ca_resolutions
        res_dict.append(attn_res)

        # if apply_softmax:
        #     attn_map = attn_map.softmax(dim=softmax_dim)
        
        # if seq_len != attn_res**2:
        #     attn_map = attn_map.sigmoid()#softmax(dim=-2)
        # else:
        #     attn_map = attn_map.softmax(dim=-1)

        if layer.endswith("attn2"):
            if label_indices is not None: # average over multiple tokens of the same label
                for lids in label_indices: # assign the average attention map to the first token of the label
                    attn_map[..., lids[0]] = attn_map[..., lids].mean(dim=-1)
            # if label_indices is not None: # average over multiple tokens of the same label
            #     for i, lids in enumerate(label_indices): # assign the average attention map to the first token of the label
            #         attn_map[i, ..., lids[0]] = attn_map[i, ..., lids].mean(dim=-1)

        if average_layers and not simple_average:
            attn_map = upscale_attn(attn_map.float(), output_size, is_cross=layer.endswith("attn2"))

        # attn_map = reshape_attention_maps_to_batch(attn_map, batch_size).mean(dim=1) # (BT, HW, *L) -> (B, HW, *L)

        attn_dict = sa_maps if layer.endswith("attn1") else ca_maps
        attn_dict.append(attn_map)

    if average_layers:
        if not simple_average:
            ca_kwargs = {"batch_size": batch_size, "device": device, "is_cross": True, "resolutions": ca_resolutions}
            sa_kwargs = {"batch_size": batch_size, "device": device}
            agg_func = average_attention_layers
        else:
            ca_kwargs, sa_kwargs = {}, {}
            agg_func = simple_attn_avg
    else:
        ca_kwargs = {"attn_res": ca_resolutions}
        sa_kwargs = {"sa_res": sa_resolutions}
        agg_func = agg_same_size_maps

    if len(ca_maps) > 0:
        ca_maps = agg_func(ca_maps, **ca_kwargs)
    if len(sa_maps) > 0:
        sa_maps = agg_func(sa_maps, **sa_kwargs)

    return ca_maps, sa_maps # tensor or dict of tensors


def simple_attn_avg(attn_maps):
    # all maps must have the same spatial resolution
    return torch.stack(attn_maps, dim=0).mean(dim=0)


def average_attention_layers(attn_maps, *, batch_size, device, is_cross=False, resolutions=None, upscale_cross=False):
    # attn_maps: (batch_size, seq_len, output_size**2)
    # idea taken from DiffSeg paper https://arxiv.org/abs/2308.12469
    if resolutions is None:
        resolutions = [int(attn.shape[1]**0.5) for attn in attn_maps]
    max_res = max(resolutions)
    if is_cross: # inverse weight for cross attention maps
        sum_res = np.sum(max_res / np.array(resolutions))
    else:
        sum_res = sum(resolutions)
    seq_len = attn_maps[0].shape[2:]
    attn_merged = torch.zeros(
        (batch_size, max_res**2, *seq_len), device=device, dtype=torch.float32
    )
    for i, attn in enumerate(attn_maps):
        res_sa = resolutions[i]
        if is_cross: # inverse weight for cross attention maps
            weight = (max_res / res_sa) / sum_res
        else:
            weight = res_sa / sum_res
        factor = max_res // res_sa
        if not is_cross:
            all_indices = torch.arange(max_res)
            rows = all_indices.unsqueeze(1).div(factor, rounding_mode="floor")
            cols = all_indices.unsqueeze(0).div(factor, rounding_mode="floor")
            indices = (rows * res_sa + cols).flatten().long()
            # attn = attn / attn.sum(dim=-1, keepdim=True)
            attn = attn[:, indices, ...]
        elif upscale_cross:
            attn = upscale_attn(attn, max_res, is_cross=True)
        attn_merged += weight * attn.to(device).to(attn_merged.dtype)

    return attn_merged


def upscale_attn(attn, output_size, is_cross=True):
    assert attn.ndim == 3
    seq_len = attn.shape[2]
    res = int(attn.shape[1]**0.5)
    if is_cross:
        attn = attn.permute(0, 2, 1)
        kwargs = {"mode": "bicubic", "antialias": True}
    else:
        kwargs = {"mode": "bilinear", "antialias": False}
    attn = attn.view(-1, seq_len, res, res)
    attn = F.interpolate(attn, size=output_size, align_corners=False, **kwargs)#.to(attn.dtype)
    if is_cross:
        attn = attn.permute(0, 2, 3, 1)
        attn = attn.view(-1, output_size**2, seq_len)
    else:
        attn = attn.view(-1, seq_len, output_size**2)
    return attn


def agg_same_size_maps(attn_maps, attn_res):
    attn_dict = {}
    for res, layer in zip(attn_res, attn_maps):
        if res not in attn_dict:
            attn_dict[res] = []
        attn_dict[res].append(layer)
    attn_dict = {res: torch.stack(ls, dim=0).mean(dim=0) for res, ls in attn_dict.items()}
    return attn_dict


def latent2image(model, latents):
    latents = 1 / 0.18215 * latents
    image = model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def image2latent(vae, image, normalize=True):
    if type(image) is Image:
        image = np.array(image)
    if type(image) is torch.Tensor and image.dim() == 4:
        latents = image
    else:
        image = torch.from_numpy(image).float()
        if normalize:
            image = image / 127.5 - 1
        if image.dim() == 3:
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif image.dim() == 4:
            image = image.permute(0, 3, 1, 2) # (b, c, h ,w)
        else:
            raise ValueError("Unexpected image dimension: {}".format(image.dim()))
        image = image.to(vae.device).to(vae.dtype)
    latents = vae.encode(image)['latent_dist'].mean
    latents = latents * vae.config.scaling_factor
    return latents


def get_noisy_latents(noise_scheduler, latents, timestep=101):
    gen = torch.Generator(device=latents.device)
    gen.manual_seed(0)
    noise = torch.randn(latents.size(), generator=gen, dtype=latents.dtype, layout=latents.layout, device=latents.device)

    timesteps = torch.full((latents.shape[0],), timestep, device=latents.device, dtype=torch.long)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    latents = noise_scheduler.add_noise(latents, noise, timesteps)
    latents = noise_scheduler.scale_model_input(latents, timestep)

    return latents, noise, timesteps


@torch.no_grad()
def diffusion_step(
    model, latents, context, timesteps, attention_store,
    guidance_scale=7.5, low_resource=False, no_uncond=False
):
    if no_uncond:
        noise_pred_uncond = torch.zeros_like(latents)
        noise_prediction_text = model.unet(latents, timesteps, encoder_hidden_states=context)["sample"]
        guidance_scale = 1.0
    else:
        if low_resource:
            attention_store.uncond_fwpass = True
            noise_pred_uncond = model.unet(latents, timesteps, encoder_hidden_states=context[0])["sample"]
            attention_store.uncond_fwpass = False
            noise_prediction_text = model.unet(latents, timesteps, encoder_hidden_states=context[1])["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = model.unet(latents_input, timesteps, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    # latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    # return latents
    return noise_pred


@torch.no_grad()
def training_step(
    model, text_embeddings, images, attention_store,
    no_uncond=True, normalize=True, guidance_scale=7.5, low_resource=False, timestep=101,
):
    batch_size = text_embeddings.shape[0]
    if no_uncond:
        context = text_embeddings
    else:
        max_length = text_embeddings.shape[1]
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        context = [uncond_embeddings, text_embeddings]
        if not low_resource:
            context = torch.cat([uncond_embeddings, text_embeddings])
    latents = image2latent(model.vae, images, normalize=normalize)
    latents, noise, timesteps = get_noisy_latents(model.scheduler, latents, timestep=timestep)
    noise_pred = diffusion_step(
        model, latents, context, timesteps, attention_store,
        guidance_scale=guidance_scale, low_resource=low_resource, no_uncond=no_uncond
    )

    if model.scheduler.config.prediction_type == "epsilon":
        target = noise
    elif model.scheduler.config.prediction_type == "v_prediction":
        target = model.scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {model.scheduler.config.prediction_type}")

    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

    return loss
