import torch
from methods.diffseg import generate_masks, generate_sampling_grid


def filter_prompt(logits, val_labels, cat_to_label_id):
    # logits (N, C, H, W)
    concept_ind = []
    for c, v in cat_to_label_id.items():
        if "background" not in val_labels and c == "background": continue
        concept_ind.append(c not in val_labels)
    concept_ind = torch.tensor([concept_ind]).to(logits.device)
    logits = logits.clone()
    logits[concept_ind] = -100
    return logits


def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)


def diffseg(sa, out_res=320, refine=False, kl_thresh=0.8):
    grid = generate_sampling_grid(16, res=64) # 32
    grid = torch.from_numpy(grid).to(sa.device)
    kl_threshold = [kl_thresh]*3 # 0.8
    weights = sa.reshape(64, 64, 64, 64)
    pred = generate_masks(weights, kl_threshold, grid, out_res=out_res, refine=refine)[None]
    return pred


def label_clusters(clusters, logits, reshape_logits=True):
    # logits: (N, C, H, W)
    if reshape_logits:
        logits = logits.permute(0, 2, 3, 1).view(1, -1, logits.shape[1])
    remap = {}
    for c in clusters.unique().tolist():
        region = (clusters == c).flatten()
        region_repr = logits[:, region].mean(dim=1, keepdim=True)
        pred_cls = region_repr.argmax(dim=-1).squeeze().item()
        remap[c] = pred_cls
    remap = (torch.tensor(list(remap.keys())), torch.tensor(list(remap.values())))
    remap = remap[0].to(logits.device), remap[1].to(logits.device)
    clusters = clusters.to(logits.device)
    return remap_values(remap, clusters)


def get_voc2012_probs(logits, offset=0.8, temp=0.1):
    # logits: (N, C, H, W)
    probs = (logits * 1/temp).softmax(dim=1) # without softmax -> person class VERY bad
    max_probs = probs.max(dim=1)[0]
    bg_probs = 1 - max_probs - offset
    bg_probs = torch.where(bg_probs > 0, bg_probs, 0)
    probs = torch.cat([bg_probs[None], probs], dim=1)
    return probs