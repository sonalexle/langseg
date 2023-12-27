# adapted to pytorch from https://github.com/google/diffseg/ (originally in tensorflow)
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from methods.diffusion_seg import get_agg_map
from methods.diffusion import get_attention_maps


def generate_sampling_grid(num_of_points, res=64):
    segment_len = (res-1)//(num_of_points-1)
    total_len = segment_len*(num_of_points-1)
    start_point = ((res-1) - total_len)//2
    x_new = np.linspace(start_point, total_len+start_point, num_of_points)
    y_new = np.linspace(start_point, total_len+start_point, num_of_points)
    x_new, y_new = np.meshgrid(x_new,y_new,indexing='ij')
    points = np.concatenate(([x_new.reshape(-1,1),y_new.reshape(-1,1)]),axis=-1).astype(int)
    return points


def get_weight_ratio(weight_list, inverse=False):
    # This function assigns proportional aggergation weight 
    sizes = []
    max_res = -100
    for weights in weight_list:
        size = int(weights.shape[-2] ** 0.5)
        if size > max_res:
            max_res = size
        sizes.append(size)
    if inverse:
        sizes = np.array(sizes)
        sizes = max_res / sizes
    return sizes / np.sum(sizes)


def aggregate_weights(weight_list, weight_ratio=None):
    if weight_ratio is None:
        weight_ratio = get_weight_ratio(weight_list)

    max_res = max([int(attn.shape[1]**0.5) for attn in weight_list])

    aggre_weights = torch.zeros((max_res,max_res,max_res,max_res), device=weight_list[0].device, dtype=weight_list[0].dtype)

    for index,weights in enumerate(weight_list):
        assert weights.shape[0] == 1, "batch mode not supported"
        size = int(weights.shape[-1] ** 0.5)
        ratio = int(max_res/size)
        # Average over the multi-head channel
        weights = weights.view(*weights.shape[:2], size, size)
        # Upsample the last two dimensions to 64 x 64
        weights = F.interpolate(weights, size=max_res, mode='bilinear', antialias=False, align_corners=False)
        weights = weights.view(weights.shape[0], size, size, max_res, max_res)

        # Normalize to make sure each map sums to one
        weights = weights/torch.sum(weights, dim=(-2,-1), keepdim=True)
        weights = weights.squeeze(0) # single batch

        # Spatial tiling along the first two dimensions
        weights = torch.repeat_interleave(weights, repeats=ratio, dim=0)
        weights = torch.repeat_interleave(weights, repeats=ratio, dim=1)

        # Aggrgate accroding to weight_ratio
        aggre_weights += weight_ratio[index] * weights
    return aggre_weights


def KL(x,Y):
    quotient = x.log() - Y.log()
    kl_1 = torch.sum(x * quotient, dim=(-2, -1)) / 2
    kl_2 = -torch.sum(Y * quotient, dim=(-2, -1)) / 2
    return kl_1 + kl_2


def mask_merge(it, attns, kl_threshold, grid=None):
    res = attns.shape[-1]
    if it == 0:
        # The first iteration of merging
        anchors = attns[grid[:,0], grid[:,1], :, :] # 256 x 64 x 64
        anchors = anchors.unsqueeze(1) # 256 x 1 x 64 x 64
        attns = attns.reshape(1, res**2, res, res) 
        # 256 x 4096 x 64 x 64 is too large for a single gpu, splitting into 16 portions
        split = int(grid.shape[0]**0.5)
        kl_bin = []
        for i in range(split):
            x = anchors[i*split:(i+1)*split]#.to(torch.float16)
            Y = attns#.to(torch.float16)
            temp = KL(x, Y) < kl_threshold[it] # type cast from float32 to float16
            kl_bin.append(temp)
        kl_bin = torch.cat(kl_bin, dim=0).to(attns.dtype) # 256 x 4096
        new_attns = (kl_bin @ attns.view(-1, res**2)) / kl_bin.sum(dim=1, keepdims=True)
        new_attns = new_attns.view(-1, res, res) # 256 x 64 x 64
    else:
        # The rest of merging iterations, reducing the number of masks
        matched = set()
        new_attns = []
        for i, point in enumerate(attns):
            if i in matched: continue
            matched.add(i)
            anchor = point
            kl_bin = KL(anchor,attns) < kl_threshold[it] # 64 x 64
            if kl_bin.sum() == 0:
                continue
            matched_idx = torch.arange(len(attns), device=kl_bin.device)[kl_bin.view(-1)]
            for idx in matched_idx: matched.add(idx.item())
            aggregated_attn = attns[kl_bin].mean(dim=0)
            new_attns.append(aggregated_attn.view(1, res, res))
        new_attns = torch.cat(new_attns, dim=0)
    return new_attns


def generate_masks(attns, kl_threshold, grid, out_res=512, refine=True):
    # Iterative Attention Merging
    for i in range(len(kl_threshold)):
        if i == 0:
            attns_merged = mask_merge(i, attns, kl_threshold, grid=grid)
        else:
            attns_merged = mask_merge(i, attns_merged, kl_threshold)

    # Kmeans refinement (optional for better visual consistency)
    if refine:
        res = attns.shape[-1]
        attns = attns.reshape(-1, res**2).float().cpu().numpy()
        kmeans_init = attns_merged.reshape(-1, res**2).float().cpu().numpy()
        kmeans = KMeans(n_clusters=attns_merged.shape[0], init=kmeans_init, n_init=1).fit(attns)
        clusters = kmeans.labels_
        attns_merged = []
        for i in range(len(set(clusters))):
            cluster = (i == clusters)
            attns_merged.append(attns[cluster,:].mean(0).reshape(res, res))
        attns_merged = torch.from_numpy(np.array(attns_merged))

    attns_merged = F.interpolate(attns_merged.unsqueeze(1), size=out_res, mode='bicubic', antialias=True).squeeze(1).cpu()

    # Non-Maximum Suppression
    M_final = attns_merged.argmax(dim=0)

    return M_final


def aggregate_x_weights(weight_list, out_res=None, weight_ratio=None):
    # x_weights: bsz x size**2 x 77
    # return 512 x 512 x 77
    if weight_ratio is None:
        weight_ratio = get_weight_ratio(weight_list, inverse=False)
    if out_res is None:
        out_res = max([int(attn.shape[1]**0.5) for attn in weight_list])
    seq_len = weight_list[0].shape[-1]
    aggre_weights = torch.zeros((out_res, out_res, seq_len), device=weight_list[0].device, dtype=weight_list[0].dtype)

    for index, weights in enumerate(weight_list):
        assert weights.shape[0] == 1, "batch mode not supported"
        size = int(weights.shape[-2] ** 0.5)
        ratio = int(out_res/size)
        weights = weights.view(1, size, size, -1).permute(0, 3, 1, 2)
        weights = F.interpolate(weights, size=out_res, mode="bicubic", antialias=True)
        weights = weights.permute(0, 2, 3, 1).squeeze(0)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        aggre_weights += weights*weight_ratio[index]
    return aggre_weights


def get_semantics(pred, x_weight, concept_indices, nouns, voting="majority", average_noun_tokens=False, background=True):
    # This function assigns semantic labels to masks
    # x_weight: (size, size, L)
    assert x_weight.ndim==3 and x_weight.shape[0] == x_weight.shape[1], f"expected size x size x N but got {x_weight.shape}"

    indices = []
    if background:
        indices.append(0)
        nouns = ["background"] + nouns

    for lids in concept_indices: # assign the average attention map to the first token of the label
        if average_noun_tokens:
            x_weight[..., lids[0]] = x_weight[..., lids].mean(dim=-1)
        indices.append(lids[0])

    x_weight = x_weight[..., indices] # size x size x N

    res = x_weight.shape[0]
    x_weight = x_weight.reshape(res**2, -1)

    x_weight = F.normalize(x_weight, p=2, dim=0) # Normalize the cross-attention maps spatially
    pred = pred.reshape(res*res,-1)

    label_to_mask = defaultdict(list)
    pred_cluster_ids = [i for i in pred.unique().tolist() if i != -1]
    for i in pred_cluster_ids:
        if voting == "majority":
            logits = x_weight[(pred==i).flatten(),:].to("cpu", torch.float).numpy()
            index = logits.argmax(axis=-1)
            category = nouns[int(np.median(index))]
        else:
            logit = x_weight[(pred==i).flatten(),:].mean(0)
            category = nouns[logit.argmax(dim=-1)]
        label_to_mask[category].append(i)
    return dict(label_to_mask)


def get_pred_mask(pred, label_to_mask, cat_to_label_id):
    mask = np.zeros_like(pred)
    pred_cluster_ids = pred.unique()
    if -1 in pred_cluster_ids:
        mask[pred == -1] = -1
    for label in label_to_mask.keys():
        for cluster_id in label_to_mask[label]:
            mask[pred == cluster_id] = cat_to_label_id[label]
    return mask


def semantic_mask(image, pred, label_to_mask, transform=True):
    if transform:
        image = image.transpose((1,2,0))
        min_values = np.min(image, axis=(0, 1))
        max_values = np.max(image, axis=(0, 1))
        # Perform min-max normalization
        image = (image - min_values) / (max_values - min_values)
    num_fig = len(label_to_mask)
    plt.figure(figsize=(20, 20))
    for i,label in enumerate(label_to_mask.keys()):
        ax = plt.subplot(1, num_fig, i+1)
        image = image.reshape(512*512,-1)
        bin_mask = np.zeros_like(image)
        for mask in label_to_mask[label]:
            bin_mask[(pred.reshape(512*512)==mask).flatten(),:] = 1
        ax.imshow((image*bin_mask).reshape(512,512,-1))
        ax.set_title(label,fontdict={"fontsize":30})
        ax.axis("off")


class DiffSeg:
    def __init__(self, *, kl_threshold, refine, num_points, sampl_grid_res=64, voting="mean"):
        # Generate the grid
        self.grid = generate_sampling_grid(num_points, res=sampl_grid_res)
        # Inialize other parameters 
        self.kl_threshold = np.array(kl_threshold)
        self.refine = refine
        self.voting = voting

    def forward(self, *, attention_store, concept_indices, nouns, cat_to_label_id, out_res):
        # diffseg
        weights = [v for k, v in attention_store.attention_store.items() if "attn1" in k]
        weights = aggregate_weights(weights)
        grid = torch.from_numpy(self.grid).to(weights.device)
        pred = generate_masks(weights, self.kl_threshold, grid, out_res=out_res, refine=self.refine)
        # assign labels to clusters
        weight = [v for k, v in attention_store.attention_store.items() if "attn2" in k]
        weight = aggregate_x_weights(weight, out_res=pred.shape[1])
        label_to_mask = get_semantics(pred, weight, concept_indices, nouns, voting=self.voting, average_noun_tokens=True)
        pred = get_pred_mask(pred, label_to_mask, cat_to_label_id) # (out_res, out_res)
        return pred

    def forward_v2(self, *, attention_store, concept_indices, nouns, cat_to_label_id, out_res):
        # diffseg
        weights = [v for k, v in attention_store.attention_store.items() if "attn1" in k]
        weights = aggregate_weights(weights)
        grid = torch.from_numpy(self.grid).to(weights.device)
        pred = generate_masks(weights, self.kl_threshold, grid, out_res=out_res, refine=self.refine)
        # assign labels to clusters
        ca, sa = get_attention_maps(
            attention_store.get_average_attention(),
            batch_size=1,
            label_indices=concept_indices,
            output_size=64,
            average_layers=True,
            apply_softmax=True,
            softmax_dim=-1,
            simple_average=False
        )
        # sa = weights.view(1, 4096, -1).float()
        agg_map = get_agg_map(ca, sa, walk_len=1, beta=1)
        del sa
        weight = agg_map.view(1, 64, 64, -1)
        weight = F.interpolate(weight.permute(0, 3, 1, 2), size=out_res, mode='bicubic', antialias=True).squeeze(0).permute(1, 2, 0)
        label_to_mask = get_semantics(pred, weight, concept_indices, nouns, voting=self.voting)
        pred = get_pred_mask(pred, label_to_mask, cat_to_label_id) # (out_res, out_res)
        return pred

    def forward_v3(self, *, attention_store, concept_indices, nouns, cat_to_label_id, out_res, gt):
        # labeling the ground truth, for ablation
        # here we know the ground truth class names (from cross-attention) and ground truth clusters
        # all that's left is to correctly assign each cluster with the right class name (from the gt set)
        ca, sa = get_attention_maps(
            attention_store.get_average_attention(),
            batch_size=1,
            label_indices=concept_indices,
            output_size=64,
            average_layers=True,
            apply_softmax=True,
            softmax_dim=-1,
            simple_average=False
        )
        # sa = weights.view(1, 4096, -1).float()
        agg_map = get_agg_map(ca, sa, walk_len=1, beta=1)
        weight = agg_map.view(1, 64, 64, -1)
        weight = F.interpolate(weight.permute(0, 3, 1, 2), size=out_res, mode='bicubic', antialias=True).squeeze(0).permute(1, 2, 0)
        pred = gt.squeeze(0) # ground truth mask (clusters)
        label_to_mask = get_semantics(pred, weight, concept_indices, nouns, voting=self.voting)
        pred = get_pred_mask(pred, label_to_mask, cat_to_label_id) # (out_res, out_res)
        return pred