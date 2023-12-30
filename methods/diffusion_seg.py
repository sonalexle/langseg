import torch
from torch.nn import functional as F
from torch_kmeans import KMeans # for k-means on the GPU
from sklearn.cluster import spectral_clustering

from methods.diffusion import upscale_attn


def normalize_ca(ca, lab_ids=None):
    # ca: (b, h, w, k) or (b, hw, k)
    # normalize over hw
    input_ndim = ca.ndim
    bsz = ca.shape[0]
    if lab_ids is None:
        lab_ids = torch.arange(ca.shape[-1])
    ca = ca[..., lab_ids]
    if input_ndim == 4:
        ca = ca.view(bsz, -1, len(lab_ids))
    # min-max normalize spatially
    min_val = ca.min(dim=1, keepdim=True).values
    max_val = ca.max(dim=1, keepdim=True).values
    ca = (ca - torch.abs(min_val)) / (max_val - min_val)
    if input_ndim == 4:
        res = int(ca.shape[1]**0.5)
        ca = ca.view(bsz, res, res, -1)
    return ca


def get_agg_map(ca, sa, concept_ind=None, beta=1, walk_len=1, minmax_norm=True):
    # this is the random walk algorithm
    # ca: (bsz, hw, k)
    # sa: (bsz, hw, hw)
    assert ca.ndim == 3
    assert sa.ndim == 3
    # ca: probability vector, ca[i,k] = prob random walk at pixel i for each class k
    # ca = normalize_ca(ca)
    ca = ca / ca.sum(dim=-2, keepdim=True)
    sa = sa.permute(0, 2, 1) # (bsz, (h,w), hw)
    sa = torch.pow(sa, beta)
    sa = sa / sa.sum(dim=-1, keepdim=True) # sa = transition matrix (row sum = 1)
    sa = torch.linalg.matrix_power(sa, walk_len)
    agg_map = torch.bmm(sa, ca) # (bsz, hw, k)
    if minmax_norm:
        agg_map = normalize_ca(agg_map)
    return agg_map


def get_random_walk_mask(
    agg_map,
    cat_to_label_id, concept_ind, concepts,
    output_size=None
):
    # agg_map: (bsz, hw, k)
    if output_size is not None:
        agg_map = upscale_attn(agg_map, output_size, is_cross=True)
    else:
        output_size = int(agg_map.shape[1] ** 0.5)

    class_map = agg_map[..., concept_ind]
    mask = torch.zeros_like(agg_map[..., 0])
    m = class_map.argmax(dim=-1) # argmax over token indices in concept_ind
    # class map (b, hw, k)
    for i in reversed(range(len(concept_ind))):
        mask[m == i] = concept_ind[i] # index in input_ids
    zeros = torch.zeros_like(mask)

    pred_mask = torch.zeros_like(mask)
    for i, c in enumerate(concepts): # concept_ind is aligned with concepts
        lab_id = cat_to_label_id[c] # label index
        pred_mask[mask == concept_ind[i]] = lab_id

    return pred_mask.view(-1, output_size, output_size)


def run_specclust_torch(A, n_clusters=9, n_vecs=10, output_size=None, decomp="svd"):
    assert A.ndim == 3
    bsz, res = A.shape[0], int(A.shape[1]**0.5)
    if decomp == "svd":
        D1 = torch.diagflat(1/A.sum(dim=-1).sqrt()).unsqueeze(0)
        D2 = torch.diagflat(1/A.sum(dim=-2).sqrt()).unsqueeze(0)
        A = torch.bmm(D1, torch.bmm(A, D2))
        u, s, vh = torch.linalg.svd(A, full_matrices=False)
        u = torch.bmm(D1, u[..., :n_vecs]) # (bsz, hw, n_vecs) # u = torch.bmm(D2, vh[:, :n_vecs].permute(0, 2, 1))
    elif decomp == "eig":
        A = (A + A.permute(0,2,1))/2
        D = torch.diagflat(A.sum(dim=-1)).unsqueeze(0) # row sums
        eigvals, eigvecs = torch.lobpcg(A, B=D, k=n_vecs)
        u = eigvecs
    clusters = KMeans(init_method="k-means++", n_clusters=n_clusters, verbose=False)(u).labels
    clusters = clusters.view(bsz, res, res)
    if output_size is not None:
        clusters = F.interpolate(clusters.unsqueeze(1).float(), size=output_size, mode='nearest').squeeze(1)
    return clusters.long()


def run_specclust_sklearn(A, n_clusters=10, n_vecs=10, output_size=None):
    assert A.ndim == 3
    bsz, res = A.shape[0], int(A.shape[1]**0.5)
    assert bsz == 1, "does not support batching"
    A = (A + A.permute(0,2,1))/2.
    A = A.squeeze().cpu().numpy()
    clusters = spectral_clustering(A, n_clusters=n_clusters, n_components=n_vecs, assign_labels='cluster_qr')
    clusters = torch.from_numpy(clusters.reshape(bsz, res, res)).cuda()
    if output_size is not None:
        clusters = F.interpolate(clusters.unsqueeze(1).float(), size=output_size, mode='nearest').squeeze(1)
    return clusters.long()


def assign_names_to_clusters(clusters, ca, concept_ind, concepts, output_size=None, bg_thresh=0.35):
    # ca: (b, hw, k)
    # clusters: (b, h, w)
    assert ca.ndim == 3
    bsz, res = ca.shape[0], int(ca.shape[1]**0.5)
    if output_size is not None:
        nca = upscale_attn(ca, output_size, is_cross=True)
        res = output_size
    else:
        nca = ca
    nca = nca.view(bsz, res, res, -1)
    nca = normalize_ca(nca, concept_ind)
    cluster_names = {}
    for c in range(len(clusters.unique())):
        cluster_mask = torch.zeros_like(clusters)
        cluster_mask[clusters == c] = 1
        score_maps = [cluster_mask * nca[..., i] for i in range(len(concept_ind))]
        scores = torch.tensor([score_map.sum() / cluster_mask.sum() for score_map in score_maps])
        cluster_names[c] = concepts[torch.argmax(scores)]# if scores.max() > bg_thresh else "background"
    return cluster_names


def get_specclust_mask(
    ca, sa,
    cat_to_label_id: dict,
    concept_ind: list, concepts: list,
    output_size=None, bg_thresh=0.35,
    method="torch", decomp="svd"
):
    clust_method = run_specclust_torch if method == "torch" else run_specclust_sklearn
    kwargs = {"decomp": decomp} if method == "torch" else {}
    clusters = clust_method(sa, n_clusters=20, n_vecs=20, output_size=output_size, **kwargs)
    cluster_names = assign_names_to_clusters(clusters, ca, concept_ind, concepts, output_size=output_size, bg_thresh=bg_thresh)
    pred_mask = torch.zeros_like(clusters)
    for k in cluster_names.keys(): # cluster_id -> cluster_name (label name) -> label_id
        pred_mask[clusters == k] = cat_to_label_id[cluster_names[k]]
    return pred_mask