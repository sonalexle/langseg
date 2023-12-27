# https://dl.acm.org/doi/10.1145/1186562.1015720 (paper)
# https://github.com/MoetaYuko/GrabCut (python implementation)
import torch
from torch.nn import functional as F
import igraph as ig


BLUE = [255, 0, 0]        # rectangle color
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}


class GraphCut:
    """Assumes the unary term and the pairwise term are given as torch tensors.
    The pairwise term is given as a 2D tensor of shape (HW, HW) containing all pairwise smoothness costs.
    Unlike GrabCut, we don't use GMMs, and we only have 1 iteration of graph cut.
    """

    def __init__(self, pr_fg_mask, fg_unary_term, bg_unary_term, pairwise_term):
        self.mask = pr_fg_mask.clone() # (H, W)
        self.fg_unary_term = fg_unary_term # (H, W)
        self.bg_unary_term = bg_unary_term # (H, W)
        self.pairwise_term = pairwise_term # (HW, HW)

        self.rows, self.cols = pr_fg_mask.shape
        
        self.gamma = 50  # Best gamma suggested in paper formula (5)

        self.gc_graph = None
        self.gc_graph_capacity = None           # Edge capacities
        self.gc_source = self.cols * self.rows  # "object" terminal S
        self.gc_sink = self.gc_source + 1       # "background" terminal T

    def construct_gc_graph(self):
        """Constructs the graph for the max-flow/min-cut algorithm.
        """
        bgd_indexes = torch.where(self.mask == DRAW_BG['val'])[0]
        pr_indexes = torch.where(self.mask.reshape(-1) == DRAW_PR_FG['val'])[0]

        edges = []
        self.gc_graph_capacity = []

        # t-links
        ## uncertain pixels (to be classified)
        edges.extend(
            list(zip([self.gc_source] * len(pr_indexes), pr_indexes)))
        _D = -torch.log(self.bg_unary_term).reshape(-1)[pr_indexes]
        self.gc_graph_capacity.extend(_D.tolist()) # nll(bg_prob) as cost for object links (high bg prob -> low cost -> cut these object links)
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * len(pr_indexes), pr_indexes)))
        _D = -torch.log(self.fg_unary_term.reshape(-1)[pr_indexes])
        self.gc_graph_capacity.extend(_D.tolist()) # nll(fg_prob) as cost for background links (high fg prob -> low cost -> cut these background links)
        assert len(edges) == len(self.gc_graph_capacity)

        ## sure background
        edges.extend(
            list(zip([self.gc_source] * len(bgd_indexes), bgd_indexes)))
        self.gc_graph_capacity.extend([0] * len(bgd_indexes)) # no cost for object terminal - sure background pixels
        assert len(edges) == len(self.gc_graph_capacity)

        edges.extend(
            list(zip([self.gc_sink] * len(bgd_indexes), bgd_indexes)))
        self.gc_graph_capacity.extend([9 * self.gamma] * len(bgd_indexes)) # high cost for background terminal - sure background pixels
        assert len(edges) == len(self.gc_graph_capacity)

        # n-links
        img_indexes = torch.arange(self.rows * self.cols).reshape(self.rows, self.cols)

        # left
        mask1 = img_indexes[:, 1:].reshape(-1)
        mask2 = img_indexes[:, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.pairwise_term[mask1, mask2].tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        # upleft
        mask1 = img_indexes[1:, 1:].reshape(-1)
        mask2 = img_indexes[:-1, :-1].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.pairwise_term[mask1, mask2].tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        # up
        mask1 = img_indexes[1:, :].reshape(-1)
        mask2 = img_indexes[:-1, :].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.pairwise_term[mask1, mask2].tolist())
        assert len(edges) == len(self.gc_graph_capacity)

        # upright
        mask1 = img_indexes[1:, :-1].reshape(-1)
        mask2 = img_indexes[:-1, 1:].reshape(-1)
        edges.extend(list(zip(mask1, mask2)))
        self.gc_graph_capacity.extend(self.pairwise_term[mask1, mask2].tolist())
        assert len(edges) == len(self.gc_graph_capacity)
        
        assert len(edges) == 4 * self.cols * self.rows - 3 * (self.cols + self.rows) + 2 + \
            2 * self.cols * self.rows

#         # fully connected n-links
#         mask1, mask2 = torch.tril_indices(*self.pairwise_term.shape, offset=-1)
#         edges.extend(list(zip(mask1, mask2)))
#         self.gc_graph_capacity.extend(self.pairwise_term[mask1, mask2].tolist())
#         assert len(edges) == len(self.gc_graph_capacity)
        
#         n = self.cols * self.rows
#         expected = int((n*(n - 1)) / 2 + 2 * self.cols * self.rows)
#         assert len(edges) == expected, f"expected {expected} edges but got {len(edges)} edges"

        self.gc_graph = ig.Graph(self.cols * self.rows + 2)
        self.gc_graph.add_edges(edges)

    def estimate_segmentation(self):
        """Step 3 in Figure 3: Estimate segmentation"""
        mincut = self.gc_graph.st_mincut(
            self.gc_source, self.gc_sink, self.gc_graph_capacity)
        # print('foreground pixels: %d, background pixels: %d' % (
        #     len(mincut.partition[0]), len(mincut.partition[1])))
        pr_indexes = torch.where(torch.logical_or(
            self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
        img_indexes = torch.arange(self.rows * self.cols).reshape(self.rows, self.cols)
        # print(img_indexes[pr_indexes].shape)
        self.mask[pr_indexes] = torch.where(torch.isin(img_indexes[pr_indexes], torch.tensor(mincut.partition[0])),
                                         DRAW_PR_FG['val'], DRAW_PR_BG['val'])
        return (mincut.partition[0], mincut.partition[1])

    def run(self):
        self.construct_gc_graph()
        fg_indices, bg_indices = self.estimate_segmentation()
        mincut_mask = torch.zeros(self.rows, self.rows).reshape(-1,)
        fg_indices = fg_indices[:-1]
        bg_indices = bg_indices[:-1]
        mincut_mask[fg_indices] = 1
        mincut_mask[bg_indices] = 0
        mincut_mask = mincut_mask.reshape(self.rows, self.rows)
        return self.mask, mincut_mask


class DiffusionGraphCut():

    def __init__(self, ca, sa, concept_ind: list, concepts, cat_to_label_id, bg_thresh=0.7, bg_scale=0.5):
        """
        Args:
            ca: (HW, K) concept activation
            sa: (HW, HW) spatial activation
            concept_ind: list of int, indices of concepts to do graph cut on
        """
        ca, sa = ca.squeeze(0), sa.squeeze(0)
        assert ca.ndim == 2
        assert sa.ndim == 2
        assert ca.shape[0] == sa.shape[0] == sa.shape[1], "inconsistent shapes: %s, %s, %s" % (ca.shape, sa.shape, sa.shape)

        resolution = int(ca.shape[0]**0.5)
        ca = ca.view(resolution, resolution, -1)
        self.resolution = resolution
        pr_fg_mask = ca[..., 0] <= bg_thresh # <sos> token attention map gives high activation to background
        self.bg_scale = bg_scale
        self.pr_fg_mask = torch.where(pr_fg_mask == True, DRAW_PR_FG['val'], 0)

        self.ca = ca
        self.pairwise_term = (sa + sa.T) / 2 # symmetric

        self.concept_ind = [c for c in concept_ind] # copy
        self.cat_to_label_id = cat_to_label_id
        self.concepts = concepts

    def __call__(self, output_resolution=None):
        if len(self.concept_ind) == 1 or len(self.concept_ind) == 2:
            fg_unary_term = self.ca[..., self.concept_ind[0]]
            if len(self.concept_ind) == 2:
                bg_unary_term = self.ca[..., self.concept_ind[1]]
            else:
                bg_unary_term = self.bg_scale*self.ca[..., 0] # take the probs from <sos> token but scale them down
            mask, _ = self.run_ovr(fg_unary_term, bg_unary_term)
            mask[mask == DRAW_PR_FG['val']] = self.concept_ind[0]
            mask[mask == DRAW_PR_BG['val']] = self.concept_ind[1] if len(self.concept_ind) == 2 else DRAW_BG['val']
        else:
            masks = []
            fg_masks = [] # to identify pixels with overlapping masks
            for cls in self.concept_ind:
                fg_unary_term = self.ca[..., cls]
                rest_concept_ind = [c for c in self.concept_ind if c != cls]
                bg_unary_term = self.ca[..., rest_concept_ind].mean(dim=-1)
                fg_mask, _ = self.run_ovr(fg_unary_term, bg_unary_term)
                fg_mask[fg_mask == DRAW_PR_BG['val']] = DRAW_BG['val']
                mask = fg_mask.clone()
    
                fg_mask[fg_mask == DRAW_PR_FG['val']] = DRAW_FG['val']
                fg_masks.append(fg_mask)
                mask[mask == DRAW_PR_FG['val']] = cls
                masks.append(mask)

            mask = torch.stack(masks).sum(dim=0)
            fg_mask = torch.stack(fg_masks).sum(dim=0).reshape(-1)
            overlapping_indices = torch.where(fg_mask > 1)[0]
            mask = mask.view(-1,)
            mask[overlapping_indices] = self.resolve_overlapping_masks(overlapping_indices)
            mask = mask.view(self.resolution, self.resolution)
        
        pred_mask = torch.zeros_like(mask)
        for i, c in enumerate(self.concepts): # concept_ind is aligned with concepts
            lab_id = self.cat_to_label_id[c] # label index
            pred_mask[mask == self.concept_ind[i]] = lab_id

        if output_resolution is not None:
            pred_mask = F.interpolate(
                pred_mask[None, None].float(), size=output_resolution, mode='nearest'
            ).squeeze().long()

        return pred_mask

    def run_ovr(self, fg_unary_term, bg_unary_term):
        """One-vs-rest graph cut
        Args:
            fg_unary_term: (HW) foreground unary term (current concept)
            bg_unary_term: (HW) background unary term (mean of all other concepts)
        """
        graphcut = GraphCut(self.pr_fg_mask, fg_unary_term, bg_unary_term, self.pairwise_term)
        return graphcut.run()

    def resolve_overlapping_masks(self, overlapping_indices):
        """Resolve overlapping masks by taking the argmax of the CA scores at the overlapping pixels.
        Args:
            overlapping_indices: list of int, indices of overlapping pixels
        """
        ca = self.ca.view(-1, self.ca.shape[-1])
        class_map = ca[overlapping_indices][:, self.concept_ind]
        mask = torch.zeros(len(overlapping_indices),)
        m = class_map.argmax(dim=1) # argmax over token indices in concept_ind
        # class map (b, k, h, w)
        n_concepts = len(self.concept_ind)
        for i in reversed(range(n_concepts)):
            mask[m == i] = self.concept_ind[i] # index in input_ids
        mask = torch.where(class_map.mean(dim=1) > class_map.mean(), mask, 0) # background
        return mask.long()
