from typing import Optional
import numpy as np
import torch
from torchvision import transforms as T


# copied from https://github.com/mhamilton723/STEGO/blob/master/src/data.py
class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)

# copied from https://github.com/mhamilton723/STEGO/blob/master/src/data.py
def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([
            T.Resize(res, T.InterpolationMode.NEAREST),
            cropper,
            ToTargetTensor()
        ])
    else:
        normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return T.Compose([
            T.Resize(res, T.InterpolationMode.BICUBIC),
            cropper,
            T.ToTensor(),
            normalize
        ])


def get_dataset(
        dir_dataset: str,
        dataset_name: str,
        split: str = "val",
        resolution: int = 512,
        mask_res: int = 320,
        dense_clip_arch: Optional[str] = None,
        dir_pseudo_masks: Optional[str] = None
):
    if dataset_name == "cityscapes":
        from datasets.cityscapes import CityscapesDataset, cityscapes_categories, cityscapes_pallete

        loader_crop = None#"center"
        dataset = CityscapesDataset(
            dir_dataset=dir_dataset,
            split=split,
            transform=get_transform(res=resolution, is_label=False, crop_type=loader_crop),
            target_transform=get_transform(res=mask_res, is_label=True, crop_type=loader_crop),
            model_name=dense_clip_arch,
            dir_pseudo_masks=dir_pseudo_masks
        )
        categories = cityscapes_categories
        pallete = cityscapes_pallete

    elif dataset_name == "coco_stuff":
        from datasets.coco_stuff import COCOStuffDataset, coco_stuff_pallete, coco_stuff_categories_fine

        dataset = COCOStuffDataset(
            dir_dataset=dir_dataset,
            split=f"train10k" if split == "train" else split,
            transform=get_transform(res=resolution, is_label=False, crop_type=None),
            target_transform=get_transform(res=mask_res, is_label=True, crop_type=None),
            coarse_labels=False,
            exclude_things=False,
            model_name=dense_clip_arch,
            dir_pseudo_masks=dir_pseudo_masks
        )
        categories = coco_stuff_categories_fine  # coco_stuff_categories
        pallete = coco_stuff_pallete

    elif dataset_name == "pascal_context":
        from datasets.pascal_context import pascal_context_categories, PascalContextDataset, pascal_context_pallete

        dataset = PascalContextDataset(
            dir_dataset=dir_dataset,
            split=split,
            transform=get_transform(res=resolution, is_label=False, crop_type=None),
            target_transform=get_transform(res=mask_res, is_label=True, crop_type=None)
        )

        categories = pascal_context_categories
        pallete = pascal_context_pallete

    elif dataset_name == "kitti_step":
        from datasets.kitti_step import KittiStepDataset, kitti_step_categories, kitti_step_palette

        dataset = KittiStepDataset(
            dir_dataset=dir_dataset,
            split=split,
            model_name=dense_clip_arch,
            dir_pseudo_masks=dir_pseudo_masks
        )
        categories = kitti_step_categories
        pallete = kitti_step_palette

    elif dataset_name == "voc2012":
        from datasets.voc2012 import PascalVOC2012Dataset, voc2012_categories, voc2012_palette
        # dir: /sinergia/ozaydin/datasets/VOCdevkit/VOC2012
        dir_dataset = "/sinergia/ozaydin/datasets/"
        dataset = PascalVOC2012Dataset(
            dir_dataset=dir_dataset,
            split=split,
            transform=get_transform(res=resolution, is_label=False, crop_type=None),
            target_transform=get_transform(res=mask_res, is_label=True, crop_type=None)
        )
        categories = voc2012_categories
        pallete = voc2012_palette

    else:
        raise ValueError(dataset_name)
    return dataset, categories, pallete


voc_palette = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128],
    # 255: [255, 255, 255]
}


def colourise_mask(
        mask: np.ndarray,
        image: Optional[np.ndarray] = None,
        benchmark: str = "voc",
        opacity: float = 0.5
):
    # assert label.dtype == np.uint8
    assert len(mask.shape) == 2, ValueError(mask.shape)
    h, w = mask.shape
    grid = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = set(mask.flatten())
    if "voc" in benchmark:
        palette = list(voc_palette.values())

        # coloured_label = label2rgb(label, image=image, colors=palette, alpha=opacity)
    else:
        raise ValueError(benchmark)

    for l in unique_labels:
        try:
            grid[mask == l] = np.array(palette[l])
        except IndexError:
            print(l)

    return grid


# copied from https://github.com/mhamilton723/STEGO/blob/master/src/utils.py
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])