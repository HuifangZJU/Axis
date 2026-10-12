"""Datasets of (fixed, moving) sample pairs for transition-field learning.

:class:`PairDataset` is the in-memory dataset used by :func:`axis.tl.fit_transition`.
The remaining classes read the on-disk ``network_input`` layout produced by the
preprocessing scripts in ``reproducibility/pipelines/preprocessing``::

    {root}/network_input/{group}_{sample}_image[_{size}].png      morphology canvas
    {root}/network_input/{group}_{sample}_pca_out[_{size}].npy    pseudo-image (h, w, C)
    {root}/network_input/{group}_{sample}_pca_mask[_{size}].npy   molecular support (h, w)
    {root}/network_input/{group}_{sample}_validation[_{size}].npz coord (N, 2), label (N,)

All roots default to locations below ``AXIS_DATA_DIR`` (see :mod:`axis.settings`).
"""

from __future__ import annotations

import os
from itertools import permutations
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..settings import data_path

__all__ = [
    "PairDataset",
    "cat_mask_with_features",
    "default_image_transform",
    "default_gene_transform",
    "Image_pca_pair",
    "Image_svg_pair",
    "Image_gene_pair",
    "Image_pair",
    "scc_ablation_pair",
    "Sennet_image_pair",
    "phenocycler_image_pair",
    "phenocycler_omics_pair",
    "augment_image",
]


def cat_mask_with_features(feature: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Append the molecular-support mask as the last channel of a pseudo-image."""
    return np.concatenate([feature, mask[..., np.newaxis]], axis=-1)


def default_image_transform(mode: str = "RGB", size: Optional[int] = None):
    """``ToTensor`` + normalisation to [-1, 1] (optionally resizing first)."""
    import torchvision.transforms as transforms

    steps = [transforms.Resize((size, size))] if size else []
    n = 3 if mode == "RGB" else 1
    steps += [transforms.ToTensor(), transforms.Normalize(mean=(0.5,) * n, std=(0.5,) * n)]
    return transforms.Compose(steps)


def default_gene_transform():
    import torchvision.transforms as transforms

    return transforms.Compose([transforms.ToTensor()])


def _image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """``(H, W[, C])`` uint8/float image -> ``[C, H, W]`` float tensor in [-1, 1]."""
    img = np.asarray(image)
    if img.ndim == 2:
        img = img[..., None]
    img = img.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    return torch.from_numpy((img - 0.5) / 0.5).permute(2, 0, 1).contiguous()


def _gene_to_tensor(pseudo: np.ndarray, mask: np.ndarray) -> torch.Tensor:
    combo = cat_mask_with_features(np.asarray(pseudo, dtype=np.float32), np.asarray(mask, dtype=np.float32))
    return torch.from_numpy(combo).permute(2, 0, 1).contiguous()


class PairDataset(Dataset):
    """All ordered pairs of an in-memory set of samples.

    Parameters
    ----------
    images
        Morphology canvases, ``(H, W)`` or ``(H, W, C)``; all the same size.
    pseudoimages
        Pseudo-images ``(H/8, W/8, C_p)``.
    masks
        Molecular-support masks ``(H/8, W/8)``.
    pairs
        Ordered ``(fixed, moving)`` index pairs. Defaults to all ordered pairs.
    coords, labels
        Optional per-sample spot coordinates / labels returned in ``"val"`` mode.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        pseudoimages: Sequence[np.ndarray],
        masks: Sequence[np.ndarray],
        pairs: Optional[Iterable[Tuple[int, int]]] = None,
        coords: Optional[Sequence[np.ndarray]] = None,
        labels: Optional[Sequence[np.ndarray]] = None,
        mode: str = "train",
    ):
        if not (len(images) == len(pseudoimages) == len(masks)):
            raise ValueError("images, pseudoimages and masks must have the same length")
        self.images = [_image_to_tensor(im) for im in images]
        self.genes = [_gene_to_tensor(p, m) for p, m in zip(pseudoimages, masks)]
        self.pairs = list(pairs) if pairs is not None else list(permutations(range(len(images)), 2))
        self.coords = coords
        self.labels = labels
        self.mode = mode

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        f, m = self.pairs[index % len(self.pairs)]
        out = {
            "data_pair": (f, m),
            "fixed_image": self.images[f],
            "fixed_gene": self.genes[f],
            "moving_image": self.images[m],
            "moving_gene": self.genes[m],
        }
        if self.mode == "train" or self.coords is None:
            return out
        val = {
            "fixed_coor": torch.as_tensor(np.asarray(self.coords[f], dtype=np.float32)),
            "moving_coor": torch.as_tensor(np.asarray(self.coords[m], dtype=np.float32)),
            "fixed_label": torch.as_tensor(np.asarray(self.labels[f])) if self.labels is not None else torch.zeros(0),
            "moving_label": torch.as_tensor(np.asarray(self.labels[m])) if self.labels is not None else torch.zeros(0),
        }
        return out, val


# =============================================================================
# On-disk datasets used by the reproducibility scripts
# =============================================================================
def _load_validation(prefix: str) -> Tuple[np.ndarray, np.ndarray]:
    val = np.load(prefix)
    return val["coord"], val["label"]


def _val_set(fixed_npz: str, moving_npz: str) -> Dict[str, np.ndarray]:
    fc, fl = _load_validation(fixed_npz)
    mc, ml = _load_validation(moving_npz)
    return {"fixed_coor": fc, "fixed_label": fl, "moving_coor": mc, "moving_label": ml}


def _apply(transform: Optional[Callable], value):
    return transform(value) if transform is not None else value


class Image_pca_pair(Dataset):
    """All ordered pairs of one sample group in ``{data_dir}/network_input``."""

    def __init__(
        self, data_dir, group_id, sample_id, base_size=None, image_transformer=None, gene_transformer=None, mode="train"
    ):
        super().__init__()
        self.group_id = group_id
        self.mode = mode
        self.data_dir = os.path.join(data_dir, "network_input")
        self.sample_id = sample_id
        self.base_size = base_size
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        self.data = [[i, j] for i in sample_id for j in sample_id if i != j]

    def __getitem__(self, index):
        data_pair = self.data[index % len(self.data)]
        fixed_prefix = f"{self.data_dir}/{self.group_id}_{data_pair[0]}"
        moving_prefix = f"{self.data_dir}/{self.group_id}_{data_pair[1]}"
        suffix = f"_{self.base_size}" if self.base_size is not None else ""

        def load(prefix):
            image = Image.open(f"{prefix}_image{suffix}.png").convert("RGB")
            gene = cat_mask_with_features(np.load(f"{prefix}_pca_out{suffix}.npy"), np.load(f"{prefix}_pca_mask{suffix}.npy"))
            return _apply(self.image_transformer, image), _apply(self.gene_transformer, gene)

        fixed_image, fixed_gene = load(fixed_prefix)
        moving_image, moving_gene = load(moving_prefix)
        out = {
            "data_pair": data_pair,
            "fixed_image": fixed_image,
            "fixed_gene": fixed_gene,
            "fixed_hires_image": f"{fixed_prefix}_hires_image{suffix}.jpg",
            "moving_image": moving_image,
            "moving_gene": moving_gene,
            "moving_hires_image": f"{moving_prefix}_hires_image{suffix}.jpg",
        }
        if self.mode == "train":
            return out
        return out, _val_set(f"{fixed_prefix}_validation{suffix}.npz", f"{moving_prefix}_validation{suffix}.npz")

    def __len__(self):
        return len(self.data)


class Image_svg_pair(Dataset):
    """Pairs listed in a text file (``fixed_prefix moving_prefix`` per line)."""

    def __init__(
        self,
        datalist="train_data_list.txt",
        base_size=512,
        image_transformer=None,
        gene_transformer=None,
        mode="train",
        root=None,
    ):
        super().__init__()
        self.root = root if root is not None else data_path("")
        self.mode = mode
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        with open(os.path.join(self.root, datalist)) as handle:
            self.data = handle.readlines()
        self.base_size = base_size

    def __getitem__(self, index):
        fixed, moving = self.data[index % len(self.data)].strip().split(" ")
        s = str(self.base_size)

        def load(name):
            image = Image.open(f"{self.root}{name}_image_{s}.png")
            gene = cat_mask_with_features(
                np.load(f"{self.root}{name}_pca_out_{s}.npy"), np.load(f"{self.root}{name}_pca_mask_{s}.npy")
            )
            return _apply(self.image_transformer, image), _apply(self.gene_transformer, gene)

        fixed_image, fixed_gene = load(fixed)
        moving_image, moving_gene = load(moving)
        out = {"fixed_image": fixed_image, "fixed_gene": fixed_gene, "moving_image": moving_image, "moving_gene": moving_gene}
        if self.mode == "train":
            return out
        return out, _val_set(f"{self.root}{fixed}_validation_{s}.npz", f"{self.root}{moving}_validation_{s}.npz")

    def __len__(self):
        return len(self.data)


class Image_gene_pair(Dataset):
    """Pairs listed in a text file, without the molecular-support mask."""

    def __init__(self, datalist="train_data_list.txt", image_transformer=None, gene_transformer=None, mode="train", root=None):
        super().__init__()
        self.root = root if root is not None else data_path("")
        self.mode = mode
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        with open(os.path.join(self.root, datalist)) as handle:
            self.data = handle.readlines()

    def __getitem__(self, index):
        fixed, moving = self.data[index % len(self.data)].strip().split(" ")
        out = {
            "fixed_image": _apply(self.image_transformer, Image.open(self.root + fixed + "_image.png").convert("RGB")),
            "fixed_gene": _apply(self.gene_transformer, np.load(self.root + fixed + "_pca_out.npy")),
            "moving_image": _apply(self.image_transformer, Image.open(self.root + moving + "_image.png").convert("RGB")),
            "moving_gene": _apply(self.gene_transformer, np.load(self.root + moving + "_pca_out.npy")),
        }
        if self.mode == "train":
            return out
        return out, _val_set(self.root + fixed + "_validation.npz", self.root + moving + "_validation.npz")

    def __len__(self):
        return len(self.data)


class Image_pair(Dataset):
    """Morphology-only pairs listed in a text file (image pre-training)."""

    def __init__(self, datalist="train_data_list.txt", image_transformer=None, root=None):
        super().__init__()
        self.root = root if root is not None else data_path("")
        self.image_transformer = image_transformer
        with open(os.path.join(self.root, datalist)) as handle:
            self.data = handle.readlines()

    def __getitem__(self, index):
        fixed, moving = self.data[index % len(self.data)].strip().split(" ")
        return {
            "fixed_image": _apply(self.image_transformer, Image.open(self.root + fixed + "_image_512.png").convert("RGB")),
            "moving_image": _apply(self.image_transformer, Image.open(self.root + moving + "_image_512.png").convert("RGB")),
        }

    def __len__(self):
        return len(self.data)


class scc_ablation_pair(Dataset):
    """SCC morphology-ablation pairs (Fig. 2j), ``{root}/{type}/{level}/patient_{p}_{i}_*``."""

    def __init__(
        self,
        ablation_type="image",
        ablation_level=1,
        patient=2,
        image_transformer=None,
        gene_transformer=None,
        mode="train",
        root=None,
        val_pairs=((0, 1), (0, 2)),
    ):
        super().__init__()
        self.root = root if root is not None else data_path("SCC/huifang/ablation/")
        self.ablation_type = ablation_type
        self.ablation_level = ablation_level
        self.patient = patient
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        self.mode = mode
        self.data = [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]] if mode == "train" else [list(p) for p in val_pairs]

    def __getitem__(self, index):
        data_pair = self.data[index % len(self.data)]
        base = os.path.join(self.root, self.ablation_type, str(self.ablation_level), f"patient_{self.patient}_")
        fixed_prefix, moving_prefix = base + str(data_pair[0]), base + str(data_pair[1])

        def load(prefix):
            image = Image.open(prefix + "_image_512.png").convert("RGB")
            gene = cat_mask_with_features(np.load(prefix + "_pca_out.npy"), np.load(prefix + "_pca_mask.npy"))
            return _apply(self.image_transformer, image), _apply(self.gene_transformer, gene)

        fixed_image, fixed_gene = load(fixed_prefix)
        moving_image, moving_gene = load(moving_prefix)
        out = {"fixed_image": fixed_image, "fixed_gene": fixed_gene, "moving_image": moving_image, "moving_gene": moving_gene}
        if self.mode == "train":
            return out
        return out, _val_set(fixed_prefix + "_validation.npz", moving_prefix + "_validation.npz")

    def __len__(self):
        return len(self.data)


class Sennet_image_pair(Dataset):
    """Xenium/CODEX morphology pairs (exploratory; not used in the manuscript)."""

    def __init__(self, datalist="train_data_list.txt", image_transformer=None, root=None):
        super().__init__()
        self.root = root if root is not None else data_path("sennet/hf_aligned_data/")
        self.image_transformer = image_transformer
        with open(datalist) as handle:
            self.data = handle.readlines()

    def __getitem__(self, index):
        xs, xr, cs, cr = self.data[index % len(self.data)].strip().split(" ")
        fixed = Image.open(os.path.join(self.root, f"xenium_{xs}_{xr}_enhanced.png"))
        moving = Image.open(os.path.join(self.root, f"codex_{cs}_{cr}_enhanced.png"))
        return {"fixed_image": _apply(self.image_transformer, fixed), "moving_image": _apply(self.image_transformer, moving)}

    def __len__(self):
        return len(self.data)


def augment_image(image):
    """Mild random scale / rotation / brightness / blur augmentation."""
    import imgaug.augmenters as iaa

    image = np.asarray(image).astype(np.uint8)
    h, w = image.shape[:2]
    seq = iaa.Sequential(
        [
            iaa.Resize({"height": h, "width": w}),
            iaa.Affine(scale=(0.9, 1.0), rotate=(-10, 10)),
            iaa.Multiply((0.9, 1.1)),
            iaa.GaussianBlur(sigma=(0, 1.0)),
        ],
        random_order=False,
    )
    return Image.fromarray(seq(images=[image])[0])


def _phenocycler_image_name(prefix: str, modality: str) -> str:
    return prefix + ("_GRAY_trans_enhanced.png" if modality == "visium" else "_DAPI_trans_enhanced.png")


class phenocycler_image_pair(Dataset):
    """Morphology pairs across PhenoCycler / Xenium / Visium (image pre-training)."""

    def __init__(self, datalist="train_data_list.txt", image_transformer=None, mode="train", root=None):
        super().__init__()
        self.root = root if root is not None else data_path("phenocycler/huifang/network_input/")
        self.image_transformer = image_transformer
        with open(datalist) as handle:
            self.data = handle.readlines()
        self.mode = mode

    def __getitem__(self, index):
        data_name, fixed_mod, moving_mod = self.data[index % len(self.data)].strip().split(" ")
        fixed = Image.open(_phenocycler_image_name(os.path.join(self.root, f"{data_name}_{fixed_mod}"), fixed_mod))
        moving = Image.open(_phenocycler_image_name(os.path.join(self.root, f"{data_name}_{moving_mod}"), moving_mod))
        return {"fixed_image": _apply(self.image_transformer, fixed), "moving_image": _apply(self.image_transformer, moving)}

    def __len__(self):
        return len(self.data)


class phenocycler_omics_pair(Dataset):
    """Cross-modality pairs (protein / Xenium / Visium) of one tissue.

    Parameters
    ----------
    data_dir
        Dataset root; files are read from ``{data_dir}/network_input``. Defaults
        to ``$AXIS_DATA_DIR/phenocycler/huifang``.
    val_suffix
        Validation file suffix (``"_validation_full.npz"`` for inference,
        ``"_validation.npz"`` for the training-time monitor).
    """

    _ALL_PAIRS = [
        ["protein", "visium"],
        ["protein", "xenium"],
        ["xenium", "visium"],
        ["xenium", "protein"],
        ["visium", "protein"],
        ["visium", "xenium"],
    ]

    def __init__(
        self,
        data_dir=None,
        data_name="LUAD_2_A",
        use_modalities=("protein", "xenium", "visium"),
        image_transformer=None,
        gene_transformer=None,
        mode="train",
        val_suffix="_validation_full.npz",
    ):
        super().__init__()
        data_dir = data_dir if data_dir is not None else data_path("phenocycler/huifang")
        self.data_dir = os.path.join(data_dir, "network_input")
        self.image_transformer = image_transformer
        self.gene_transformer = gene_transformer
        self.mode = mode
        self.data_name = data_name
        self.val_suffix = val_suffix
        self.data = [p for p in self._ALL_PAIRS if all(m in use_modalities for m in p)]

    def __getitem__(self, index):
        fixed_mod, moving_mod = self.data[index % len(self.data)]
        fixed_prefix = f"{self.data_dir}/{self.data_name}_{fixed_mod}"
        moving_prefix = f"{self.data_dir}/{self.data_name}_{moving_mod}"

        def load(prefix, modality):
            image = Image.open(_phenocycler_image_name(prefix, modality))
            gene = cat_mask_with_features(np.load(prefix + "_feature_out.npy"), np.load(prefix + "_feature_mask.npy"))
            return _apply(self.image_transformer, image), _apply(self.gene_transformer, gene)

        fixed_image, fixed_gene = load(fixed_prefix, fixed_mod)
        moving_image, moving_gene = load(moving_prefix, moving_mod)
        out = {
            "data_pair": [fixed_mod, moving_mod],
            "fixed_image": fixed_image,
            "fixed_gene": fixed_gene,
            "moving_image": moving_image,
            "moving_gene": moving_gene,
        }
        if self.mode == "train":
            return out
        return out, _val_set(fixed_prefix + self.val_suffix, moving_prefix + self.val_suffix)

    def __len__(self):
        return len(self.data)
