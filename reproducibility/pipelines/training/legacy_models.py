"""Model classes with the exact interface used by the original training scripts.

``NICE_Trans`` reproduces the former ``hf_networks.NICE_Trans``: the AXIS
multimodal network with convolution-based fusion whose forward pass returns the
2-channel ``(dy, dx)`` field. Checkpoints are interchangeable with
:class:`axis.registration.TransitionFieldNet` (``fusion_module="linear"``).
"""

from axis.registration import ImageTransitionNet as NICE_Trans_img  # noqa: F401  (former hf_img_networks.NICE_Trans_img)
from axis.registration import TransitionFieldNet


class NICE_Trans(TransitionFieldNet):
    def __init__(self, *args, fusion_module: str = "linear", **kwargs):
        super().__init__(*args, fusion_module=fusion_module, **kwargs)

    def forward(self, fixed_image, fixed_gene, moving_image, moving_gene):
        outputs = list(super().forward(fixed_image, fixed_gene, moving_image, moving_gene))
        outputs[2] = outputs[2][:, 1:3]
        return tuple(outputs)
