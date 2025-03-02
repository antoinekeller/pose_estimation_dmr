import torch
import numpy as np
import cv2
import torch.nn as nn

from lie_groups import exp_map_SO3
from loss import ClippedL2Loss


class Model(nn.Module):
    def __init__(
        self,
        meshes,
        renderer,
        image_ref,
        init_trans,
        init_log_rot,
        init_alphas,
        init_betas,
    ):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        self.image_ref = torch.from_numpy(image_ref.astype(np.float32)).to(self.device)

        # Parameters
        self.camera_trans = nn.Parameter(init_trans)
        self.camera_rot = nn.Parameter(init_log_rot)
        self.alphas = nn.Parameter(init_alphas)
        self.betas = nn.Parameter(init_betas)

    def forward(self, debug=False):

        R = exp_map_SO3(self.camera_rot)

        image = self.renderer(
            meshes_world=self.meshes.clone(), R=R, T=self.camera_trans
        )[0]

        mask = image[:, :, 3] > 0 & (image[:, :, 0] > 0.1) & (image[:, :, 1] > 0.1) & (
            image[:, :, 2] > 0.1
        )
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, 3)[0]

        # assert mask is not None

        image = image[:, :, :3]

        if debug:
            cv2.imshow(
                "image",
                (
                    torch.clamp(
                        image * self.alphas + self.betas,
                        0.0,
                        1.0,
                    )
                    * mask
                    * 255.0
                )
                .detach()
                .squeeze()
                .cpu()
                .numpy()
                .astype(np.uint8)[::-1, ::-1, ::-1],
            )
            cv2.imshow(
                "ref",
                (self.image_ref * mask * 255.0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.uint8)[::-1, ::-1, ::-1],
            )
            cv2.waitKey(0)

        loss = ClippedL2Loss(0.2)(
            torch.clamp(image * self.alphas + self.betas, 0.0, 1.0) * mask,
            self.image_ref * mask,
        )

        return loss, image
