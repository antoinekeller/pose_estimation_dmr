import torch
import numpy as np
from tqdm import tqdm
import cv2
from pytorch3d.io import load_objs_as_meshes

# Data structures and functions for rendering
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)

from model import Model
from lie_groups import exp_map_SO3


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


assert torch.cuda.is_available()

device = "cuda"

mesh = load_objs_as_meshes(
    ["data/mesh/mesh.obj"],
    device=device,
)

width = 960
height = 539
vfov = 39.31968

init_alphas = torch.tensor([1.0, 1.0, 1.0], device=device)
init_betas = torch.tensor([0.0, 0.0, 0.0], device=device)
init_trans = torch.tensor([[-0.961, -0.655, 4.009]], device=device)
init_log_rot = torch.tensor([[0.022, -0.375, -0.016]], device=device)

# Initialize a camera.
R = exp_map_SO3(init_log_rot)
cameras = FoVPerspectiveCameras(fov=vfov, R=R, T=init_trans, device=device)

# Create rendering pipeline
raster_settings = RasterizationSettings(
    image_size=(height, width),
)

lights = AmbientLights(ambient_color=[1.0, 1.0, 1.0], device=device)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights),
)

NB_EPOCHS = 300

cap = cv2.VideoCapture("data/castle_video.mp4")

idx = 0


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if idx % 4 != 0:
        idx += 1
        continue

    print(f"Frame {idx}")

    orig_image = cv2.resize(frame, (width, height))

    image_reference = orig_image[::-1, ::-1, ::-1] / 255.0

    # Initialize a model using the renderer, mesh and reference image
    model = Model(
        meshes=mesh,
        renderer=renderer,
        image_ref=image_reference,
        init_trans=init_trans.clone().detach(),
        init_log_rot=init_log_rot.clone().detach(),
        init_alphas=init_alphas.clone().detach(),
        init_betas=init_betas.clone().detach(),
    ).to(device)

    early_stopping = EarlyStopping(patience=25, verbose=False, delta=1e-5)

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    optimizer = torch.optim.Adam(
        [
            {"params": model.camera_trans, "lr": 0.001},
            {"params": model.camera_rot, "lr": 0.001},
            {"params": model.alphas, "lr": 0.001},
            {"params": model.betas, "lr": 0.001},
        ],
        lr=0.001,
    )

    best_position = model.camera_trans[0]
    best_log_rot = model.camera_rot[0]

    loss, image = model()
    init_loss = loss.item()
    print(f"Init loss = {init_loss:.6f}")
    best_loss = init_loss

    t = tqdm(range(NB_EPOCHS))
    for i in t:
        optimizer.zero_grad()
        loss, image = model()
        loss.backward()
        optimizer.step()

        early_stopping(loss)
        if early_stopping.early_stop:
            break

        position = model.camera_trans[0].detach().cpu().numpy()
        log_rot = model.camera_rot[0].detach().cpu().numpy()
        alpha = model.alphas.detach().cpu().numpy()
        beta = model.betas.detach().cpu().numpy()

        t.set_description(
            f"Optimizing (loss {loss.data:.6f}) "
            f"alpha = [{alpha[0]:.2f} {alpha[1]:.2f} {alpha[2]:.2f}] "
            f"beta = [{beta[0]:.2f} {beta[1]:.2f} {beta[2]:.2f}] "
            f"{position[0]:.3f} {position[1].item():.3f} {position[2].item():.3f} "
            f"{log_rot[0]:.3f} {log_rot[1].item():.3f} {log_rot[2].item():.3f}"
        )

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_alphas = model.alphas.clone().detach()
            best_betas = model.betas.clone().detach()
            best_position = model.camera_trans.clone().detach()
            best_log_rot = model.camera_rot.clone().detach()

        R = exp_map_SO3(model.camera_rot)

        image_np = image.detach().cpu().numpy()
        mask = image_np[::-1, ::-1, 0] < 1

        cv2.imshow("Iter", image_np[::-1, ::-1, ::-1])

        image_overlay = np.zeros_like(orig_image)
        image_overlay[mask] = [0, 0, 255]
        alpha = 0.3
        img = cv2.addWeighted(orig_image, 1 - alpha, image_overlay, alpha, 0)
        cv2.imshow("Image", img)
        k = cv2.waitKey(1)

        # cv2.imwrite(f"reloc_example/images/frame_{i:04d}.png", img)
        # cv2.imwrite(
        #     f"reloc_example/mesh/frame_{i:04d}.png",
        #     (image_np[::-1, ::-1, ::-1] * 255).astype(np.uint8),
        # )

        if k == 27:
            break

    # dist = np.linalg.norm((model.camera_trans - init_trans).detach().cpu().numpy())
    # print(f"Distance: {dist:.4f}")

    # Set next initial position
    init_trans = best_position
    init_log_rot = best_log_rot
    init_alphas = best_alphas
    init_betas = best_betas

    print(f"Best loss: {best_loss:.6f}")

    cv2.waitKey(1)

    # cv2.imwrite(f"output_test_1/images/frame_{idx:04d}.png", img)
    # cv2.imwrite(
    #     f"output_test_1/mesh/frame_{idx:04d}.png",
    #     (image.detach().cpu().numpy()[::-1, ::-1, ::-1] * 255).astype(np.uint8),
    # )

    if k == 27:
        break

    idx += 1


cap.release()
