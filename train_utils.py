import torch
import torch.nn as nn
import torch.nn.functional as F
from model import NeRF
from dataset import DummyCubeDataset

from torch.utils.data import DataLoader
import cv2
import os
import numpy as np


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def rotation_matrix(elevation, azimuth):
    elev = np.radians(elevation)
    azim = np.radians(azimuth)
    rot_elev = np.array(
        [
            [1, 0, 0],
            [0, np.cos(elev), -np.sin(elev)],
            [0, np.sin(elev), np.cos(elev)],
        ]
    )
    rot_azim = np.array(
        [
            [np.cos(azim), -np.sin(azim), 0],
            [np.sin(azim), np.cos(azim), 0],
            [0, 0, 1],
        ]
    )
    return np.dot(rot_azim, rot_elev)


def pose_to_matrix(pose):
    azimuth, elevation, translation_vector = pose[0], pose[1], pose[2:]
    c2w = np.eye(4)
    c2w[:3, :3] = rotation_matrix(elevation, azimuth)
    c2w[:3, -1] = translation_vector
    return c2w


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="ij")
    i = i.to(c2w.device)
    j = j.to(c2w.device)
    i, j = i.t(), j.t()
    dirs = torch.stack(
        [(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1
    )

    # c2w is 4x4 homogeneous matrix
    # Convert rays_d and rays_o according to the camera pose
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def sample_points_along_rays(rays_o, rays_d, num_samples, near, far):
    t_vals = torch.linspace(near, far, num_samples).to(rays_o.device)
    t_vals = t_vals.expand(rays_o.shape[0], num_samples)
    points = rays_o[:, None, :] + t_vals[..., None] * rays_d[:, None, :]
    return points, t_vals


def render_rays(model, rays_o, rays_d, num_samples, near, far):
    points, t_vals = sample_points_along_rays(rays_o, rays_d, num_samples, near, far)
    points_flat = points.view(-1, 3)
    view_dirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    view_dirs = view_dirs[:, None, :].expand(points.shape).contiguous()
    view_dirs_flat = view_dirs.view(-1, 3)
    inputs = torch.cat([points_flat, view_dirs_flat], -1)
    raw_outputs = model(inputs)
    raw_outputs = raw_outputs.view(points.shape[0], num_samples, -1)
    sigma = F.relu(raw_outputs[..., 3])
    rgb = torch.sigmoid(raw_outputs[..., :3])
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(dists.device)], -1
    )
    alpha = 1.0 - torch.exp(-sigma * dists)
    weights = (
        alpha
        * torch.cumprod(
            torch.cat(
                [torch.ones((alpha.shape[0], 1)).to(alpha.device), 1.0 - alpha + 1e-10],
                -1,
            ),
            -1,
        )[:, :-1]
    )
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    return rgb_map


def generate_rays_and_rgb(images, poses, H, W, focal):
    all_rays_o = []
    all_rays_d = []
    all_rgb = []
    for i in range(len(images)):
        rays_o, rays_d = get_rays(H, W, focal, poses[i])
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)
        all_rgb.append(images[i].permute(1, 2, 0))

    rays_o = torch.stack(all_rays_o)
    rays_d = torch.stack(all_rays_d)
    rgbs = torch.stack(all_rgb)

    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)
    rgbs = rgbs.view(-1, 3)

    return rays_o, rays_d, rgbs


def train_nerf(
    model,
    dataloader,
    epochs=1000,
    lr=0.0001,
    H=50,
    W=50,
    focal=50,
    num_samples=32,
    near=-0.1,
    far=-11.0,
    inference_folder="inference",
    checkpoint_folder="model",
    device="cpu",
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    criterion.to(device)

    for epoch in range(epochs):

        model.train()
        losses = []

        for i, sample in enumerate(dataloader):

            images = sample["image"].to(device)
            poses = sample["pose"].to(device)

            rays_o, rays_d, rgbs = generate_rays_and_rgb(images, poses, H, W, focal)

            # Render the final color along the rays
            rgb_map = render_rays(model, rays_o, rays_d, num_samples, near, far)

            optimizer.zero_grad()

            loss = criterion(rgb_map, rgbs)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(
            f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {np.mean(losses)}"
        )

        if epoch % 10 == 0:
            torch.save(
                model.state_dict(), f"{checkpoint_folder}/nerf_epoch_{epoch}.pth"
            )
            inference_nerf(
                model,
                H,
                W,
                focal,
                epoch,
                num_samples,
                near,
                far,
                inference_folder,
                device,
            )


def inference_nerf(
    model, H, W, focal, epoch, num_samples, near, far, inference_folder, device
):
    model.eval()
    # pose = np.random.uniform(0, 360, size=(2,))
    pose = np.array([50, 50, 0, 0, 5])
    pose = pose_to_matrix(pose)
    pose = torch.Tensor(pose).to(device)

    rgb_image = render_image(model, pose, H, W, focal, num_samples, near, far)
    # Save with OpenCV
    cv2.imwrite(f"{inference_folder}/epoch_{epoch}.png", rgb_image * 255)


def render_image(model, pose, H, W, focal, num_samples, near, far):
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o.view(-1, 3)
    rays_d = rays_d.view(-1, 3)

    rgb_map = render_rays(
        model, rays_o, rays_d, num_samples=num_samples, near=near, far=far
    )
    rgb_image = rgb_map.view(H, W, 3).detach().cpu().numpy()

    # Normalize to 0,1
    rgb_image = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())

    return rgb_image


def visualize(
    model,
    H,
    W,
    focal,
    num_samples,
    near,
    far,
    inference_folder,
    azimuth_list=None,
    device="cpu",
):

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(f"{inference_folder}/video.mp4", fourcc, 5, (W, H))

    if azimuth_list is None:
        azimuth_list = np.linspace(0, 360, 100)

    for azimuth in azimuth_list:

        model.eval()
        # pose = np.random.uniform(0, 360, size=(2,))
        pose = np.array([azimuth, 50, 0, 0, 5])
        pose = pose_to_matrix(pose)
        pose = torch.Tensor(pose).to(device)

        rgb_image = render_image(model, pose, H, W, focal, num_samples, near, far)
        # Save with OpenCV
        video.write((rgb_image * 255).astype(np.uint8))
    video.release()