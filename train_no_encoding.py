    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from model import NeRF
    from dataset import DummyCubeDataset
    from train_utils import *
    from torch.utils.data import DataLoader
    import cv2
    import os
    import numpy as np

    if __name__ == "__main__":
        # Example usage

        seed_everything(42)

        torch.mps.empty_cache()

        H = 64
        W = 64
        FOCAL = 64
        NUM_SAMPLES = 16
        NEAR = -0.1
        FAR = -11.0
        D = 16
        L = 0
        dataset = DummyCubeDataset(
            num_images=5,
            H=H,
            W=W,
            focal=FOCAL,
            output_dir="dataset_images",
            distance_min=0.0,
            distance_max=5.0,
            azimuth_list=[48, 49, 51, 52],
            elevation_list=[50],
        )
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
        )

        device = "mps"  #  if torch.cuda.is_available() else "cpu"

        model = NeRF(D=D, L=L)
        model.to(device)

        INFERENCE_FOLDER = f"inference_{H}_{W}_{FOCAL}_{NUM_SAMPLES}_{NEAR}_{FAR}_{D}_{L}"
        if not os.path.exists(INFERENCE_FOLDER):
            os.makedirs(INFERENCE_FOLDER)

        CHECKPOINT_FOLDER = (
            f"checkpoints_{H}_{W}_{FOCAL}_{NUM_SAMPLES}_{NEAR}_{FAR}_{D}_{L}"
        )
        if not os.path.exists(CHECKPOINT_FOLDER):
            os.makedirs(CHECKPOINT_FOLDER)

        # Initialize and train the model

        train_nerf(
            model,
            dataloader,
            epochs=200,
            H=H,
            W=W,
            focal=FOCAL,
            num_samples=NUM_SAMPLES,
            near=NEAR,
            far=FAR,
            inference_folder=INFERENCE_FOLDER,
            checkpoint_folder=CHECKPOINT_FOLDER,
            device=device,
        )

        visualize(
            model,
            H,
            W,
            FOCAL,
            NUM_SAMPLES,
            NEAR,
            FAR,
            INFERENCE_FOLDER,
            azimuth_list=np.linspace(48, 52, 50),
            device=device,
        )
