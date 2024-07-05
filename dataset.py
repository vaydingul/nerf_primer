import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2


class DummyCubeDataset(Dataset):
    def __init__(
        self,
        num_images,
        H,
        W,
        focal,
        output_dir=None,
        distance_min=0,
        distance_max=10,
        azimuth_list=None,
        elevation_list=None,
    ):
        self.num_images = num_images
        self.H = H
        self.W = W
        self.focal = focal
        self.transform = transforms.ToTensor()
        self.output_dir = output_dir
        self.distance_min = distance_min
        self.distance_max = distance_max

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if output_dir:
            print(f"Saving images to {output_dir}")

            self.image_file_path_list = []
            self.pose_file_path_list = []
            for k in range(num_images):
                if azimuth_list is not None:
                    # Randomly sample from azimuth list
                    azimuth = np.random.choice(azimuth_list, size=(1,))
                else:
                    azimuth = np.random.uniform(0, 360, size=(1,))
                if elevation_list is not None:
                    # Randomly sample from elevation list
                    elevation = np.random.choice(elevation_list, size=(1,))
                else:
                    elevation = np.random.uniform(0, 360, size=(1,))
                # translation = np.random.uniform(distance_min, distance_max, size=(3,))
                translation = np.array([0, 0, 5])

                # azimuth = np.array([30])
                # elevation = np.array([30])
                # translation = np.array([0, 0, 10])
                pose = np.concatenate([azimuth, elevation, translation])
                image = self._generate_cube_image(pose)
                file_path = os.path.join(output_dir, f"image_{k}.png")
                try:
                    cv2.imwrite(file_path, image)
                    self.image_file_path_list.append(file_path)
                except Exception as e:
                    print(f"Error saving image: {e}")
                pose_matrix = self._pose_to_matrix(pose)

                try:
                    file_path = os.path.join(output_dir, f"pose_{k}.npy")
                    np.save(file_path, pose_matrix)
                    self.pose_file_path_list.append(file_path)
                except Exception as e:
                    print(f"Error saving pose matrix: {e}")

                print(f"Saved image and pose for {k}th sample!")

    def _generate_cube_image(self, pose):
        # Create an empty image
        image = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        # Define the cube vertices in 3D space
        cube_vertices = np.array(
            [
                [1, 1, 1],
                [1, 1, -1],
                [1, -1, -1],
                [1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [-1, -1, -1],
                [-1, -1, 1],
            ],
            dtype=np.float32,
        )

        # Define the cube faces with corresponding colors
        cube_faces = [
            ([0, 1, 2, 3], (255, 0, 0)),  # Red
            ([4, 5, 6, 7], (0, 255, 0)),  # Green
            ([0, 1, 5, 4], (0, 0, 255)),  # Blue
            ([2, 3, 7, 6], (255, 255, 0)),  # Yellow
            ([0, 3, 7, 4], (255, 0, 255)),  # Magenta
            ([1, 2, 6, 5], (0, 255, 255)),  # Cyan
        ]

        # Camera parameters
        camera_matrix = np.array(
            [[self.focal, 0, self.W / 2], [0, self.focal, self.H / 2], [0, 0, 1]]
        )

        # Rotation and translation vectors
        rotation_matrix = self._rotation_matrix(pose[0], pose[1])
        translation_vector = pose[2:].astype(np.float32)

        # Project the 3D points to 2D
        projected_points, _ = cv2.projectPoints(
            cube_vertices,
            rotation_matrix,
            translation_vector,
            camera_matrix,
            np.zeros(5),
        )
        projected_points = projected_points.squeeze().astype(np.int32)

        # Calculate the depth of each face
        face_depths = []
        for face, color in cube_faces:
            face_depth = np.mean(
                [
                    np.dot(rotation_matrix, cube_vertices[v])[2] + translation_vector[2]
                    for v in face
                ]
            )
            face_depths.append((face_depth, face, color))

        face_depths.sort(
            reverse=True, key=lambda x: x[0]
        )  # Sort by depth, farthest first

        # Draw the cube faces from back to front
        for _, face, color in face_depths:
            pts = projected_points[face].reshape((-1, 1, 2))
            cv2.fillConvexPoly(image, pts, color)

        return image

    def _rotation_matrix(self, elevation, azimuth):
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

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):

        if self.output_dir:
            image = cv2.imread(self.image_file_path_list[idx])
            pose_matrix = np.load(self.pose_file_path_list[idx])
            if self.transform:
                image = self.transform(image)

        else:

            # Randomly generate a pose
            azimuth = np.random.uniform(0, 360, size=(1,))
            elevation = np.random.uniform(0, 360, size=(1,))
            translation = np.random.uniform(
                self.distance_min, self.distance_max, size=(3,)
            )

            # azimuth = np.array([0])
            # elevation = np.array([0])
            # translation = np.array([0, 0, 0])
            pose = np.concatenate([azimuth, elevation, translation])
            image = self._generate_cube_image(pose)
            if self.transform:
                image = self.transform(image)
            pose_matrix = self._pose_to_matrix(pose)

        return {"image": image, "pose": torch.tensor(pose_matrix).float()}

    def _pose_to_matrix(self, pose):
        azimuth, elevation, translation_vector = pose[0], pose[1], pose[2:]
        c2w = np.eye(4)
        c2w[:3, :3] = self._rotation_matrix(elevation, azimuth)
        c2w[:3, -1] = translation_vector
        return c2w


# Example usage
if __name__ == "__main__":
    output_dir = "output_images"
    dataset = DummyCubeDataset(
        num_images=100, H=1000, W=1000, focal=1500, output_dir=output_dir
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for i, sample in enumerate(dataloader):
        images = sample["image"]
        poses = sample["pose"]
        for j, img in enumerate(images):
            np_img = img.numpy().transpose(1, 2, 0) * 255
            cv2.imshow("Cube", np_img.astype(np.uint8))
            file_path = os.path.join(output_dir, f"image_{i * 2 + j}.png")
            cv2.imwrite(file_path, np_img.astype(np.uint8))  # Save the image to file
            cv2.waitKey(0)
        print(images.shape, poses.shape)
    cv2.destroyAllWindows()
