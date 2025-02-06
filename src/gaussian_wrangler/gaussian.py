from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation

import plyfile
from plyfile import PlyData, PlyElement


class GaussianSplat:
    def __init__(
        self,
        means: np.ndarray,
        harmonics: np.ndarray,
        opacities: np.ndarray,
        scales: np.ndarray,
        rotations: np.ndarray,
    ):
        self.means = means
        self.harmonics = harmonics
        self.opacities = opacities
        self.scales = scales
        self.rotations = rotations

    @classmethod
    def from_chmo(
        cls,
        covariances: np.ndarray,
        harmonics: np.ndarray,
        means: np.ndarray,
        opacities: np.ndarray,
    ):
        """
        Create a GaussianSplat object from a set of covariances, harmonics, means, and opacities.

        Args:
            covariances: A numpy array of shape (n, 3, 3) where n is the number of Gaussians.
            harmonics: A numpy array of shape (n, 3, n_harmonics) where n is the number of Gaussians and n_harmonics is the number of harmonics.
            means: A numpy array of shape (n, 3) where n is the number of Gaussians.
            opacities: A numpy array of shape (n,) where n is the number of Gaussians.
        """
        assert (
            covariances.shape[0]
            == harmonics.shape[0]
            == means.shape[0]
            == opacities.shape[0]
        )
        assert covariances.shape[1] == covariances.shape[2] == 3
        assert harmonics.shape[1] == 3
        assert means.shape[1] == 3

        # we need to convert the covariances to rotations (quaternions)
        rotations = np.zeros((covariances.shape[0], 4))
        scales = np.zeros((covariances.shape[0], 3))

        for i in range(covariances.shape[0]):
            rotations[i], scales[i] = cls.cov_to_quat(covariances[i])

        return cls(means, harmonics, opacities, scales, rotations)

    @staticmethod
    def cov_to_quat(cov: np.ndarray) -> np.ndarray:
        """
        Convert a covariance matrix to a quaternion and set fo scales.

        Args:
            cov: A numpy array of shape (3, 3).

        Returns:
            A numpy array of shape (4,) representing the quaternion.
            A numpy array of shape (3,) representing the scales.
        """
        # perform decomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        scales = np.sqrt(eigenvalues)

        # convert eigenvectors to quaternions
        rotation = Rotation.from_matrix(eigenvectors)
        quaternion = rotation.as_quat()

        return quaternion, scales

    def filter_opacity(self, threshold: float) -> None:
        """
        Filter out Gaussians with opacity below a certain threshold.

        Args:
            threshold: The threshold to filter
        """
        opacity_mask = self.opacities > threshold

        self.means = self.means[opacity_mask]
        self.harmonics = self.harmonics[opacity_mask]
        self.opacities = self.opacities[opacity_mask]
        self.scales = self.scales[opacity_mask]
        self.rotations = self.rotations[opacity_mask]

    def clip_harmonics(self, n_harmonics) -> None:
        """
        Limit the number of harmonic components used to represent each color channel.

        Args:
            n_harmonics: The number of harmonics to keep.
        """

        self.harmonics = self.harmonics[:, :, :n_harmonics]

    def rescale(self, scale: float) -> None:
        """
        Rescale the GaussianSplat object.

        Args:
            scale: The scale to apply.
        """
        self.means *= scale
        self.scales *= scale

    def to_ply(self, path) -> None:
        """
        Write the GaussianSplat object to a PLY file.

        Args:
            path: The path to write the PLY file to.
        """

        vertices = np.concatenate(
            [
                self.means,
                np.zeros((self.means.shape[0], 3)),
                self.harmonics.reshape(self.harmonics.shape[0], -1),
                self.opacities[:, None],
                self.scales,
                self.rotations,
            ],
            axis=1,
        )
        dtype = dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            *[
                (f"f_rest_{i}", "f4")
                for i in range(self.harmonics.shape[1] * self.harmonics.shape[2] - 3)
            ],
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
        ]

        vertices = np.array([tuple(v) for v in vertices], dtype=dtype)

        el = plyfile.PlyElement.describe(vertices, "vertex")

        PlyData([el]).write(path)


if __name__ == "__main__":
    import tempfile

    means = np.random.rand(10, 3)
    harmonics = np.random.rand(10, 3, 10)
    opacities = np.random.rand(10)
    scales = np.random.rand(10, 3)
    rotations = np.random.rand(10, 4)

    splat = GaussianSplat(means, harmonics, opacities, scales, rotations)

    with tempfile.NamedTemporaryFile(suffix=".ply") as f:
        splat.to_ply(f.name)
        print(f.name)
        print(f.read())
