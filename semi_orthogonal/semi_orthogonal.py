from abc import abstractmethod
from typing import Union, Tuple

import torch
from torch import Tensor, device as Device
from torch.utils.data import DataLoader

from semi_orthogonal.utils import embeddings_concat
from semi_orthogonal.utils.distance import mahalanobis_sq
from semi_orthogonal.backbones import ResNet18, WideResNet50


def _generate_W(F: int, k: int, device: str):
    omega = torch.randn((k, F), device=device)
    q, r = torch.linalg.qr(omega)
    W = q @ torch.sign(torch.diag(torch.diag(r)))
    return W


class SemiOrthogonal:
    def __init__(self,
                 k: int,
                 device: Union[str, Device],
                 backbone: str,
                 size: Tuple[int, int] = (256, 256)):
        self.device = device
        self._init_backbone_with_size(backbone, size)
        self.num_embeddings = self.max_embeddings_size
        self.real_means, self.covs_inv = None, None
        self.k = k
        self.W = _generate_W(self.k, self.num_embeddings, device) 

        self.means = torch.zeros(
            (self.num_patches, self.k)).to(self.device)
        self.covs = torch.zeros((self.num_patches, self.k,
                                 self.k)).to(self.device)
        self.N = 0

    def _get_backbone(self):
        if isinstance(self.model, ResNet18):
            backbone = "resnet18"
        elif isinstance(self.model, WideResNet50):
            backbone = "wide_resnet50"
        else:
            raise NotImplementedError()

        return backbone

    def _init_backbone_with_size(self, backbone: str,
                                 size: Tuple[int, int]) -> None:
        if backbone == "resnet18":
            self.model = ResNet18().to(self.device)
        elif backbone == "wide_resnet50":
            self.model = WideResNet50().to(self.device)
        else:
            raise Exception(f"unknown backbone {backbone}, "
                            "choose one of ['resnet18', 'wide_resnet50']")

        empty_batch = torch.zeros((1, 3) + size, device=self.device)
        feature_1, _, _ = self.model(empty_batch)
        _, _, w, h = feature_1.shape
        num_patches = w * h
        self.model.num_patches = num_patches
        self.num_patches = num_patches
        self.max_embeddings_size = self.model.embeddings_size

    def _embed_batch(self, imgs: Tensor, with_grad: bool = False) -> Tensor:
        with torch.set_grad_enabled(with_grad):
            feature_1, feature_2, feature_3 = self.model(imgs.to(self.device))
        embeddings = embeddings_concat(feature_1, feature_2)
        embeddings = embeddings_concat(embeddings, feature_3)
        embeddings = embeddings.permute(2, 3, 1, 0)
        #                  (k, F)     (W, H, F, N)
        return torch.matmul(self.W.T, embeddings)  # tensor of size W, H, k, N

    def train(self, dataloader: DataLoader) -> None:
        for imgs in dataloader:
            self.train_one_batch(imgs)
        self.finalize_training()

    def train_one_batch(self, imgs: Tensor) -> None:
        with torch.no_grad():
            # w * h * k * b
            embeddings = self._embed_batch(imgs.to(self.device))
            b = embeddings.size(3)
            embeddings = embeddings.reshape(
                (self.num_patches, self.k, -1))  # (w * h) * k * b
            for i in range(self.num_patches):
                patch_embeddings = embeddings[i, :, :]  # k * b
                for j in range(b):
                    self.covs[i, :, :] += torch.outer(
                        patch_embeddings[:, j],
                        patch_embeddings[:, j])  # c * c
                self.means[i, :] += patch_embeddings.sum(dim=1)  # c
            self.N += b  # number of images

    def finalize_training(self):
        "compute the approx of C^-1"
        means, C = self._get_params()
        # free memory
        self.means = None
        self.covs = None
        covs_inv = torch.linalg.inv(C)

        self.covs_inv = covs_inv
        self.real_means = means

    def _get_params(self,
                   epsilon: float = 0.01) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the mean vectors and covariance matrices from the
        intermediary state
        Params
        ======
            epsilon: float - coefficient for the identity matrix
        Returns
        =======
            means: Tensor - the computed mean vectors
            covs: Tensor - the computed covariance matrices
            embedding_ids: Tensor - the embedding indices
        """
        means = self.means.detach().clone()
        covs = self.covs.detach().clone()

        identity = torch.eye(self.k).to(self.device)
        means /= self.N
        for i in range(self.num_patches):
            covs[i, :, :] -= self.N * torch.outer(means[i, :], means[i, :])
            covs[i, :, :] /= self.N - 1  # corrected covariance
            covs[i, :, :] += epsilon * identity  # constant term

        return means, covs

    def predict(self, new_imgs: Tensor) -> Tensor:
        """
        Computes the distance matrix for each image * patch
        Params
        ======
            imgs: Tensor - (b * W * H) tensor of images
        Returns
        =======
            distances: Tensor - (c * b) array of distances
        """
        means, inv_cvars = self.real_means, self.covs_inv
        if means is None or inv_cvars is None:
            raise Exception("call finalize_training() before predicting")

        embeddings = self._embed_batch(new_imgs)
        w, h, k, _ = embeddings.shape
        embeddings = embeddings.view(w * h, k)

        distances = mahalanobis_sq(embeddings, means, inv_cvars)
        # TODO: handle more than 1 image
        return torch.sqrt(distances).view(1, w, h)
