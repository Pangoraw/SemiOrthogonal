import argparse
import os
from os import path
import sys

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append("./")

from semi_orthogonal import SemiOrthogonal
from semi_orthogonal.utils import compute_pro_score

parser = argparse.ArgumentParser("SemiOrthogonal test on MVTEC")
parser.add_argument("--data_root", required=True)
parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "wide_resnet50"])
parser.add_argument("--size", default="256x256")
parser.add_argument("-k", type=int, default=100)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
size = tuple(map(int, args.size.split("x")))
semi_orthogonal = SemiOrthogonal(
    k=args.k, device=device, backbone=args.backbone, size=size)

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    )
])

train_dataset = ImageFolder(root=args.data_root + "/train",
    transform=img_transforms)

class MVTecADTestDataset(Dataset):
    def __init__(self, root, transform, mask_transform):
        self.test_dir = path.join(root, "test")
        self.ground_truth_dir = path.join(root, "ground_truth")
        self.classes = os.listdir(self.test_dir)
        self.transform = transform
        self.mask_transform = mask_transform

        self.current_class = 0
        self.current_class_idx = 0
        self.classes_files = {
            i: os.listdir(path.join(self.test_dir, cls)) 
            for i, cls in enumerate(self.classes)
        }
    
    def __getitem__(self, index):
        if self.current_class_idx == len(self.classes_files[self.current_class]):
            self.current_class_idx = 0
            self.current_class += 1

        item_file = path.join(self.classes[self.current_class], self.classes_files[self.current_class][self.current_class_idx])
        img_file = path.join(self.test_dir, item_file)

        mask_file = item_file.replace(".png", "_mask.png")
        mask_file = path.join(self.ground_truth_dir, mask_file)
        img = Image.open(img_file)
        img = img.convert("RGB")
        img = self.transform(img)

        is_good_img = self.classes[self.current_class] == "good"
        if not is_good_img:
            mask = Image.open(mask_file)
            mask = self.mask_transform(mask)
            mask[mask != 0] = 1.
        else:
            mask = torch.zeros((1,) + img.shape[1:])

        self.current_class_idx += 1
        return img, mask, int(not is_good_img)

    def __len__(self):
        return sum(len(files) for files in self.classes_files.values())


test_dataset = MVTecADTestDataset(
    root=args.data_root,
    transform=img_transforms,
    mask_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size)
    ]),
)

train_dataloader = DataLoader(
    batch_size=4,
    dataset=train_dataset,
)


print(">> Training")
for imgs, _ in tqdm(train_dataloader):
    imgs = imgs.to(device)
    semi_orthogonal.train_one_batch(imgs)
semi_orthogonal.finalize_training()

n_test_images = len(test_dataset)
print(f">> Testing on {n_test_images} images")
y_trues = []
y_scores = []

amaps = []
masks = []

limited_generator = (x for _, x in zip(range(n_test_images), test_dataset))

for i, (img, mask, label) in tqdm(enumerate(limited_generator), total=n_test_images):
    img = img.to(device)
    preds = semi_orthogonal.predict(img.unsqueeze(0))
    y_scores.append(preds.max().item())
    y_trues.append(label)

    masks.append(mask.unsqueeze(0))
    amaps.append(preds.unsqueeze(0))

gaussian_smoothing = transforms.GaussianBlur(9)

amaps = torch.cat(amaps)
amaps = F.interpolate(amaps, size, mode="bilinear", align_corners=True)
amaps = gaussian_smoothing(amaps)
amaps -= amaps.min()
amaps /= amaps.max()
masks = torch.cat(masks)

amaps = amaps.squeeze().cpu().numpy()
masks = masks.squeeze().cpu().numpy()

roc_score = roc_auc_score(y_trues, y_scores)
print(f">> ROC AUC Score = {roc_score}")

pro_score = compute_pro_score(amaps, masks) 
print(f">> PRO Score     = {pro_score}")
