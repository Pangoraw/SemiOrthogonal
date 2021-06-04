import argparse
import sys

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

sys.path.append("./")

from semi_orthogonal import SemiOrthogonal

parser = argparse.ArgumentParser("SemiOrthogonal test on MVTEC")
parser.add_argument("--data_root")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
semi_orthogonal = SemiOrthogonal(
    k=300, device=device, backbone="resnet18")

img_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=True,
    )
])

train_dataset = ImageFolder(root=args.data_root + "/train",
    transform=img_transforms,
    target_transform=lambda x: 1)
test_dataset = ImageFolder(root=args.data_root + "/test",
    transform=img_transforms,
    target_transform=lambda x: int(test_dataset.class_to_idx["good"] != x))

train_dataloader = DataLoader(
    batch_size=4,
    dataset=train_dataset,
)

print(">> Training")
for imgs, _ in tqdm(train_dataloader):
    imgs = imgs.to(device)
    semi_orthogonal.train_one_batch(imgs)
semi_orthogonal.finalize_training()

print(">> Testing")
y_trues = []
y_scores = []
for img, label in tqdm(test_dataset):
    img = img.to(device)
    preds = semi_orthogonal.predict(img.unsqueeze(0))
    y_scores.append(preds.max().item())
    y_trues.append(label)

score = roc_auc_score(y_trues, y_scores)
print(f">> ROC AUC Score = {score}")
