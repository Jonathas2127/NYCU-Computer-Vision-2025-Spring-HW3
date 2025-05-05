import os

# Set CUDA memory allocation strategy BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm


# Load COCO pretrained transforms
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transform = weights.transforms()


def read_maskfile(filepath):
    return imread(filepath)


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_id, 'image.tif')
        image = Image.open(img_path).convert('RGB')
        image = transform(image)

        masks, boxes, labels = [], [], []

        for i in range(1, 5):  # class1.tif ~ class4.tif
            mask_path = os.path.join(self.root_dir, img_id, f'class{i}.tif')
            if os.path.exists(mask_path):
                mask = imread(mask_path)
                for inst_id in range(1, int(mask.max()) + 1):
                    inst_mask = (mask == inst_id).astype(np.uint8)
                    pos = np.where(inst_mask > 0)
                    if pos[0].size == 0 or pos[1].size == 0:
                        continue
                    x_min, y_min = pos[1].min(), pos[0].min()
                    x_max, y_max = pos[1].max(), pos[0].max()
                    boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(inst_mask)
                    labels.append(i)

        if not masks:
            masks = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx])
        }

        return image, target


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TRAIN_DIR = '/kaggle/input/hw3-data/train'

    dataset = CustomDataset(TRAIN_DIR)
    train_loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = maskrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 5)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 5)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in tqdm(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    torch.save(model.state_dict(), 'model_final.pth')
