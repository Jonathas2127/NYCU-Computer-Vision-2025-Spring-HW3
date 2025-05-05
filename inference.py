import os
import json

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

# Load image ID mapping
with open('./data/test_image_name_to_ids.json', 'r') as f:
    image_id_mapping = {
        item['file_name']: item['id'] for item in json.load(f)
    }

weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transform = weights.transforms()


class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_list = sorted([
            f for f in os.listdir(root_dir) if f.lower().endswith('.tif')
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_id = self.img_list[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = imread(img_path)

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:
            image = image[..., :3]
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        image = transform(Image.fromarray(image))
        return image, img_id


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = './data/test_release'
MODEL_PATH = './model_final.pth'

test_dataset = TestDataset(TEST_DIR)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = maskrcnn_resnet50_fpn(weights=weights)
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, 5
)
model.roi_heads.mask_predictor = MaskRCNNPredictor(
    model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 5
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

results = []
threshold = 0.0

with torch.no_grad():
    for images, img_ids in tqdm(test_loader):
        image = images[0].to(device)
        filename = os.path.splitext(os.path.basename(img_ids[0]))[0] + '.tif'
        coco_image_id = image_id_mapping.get(filename)

        outputs = model([image])[0]
        scores = outputs['scores']
        masks = outputs['masks']
        boxes = outputs['boxes']
        labels = outputs['labels']

        for i in range(len(scores)):
            if scores[i] >= threshold:
                mask = (masks[i, 0].cpu().numpy() > 0.7).astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(mask))
                rle['counts'] = rle['counts'].decode('utf-8')

                x1, y1, x2, y2 = boxes[i].cpu().tolist()
                bbox = [x1, y1, x2 - x1, y2 - y1]

                results.append({
                    'image_id': coco_image_id,
                    'bbox': bbox,
                    'score': scores[i].item(),
                    'category_id': labels[i].item(),
                    'segmentation': rle
                })

with open('test-results.json', 'w') as f:
    json.dump(results, f)

print("test-results.json has been saved.")
