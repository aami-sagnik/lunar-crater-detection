import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import polygon
import json
from src.utils import visualize_detections_masks, visualize_detections_targets

MEAN=(0.485, 0.456, 0.406)
STD=(0.229, 0.224, 0.225)

class LunarCraterDataset(Dataset):
    image_transforms = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=MEAN,
            std=STD
        )
    ])

    def __init__(self, source_h, source_w, dataset_path, masks=False, n_att=None):
        self.dataset_path = dataset_path
        self.n_att = n_att
        self.source_h = source_h
        self.source_w = source_w
        self.resize_transform = None
        self.images_path = os.path.join(dataset_path, "images")
        self.targets_path = os.path.join(dataset_path, "annotations")
        self.masks = masks

        # Collect all PNG image filenames in the dataset directory
        self.png_files = [f for f in os.listdir(self.images_path) if f.endswith(".png") and os.path.isfile(os.path.join(self.images_path, f))]

    def __len__(self):
        return len(self.png_files)

    def __getitem__(self, idx):
        # Lazy load image from disk
        img_path = os.path.join(self.images_path, self.png_files[idx])
        image = Image.open(img_path) 
        image = self.image_transforms(image)

        # Lazy load corresponding masks from JSON
        json_file_base = self.png_files[idx].split(".png")[0]
        if self.masks:
            masks, num_mask = self.get_masks_from_json(
                os.path.join(self.targets_path, f"{json_file_base}.json"),
                self.source_w, self.source_h, self.n_att
            )
            if self.resize_transform is not None:
                image = self.resize_transform(image)
                masks = self.resize_transform(masks)
            return image, masks
        
        else:
            bboxes, num_bboxes = self.get_bboxes_from_json(
                os.path.join(self.targets_path, f"{json_file_base}.json")
            )
            labels = torch.ones(num_bboxes, dtype=torch.int64)
            if num_bboxes == 0:
                bboxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            if self.resize_transform is not None:
                old_h = image.shape[1]
                old_w = image.shape[2]
                image = self.resize_transform(image)
                new_h = self.h
                new_w = self.w
                bboxes = self.transform_bboxes(bboxes, old_h, old_w, new_h, new_w)
            return image, {"boxes": bboxes, "labels": labels}

    def resize(self, h, w):
        self.h = h
        self.w = w
        self.resize_transform = transforms.Compose([
            transforms.Resize((h, w))
        ])
        
    def transform_bboxes(self, bboxes, old_h, old_w, new_h, new_w):
        scale_x = new_w / old_w
        scale_y = new_h / old_h
        bboxes[:, [0, 2]] *= scale_x  # Scale x_min and x_max
        bboxes[:, [1, 3]] *= scale_y  # Scale y_min and y_max
        return bboxes

    def __str__(self):
        return f"{len(self.png_files)} images"
    
    def view(self, index):
        image, masks_or_targets = self.__getitem__(index)
        LunarCraterDataset.view_image(image, masks_or_targets, masks=self.masks)

    @staticmethod
    def view_image(image, masks_or_targets=None, masks=False):
        denormalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(MEAN, STD)],
            std=[1/s for s in STD]
        )
        image = denormalize(image).detach().cpu()
        if masks_or_targets is not None:
            if masks:
                print("Masks tensor detected. Visualizing as masks.")
                visualize_detections_masks(image, masks_or_targets, denormalize=False)
            else:
                print("Targets dict detected. Visualizing as bounding boxes.")
                visualize_detections_targets(image, masks_or_targets, denormalize=False)

    @staticmethod
    def view_bbox(image, target, class_name="Crater"):
        denormalize = transforms.Normalize(
            mean=[-m/s for m, s in zip(MEAN, STD)], 
            std=[1/s for s in STD]
        )
        img_array = denormalize(image).cpu().numpy().transpose(1, 2, 0)
        img_array = np.clip(img_array, 0, 1)
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        H, W, C = img_array.shape

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(img_array)

        for i, (box, label) in enumerate(zip(boxes, labels)):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 5, f'{class_name} ({i})', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.7, edgecolor='none'))

        ax.set_title(f"Image Visualization ({len(boxes)} detections)")
        plt.show()

    @staticmethod
    def get_masks_from_json(json_name, w, h, n_att):
        with open(json_name) as f:
            data_dict = json.load(f)
        base_name = os.path.basename(json_name).split("_annotations.json")[0] + ".png"
        polygon_points = [e["points"] for e in data_dict.get(base_name, [])]
        num_masks = len(polygon_points)

        masks = torch.zeros((max(num_masks, n_att), w, h), dtype=torch.float)

        for i, points in enumerate(polygon_points):
            polygon_np = np.array(points)
            r = polygon_np[:, 1]
            c = polygon_np[:, 0]
            rr, cc = polygon(r, c, shape=(w, h))
            masks[i, rr, cc] = 1

        # Padding if num_masks < n_att
        return masks, num_masks

    @staticmethod
    def get_bboxes_from_json(json_path):
        with open(json_path) as f:
            data_dict = json.load(f)
        # Extract all bbox lists from the dictionary values
        bboxes_list = []
        for key in data_dict:
            bboxes_list.append(data_dict[key])
        # Convert to numpy array with shape (N, 4) where N is total number of bboxes
        bboxes_array = np.array(bboxes_list, dtype=np.float32)
        
        # Convert to PyTorch tensor
        bboxes = torch.from_numpy(bboxes_array)
        
        return bboxes, len(bboxes_array)

def get_datasets(dataset_path, dim, n_att=None): # dataset_path="./fpsnet_dataset", n_att=N_ATT
    train_dataset = LunarCraterDataset(n_att=n_att, source_h=1200, source_w=1200, dataset_path=os.path.join(dataset_path, "train"))
    test_dataset = LunarCraterDataset(n_att=n_att, source_h=1200, source_w=1200, dataset_path=os.path.join(dataset_path, "test"))
    train_dataset.resize(dim[0], dim[1])
    test_dataset.resize(dim[0], dim[1])
    return train_dataset, test_dataset
