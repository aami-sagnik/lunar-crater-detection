from dotenv import load_dotenv
import os
api_key = os.getenv("ROBOFLOW_API_KEY")

BATCH_SIZE = 16

from roboflow import Roboflow
rf = Roboflow(api_key=api_key)
project = rf.workspace("crater-zqpjg").project("crater-vrqmn")
version = project.version(1)
dataset = version.download("coco")

dataset_path = "./crater-1"

import os

for split in ["train", "valid", "test"]:
    images = os.listdir(f"{dataset_path}/{split}")
    print(f"{split}: {len(images)} images")

import torch
import json
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_file, augment=False):
        self.images_dir = images_dir
        self.augment = augment

        # Load COCO annotations JSON
        with open(annotations_file, "r") as f:
            coco = json.load(f)

        # Map image_id -> image info
        self.images = {img["id"]: img for img in coco["images"]}
        self.image_ids = list(self.images.keys())

        # Map image_id -> list of annotations
        self.annotations = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            self.annotations.setdefault(img_id, []).append(ann)

        self.categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]

        # Load image
        img_path = os.path.join(self.images_dir, image_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Get annotations for this image
        anns = self.annotations.get(image_id, [])

        coco_annotations = []
        for ann in anns:
            coco_annotations.append({
                "id":          ann["id"],
                "image_id":    image_id,
                "category_id": ann["category_id"] - 1,  # 1 → 0
                "bbox":        ann["bbox"],
                "area":        ann["area"],
                "iscrowd":     ann.get("iscrowd", 0),
            })

        target = {
            "image_id":    image_id,
            "annotations": coco_annotations,
        }

        # Augmentation on PIL image before processor
        if self.augment:
            import torchvision.transforms.functional as TF
            import random
            if random.random() > 0.5:
                image = TF.hflip(image)
                # Flip boxes too: new_x = img_w - x - w
                w_img = image.width
                for ann in target["annotations"]:
                    x, y, w, h = ann["bbox"]
                    ann["bbox"] = [w_img - x - w, y, w, h]

        return image, target

train_dataset = COCODataset(
    images_dir=f"{dataset_path}/train",
    annotations_file=f"{dataset_path}/train/_annotations.coco.json",
    augment=True
)

valid_dataset = COCODataset(
    images_dir=f"{dataset_path}/valid",
    annotations_file=f"{dataset_path}/valid/_annotations.coco.json",
    augment=False
)

test_dataset = COCODataset(
    images_dir=f"{dataset_path}/test",
    annotations_file=f"{dataset_path}/test/_annotations.coco.json",
    augment=False
)

def collate_fn(batch):
    images, targets = zip(*batch)

    # Processor handles resize, normalize, padding, and pixel_mask
    encoding = processor(
        images=list(images),
        annotations=list(targets),
        return_tensors="pt"
    )
    return encoding

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

batch = next(iter(train_loader))
print(f"pixel_values shape : {batch['pixel_values'].shape}")   # [B, 3, H, W]
print(f"pixel_mask shape   : {batch['pixel_mask'].shape}")     # [B, H, W]
print(f"labels[0] keys     : {batch['labels'][0].keys()}")
print(f"boxes (img 0)      : {batch['labels'][0]['boxes']}")   # normalised [cx,cy,w,h]
print(f"class_labels (img 0): {batch['labels'][0]['class_labels']}")

import json

def count_object_sizes(annotations_file):
    with open(annotations_file) as f:
        coco = json.load(f)

    small, medium, large = 0, 0, 0

    for ann in coco["annotations"]:
        w, h = ann["bbox"][2], ann["bbox"][3]
        area = w * h

        if area < 32**2:
            small += 1
        elif area < 96**2:
            medium += 1
        else:
            large += 1

    total = small + medium + large

    print(f"Total objects : {total}")
    print(f"Small         : {small}  ({100*small/total:.1f}%)")
    print(f"Medium        : {medium} ({100*medium/total:.1f}%)")
    print(f"Large         : {large}  ({100*large/total:.1f}%)")

    return {"small": small, "medium": medium, "large": large, "total": total}

for split in ["train", "valid", "test"]:
    print(f"\n── {split.upper()} ──")
    count_object_sizes(f"{dataset_path}/{split}/_annotations.coco.json")

from transformers import AutoModelForObjectDetection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DETR num_labels = actual classes only, NO background token
# Your dataset has 1 class (crater), so num_labels=1
NUM_CLASSES = 1

model = AutoModelForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=NUM_CLASSES,
    id2label={0: "crater"},
    label2id={"crater": 0},
    ignore_mismatched_sizes=True
)
model.to(device)

param_groups = [
    {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": 1e-5},
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": 1e-4},
]

optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

# Cosine LR decay — works better than StepLR for transformers
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50,   # set to NUM_EPOCHS below
    eta_min=1e-6
)

def train_one_epoch(model, optimizer, dataloader, device, epoch):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict   # cls, bbox, giou components

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # standard for DETR
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"(cls: {loss_dict['loss_ce'].item():.4f}, "
                  f"bbox: {loss_dict['loss_bbox'].item():.4f}, "
                  f"giou: {loss_dict['loss_giou'].item():.4f})")

    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        labels       = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

        # DETR returns loss in eval mode too when labels are passed — no train() trick needed
        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels
        )
        total_loss += outputs.loss.item()

    return total_loss / len(dataloader)

# NUM_EPOCHS = 50
# best_val_loss = float("inf")

# for epoch in range(1, NUM_EPOCHS + 1):
#     train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
#     val_loss   = evaluate(model, valid_loader, device)
#     lr_scheduler.step()

#     print(f"\nEpoch [{epoch}/{NUM_EPOCHS}] "
#           f"Train Loss: {train_loss:.4f} | "
#           f"Val Loss: {val_loss:.4f}\n")

#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(model.state_dict(), "best_detr.pth")
#         print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")

# print("Training complete!")

def _cxcywh_to_xyxy_pixels(boxes, orig_size):
    """Convert normalised [cx,cy,w,h] to pixel [x1,y1,x2,y2]."""
    h, w = orig_size[0].item(), orig_size[1].item()
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate_map(model, dataloader, device):
    model.eval()

    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5, 0.75],
        max_detection_thresholds=[1, 10, 100]
    )

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask   = batch["pixel_mask"].to(device)
        orig_sizes   = torch.stack([
            torch.tensor([t["orig_size"][0], t["orig_size"][1]]) for t in batch["labels"]
        ]).to(device)

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # Post-process converts normalised [cx,cy,w,h] → [x1,y1,x2,y2] in pixel coords
        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=orig_sizes
        )

        preds = [{
            "boxes":  r["boxes"].cpu(),
            "scores": r["scores"].cpu(),
            "labels": r["labels"].cpu(),
        } for r in results]

        gts = [{
            # labels store boxes as normalised cxcywh — convert back to xyxy pixels
            "boxes":  _cxcywh_to_xyxy_pixels(t["boxes"], t["orig_size"]).cpu(),
            "labels": t["class_labels"].cpu(),
        } for t in batch["labels"]]

        metric.update(preds, gts)

    return metric.compute()

model.load_state_dict(torch.load("best_detr.pth", map_location=device))
# results = evaluate_map(model, test_loader, device)

# metrics_to_print = {
#     "mAP@0.50:0.95 (all)"    : "map",
#     "mAP@0.50      (all)"    : "map_50",
#     "mAP@0.75      (all)"    : "map_75",
#     "mAP@0.50:0.95 (small)"  : "map_small",
#     "mAP@0.50:0.95 (medium)" : "map_medium",
#     "mAP@0.50:0.95 (large)"  : "map_large",
# }

# print("=" * 40)
# print(f"{'Metric':<30} {'Value':>6}")
# print("=" * 40)
# for label, key in metrics_to_print.items():
#     val = results.get(key, torch.tensor(float("nan"))).item()
#     print(f"{label:<30} {val:>6.4f}")
# print("=" * 40)

# ./sample_test_images directory contains test images along with their corresponding ground truth annotations in COCO format (_annotations.coco.json). Most images referenced in the _annotations.coco.json would not be present in the ./sample_test_images directory as ./sample_test_images is just a small subset of the test dataset. The _annotations.coco.json file contains annotations for all images in the test dataset, not just those in ./sample_test_images.

# Below is the code for drawing ground truth annotations on test images in the ./sample_test_images directory and saving the output images with ground truth bounding boxes drawn on them (in yellow) in the ./sample_annotated_images directory

from PIL import ImageDraw
import os
import json

sample_dir = "./sample_test_images"
out_gt_dir = "./sample_annotated_images"
os.makedirs(out_gt_dir, exist_ok=True)

# Load the test annotations containing all images
anno_file = os.path.join(sample_dir, "_annotations.coco.json")
with open(anno_file, "r") as f:
    coco_data = json.load(f)

# Create lookup dictionaries for fast access
img_name_to_id = {img["file_name"]: img["id"] for img in coco_data["images"]}
id_to_annos = {}
for ann in coco_data["annotations"]:
    id_to_annos.setdefault(ann["image_id"], []).append(ann)

# Process only the images physically present in the sample directory
for filename in os.listdir(sample_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    img_path = os.path.join(sample_dir, filename)
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    img_id = img_name_to_id.get(filename)
    if img_id is not None:
        annos = id_to_annos.get(img_id, [])
        for ann in annos:
            # COCO format is [x, y, width, height]
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=3)
            
    image.save(os.path.join(out_gt_dir, filename))

print(f"✅ Ground truth images saved to {out_gt_dir}")

# Below is the code for generating predictions on test images in the ./sample_test_images directory and saving the output images with predicted bounding boxes as well as as ground truth annotations drawn on them (different colors, red for predicted and yellow for ground truth) in the ./sample_predicted_images directory

import torch

out_pred_dir = "./sample_predicted_images"
os.makedirs(out_pred_dir, exist_ok=True)

model.eval()

for filename in os.listdir(sample_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(sample_dir, filename)
    image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # --- 1. Draw Ground Truth (Yellow) ---
    img_id = img_name_to_id.get(filename)
    if img_id is not None:
        annos = id_to_annos.get(img_id, [])
        for ann in annos:
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline="yellow", width=3)

    # --- 2. Generate and Draw Predictions (Red) ---
    # Prepare image for the model
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    # post_process expects target sizes as (height, width)
    target_sizes = torch.tensor([image.size[::-1]]).to(device) 
    
    # Filter predictions by confidence threshold (0.5)
    results = processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes, 
        threshold=0.5
    )[0]

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        # DETR post_process outputs absolute [xmin, ymin, xmax, ymax]
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline="red", width=3)
        
        # Optionally add a label/score text just above the box
        text_y = max(0, box[1] - 15)
        draw.text((box[0], text_y), f"crater {score.item():.2f}", fill="red")

    image.save(os.path.join(out_pred_dir, filename))

print(f"✅ Predicted images saved to {out_pred_dir}")