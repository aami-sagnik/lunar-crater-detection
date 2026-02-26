from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
import torch
from src.utils import inflate_annots, masks_list_to_targets_list, remove_duplicates, remove_low_confidence_edge_boxes
import time

class RetinaNet(nn.Module):
    def __init__(self, device="cpu", state_dict=None):
        super(RetinaNet, self).__init__()
        self.device = device
        self.model = retinanet_resnet50_fpn_v2(
            weights_backbone=ResNet50_Weights.DEFAULT,
            num_classes=2, 
            trainable_backbone_layers=3
            ).to(self.device)

        if state_dict is not None:
            self.model.load_state_dict(state_dict)        
    
    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        return self.model(images)

    def train_model(self, train_dataloader, lr, epochs, targets_are_masks = False):
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params_to_optimize, lr=lr)

        self.train()

        for epoch in range(epochs):
            start_time = time.time()
            total_loss = 0
            
            for batch_idx, (images, targets_list) in enumerate(train_dataloader):
                images = images.to(self.device)

                if targets_are_masks:
                    targets_list = masks_list_to_targets_list(targets_list)
                
                # Move targets (List[Dict]) to the correct device
                targets_list = [{k: v.to(self.device) for k, v in t.items()} for t in targets_list]
                
                loss_dict = self(images, targets=targets_list)
                loss_detection_cls = loss_dict['classification'] # Focal Loss
                loss_detection_box = loss_dict['bbox_regression'] # Smooth L1 Loss
                
                # Total Loss
                loss = loss_detection_cls + loss_detection_box 
                
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_dataloader)

            print(f"--- Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} Time: {time.time()-start_time:.2f} seconds ---") 

        self.eval()
        print("Training Complete")

    def test_model(self, test_dataloader, targets_are_masks = False):
        evaluator = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

        with torch.inference_mode():
            self.eval()
            for batch_idx, (images, targets_list) in enumerate(test_dataloader):
                images = images.to(self.device)

                if targets_are_masks:
                    targets_list = masks_list_to_targets_list(targets_list)

                targets_on_device = [{k: v.to(self.device) for k, v in t.items()} for t in targets_list]
                detections = self(images)

                # filtering detections with score more than 0.4
                 
                final_detections = []
                
                for image, detection in zip(images, detections):
                    keep_indices = detection['scores'] > 0.1
                    filtered_detection = {
                        'boxes': detection['boxes'][keep_indices],
                        'labels': detection['labels'][keep_indices],
                        'scores': detection['scores'][keep_indices]
                    }
                    removed_duplicates = remove_duplicates(filtered_detection, iou_threshold=0)
                    inflated_annots = inflate_annots(removed_duplicates, image.shape[1], image.shape[2], -0.1, -0.1)
                    edge_cleaned_annots = remove_low_confidence_edge_boxes(inflated_annots, image.shape[1], image.shape[2], 0.2, 0.01)
                    final_detections.append(edge_cleaned_annots)

                evaluator.update(final_detections, targets_on_device)
            
            metrics = evaluator.compute()
            return metrics
