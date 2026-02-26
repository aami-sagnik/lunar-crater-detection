from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.resnet import ResNet50_Weights
import torch.nn as nn
import torch
from src.utils import inflate_annots, masks_list_to_targets_list, remove_duplicates, remove_low_confidence_edge_boxes
import time

class FasterRCNN(nn.Module):
    def __init__(self, device="cpu", state_dict=None):
        super(FasterRCNN, self).__init__()
        self.device = device
        self.model = fasterrcnn_resnet50_fpn_v2(
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
                loss_classifier    = loss_dict['loss_classifier']   # RoI Head Classification Loss (Cross Entropy)
                loss_box_reg       = loss_dict['loss_box_reg']      # RoI Head BBox Regression Loss (Smooth L1)
                loss_objectness    = loss_dict['loss_objectness']   # RPN Objectness Loss (BCE)
                loss_rpn_box_reg   = loss_dict['loss_rpn_box_reg']  # RPN BBox Regression Loss (Smooth L1)

                # Total Loss
                loss = (
                    loss_classifier +
                    loss_box_reg +
                    loss_objectness +
                    loss_rpn_box_reg
                )
                
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = total_loss / len(train_dataloader)

            print(f"--- Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} Time: {time.time()-start_time:.2f} seconds ---")

        self.eval()
        print("Training Complete")

    def test_model(self, test_dataloader, targets_are_masks = False):
        evaluator = MeanAveragePrecision(iou_type="bbox")

        with torch.inference_mode():
            self.eval()
            for batch_idx, (images, targets_list) in enumerate(test_dataloader):
                images = images.to(self.device)
                
                if targets_are_masks:
                    targets_list = masks_list_to_targets_list(targets_list)
                
                targets_on_device = [{k: v.to(self.device) for k, v in t.items()} for t in targets_list]
                detections = self(images)
                evaluator.update(detections, targets_on_device)
            
            metrics = evaluator.compute()
            return metrics

