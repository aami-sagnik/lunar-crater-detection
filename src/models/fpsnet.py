import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from src.utils import masks_to_targets, create_gt_panoptic_map, convert_gt_to_pq_format, convert_preds_to_pq_format
from torchmetrics.detection import PanopticQuality

class FPSNet(nn.Module):
    def __init__(self, n_att=50, n_stuff=0, c_att=50, retina_net_state=None, device="cpu"):
        super(FPSNet, self).__init__()
        retina_net_weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        self.retina_net = retinanet_resnet50_fpn_v2(weights=retina_net_weights)
        self.device = device
        
        FINAL_OUTPUT_CLASSES = 1 + 1 # 1 - crater, 1 - background

        original_head = self.retina_net.head.classification_head
        num_anchors = original_head.num_anchors # Usually 9

        # Determine the input channels to the final classification layer (C=256 in FPN architecture [4, 8])
        num_in_channels = original_head.cls_logits.in_channels 

        # Re-initialize the final convolutional layer to match the new output size (2 classes * num_anchors)
        self.retina_net.head.classification_head.cls_logits = torch.nn.Conv2d(
            num_in_channels,
            FINAL_OUTPUT_CLASSES * num_anchors, 
            kernel_size=3,
            stride=1,
            padding=1
        ).to(self.device)

        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.retina_net.head.classification_head.cls_logits.bias, bias_value)
        self.retina_net.head.classification_head.num_classes = FINAL_OUTPUT_CLASSES

        if retina_net_state is not None:
            self.retina_net.load_state_dict(retina_net_state)
        
        for name, param in self.retina_net.named_parameters():
            if 'backbone' in name:
                param.requires_grad = False

        self.fpn = self.retina_net.backbone
        self.upsample_f_5 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.upsample_f_4 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
        self.upsample_f_3 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding="same"),
            nn.ReLU(),
        )
        self.detection_head = self.retina_net.head
        self.anchor_generator = self.retina_net.anchor_generator
        self.postprocess_detections = self.retina_net.postprocess_detections
        self.n_att = n_att
        self.n_stuff = n_stuff
        self.n_out = n_att + n_stuff + 2 # things with ids (n_att) + stuff (n_stuff) + unmatched things (1) + unlabelled pixels(1)
        self.c_att = c_att
        # self.retina_net.score_thresh = 0.4

        self.pre_panoptic_head = nn.Sequential(
            nn.Conv2d(128 + self.n_att, 128, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.panoptic_head = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, self.n_out, 1)
        )

        self.max_int = torch.iinfo(torch.int).max

    def to_single_fmap(self, fmaps):
        f_3 = fmaps[0]
        f_4 = fmaps[1]
        f_5 = fmaps[2]

        resized_maps = [
            self.upsample_f_3(f_3),
            self.upsample_f_4(f_4),
            self.upsample_f_5(f_5),
        ]

        # Find minimum height and width
        min_h = min([f.shape[2] for f in resized_maps])
        min_w = min([f.shape[3] for f in resized_maps])

        # Crop to same size
        cropped_maps = [f[..., :min_h, :min_w] for f in resized_maps]

        aggregated_map = sum(cropped_maps)
        return aggregated_map


    def generate_detections(self, fmaps, images):
        head_outputs = self.detection_head(fmaps)

        image_sizes = [(img.shape[1], img.shape[2]) for img in images]
        image_list = ImageList(images, image_sizes)

        anchors = self.anchor_generator(image_list, fmaps)

        num_anchors_per_level = [x.size(2) * x.size(3) for x in fmaps]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs["cls_logits"].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: dict[str, list[torch.Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # compute the detections
        detections = self.postprocess_detections(split_head_outputs, split_anchors, image_list.image_sizes)

        return detections

    def soft_attention_mask(self, mu, C, box_dim, mask_w, mask_h):
        # mu: (2,) tensor, C: (2,2) tensor, w, h, n: ints

        # Create n x n grid
        ys = torch.arange(mask_h, dtype=torch.float32).to(mu.device)
        xs = torch.arange(mask_w, dtype=torch.float32).to(mu.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1)  # shape: (n, n, 2)

        # Bounding box coordinates
        y1 = int(mu[0] - box_dim[1] // 2)
        y2 = int(mu[0] + box_dim[1] // 2)
        x1 = int(mu[1] - box_dim[0] // 2)
        x2 = int(mu[1] + box_dim[0] // 2)

        # Mask for bounding box: True inside the box
        mask_box = (
            (grid_y >= y1) & (grid_y < y2) &
            (grid_x >= x1) & (grid_x < x2)
        )

        # Flatten grid for Gaussian
        coords_flat = coords.reshape(-1, 2)  # (n*n, 2)
        mvn = torch.distributions.MultivariateNormal(mu, C)
        gauss_flat = torch.exp(mvn.log_prob(coords_flat))  # (n*n,)

        # Restore mask shape
        gauss_mask = gauss_flat.reshape(mask_w, mask_h)
        gauss_mask *= mask_box  # zero outside the bounding box

        # Normalize soft mask
        sum_gauss = gauss_mask.sum()
        gauss_mask /= (sum_gauss + 1e-7)

        return gauss_mask
    
    def iou_binary_masks(self, soft_mask, binary_mask):
        intersection = torch.sum(soft_mask * binary_mask)
        union = torch.sum(soft_mask) + torch.sum(binary_mask) - intersection
        iou = intersection / (union + 1e-7)
        return iou.item()
    
    def rearrange_actual_masks_iou(self, actual_masks, predicted_masks, predicted_labels, iou_thresh=0.001):
        taken = torch.zeros(len(predicted_masks))
        chosen = torch.zeros(len(actual_masks)) - 1
        
        for i in range(len(actual_masks)):
            actual_mask = actual_masks[i]
            max_iou = 0
            selected = self.max_int
            for j in range(len(predicted_masks)):
                predicted_mask = predicted_masks[j]

                if predicted_labels[j] == -1: # these are zero-value masks for padding
                    continue

                predicted_mask = predicted_mask.unsqueeze(0).unsqueeze(0)
                predicted_mask = F.interpolate(predicted_mask, size=actual_mask.shape, mode='nearest')
                predicted_mask = predicted_mask.squeeze(0).squeeze(0)
                
                iou = self.iou_binary_masks(predicted_mask, actual_mask)
                
                if iou > iou_thresh and iou > max_iou and taken[j] == 0:
                    max_iou = iou
                    selected = j
                    taken[j] = 1

            chosen[i] = selected

        indices = chosen.argsort()
        return (actual_masks[indices], chosen[indices])
    
    def get_unlabeled_mask(self, matched_masks, aggregated_unmatched_mask):
        all_masks = torch.cat([matched_masks, aggregated_unmatched_mask], dim=0)  # Shape: (n_att+1, h, w)

        # Compute pixel-wise logical OR across the first dimension (all masks combined)
        combined_mask = torch.any(all_masks.bool(), dim=0)  # Shape: (h, w), bool

        # Invert the combined mask to get unlabeled mask
        unlabeled_mask = ~combined_mask  # bool, True where no mask is 1
        unlabeled_mask = unlabeled_mask.unsqueeze(0).to(matched_masks.dtype)
        return unlabeled_mask

    def generate_attention_masks(self, boxes, scores, labels, ms_w, ms_h, h_sf, w_sf, actual_masks=None):
        # x_sf, y_sf represent the x and y scaling factors respectively
        # x_sf * x_image_space = x_mask_space
        # y_sf * y_image_space = y_mask_space

        attention_masks = []
        attention_labels = []
        attention_scores = []

        included = 0

        for _ in range(self.n_att):
            if included < boxes.shape[0]:
                box = boxes[included]
                score = scores[included]
                label = labels[included]

                mean = torch.Tensor([box[0].item()*h_sf, box[1].item()*w_sf]).to(self.device)
                mean = torch.floor(mean).to(self.device)
                box_dim = torch.Tensor([box[2].item()*h_sf, box[3].item()*w_sf]).to(self.device)
                box_dim = torch.floor(box_dim)

                C = torch.diag(box_dim).to(self.device)

                # this prevents C from having a zero determinant if box_dim is [0, x] or [x, 0]
                C = C + torch.eye(C.shape[0]).to(self.device) * 1e-7

                attention_mask = self.soft_attention_mask(mean, C, box_dim, ms_w, ms_h)

                attention_masks.append(attention_mask)
                attention_labels.append(label.to(self.device))
                attention_scores.append(score)

                included += 1
            else:
                attention_masks.append(torch.zeros(ms_w, ms_h).to(self.device))
                attention_labels.append(torch.tensor(-1).to(self.device))
                attention_scores.append(torch.tensor(0).to(self.device))

        attention_masks = torch.stack(attention_masks).to(self.device)
        attention_labels = torch.stack(attention_labels).to(self.device)
        attention_scores = torch.stack(attention_scores).to(self.device)

        indices = torch.randperm(attention_masks.size(0))

        attention_masks = attention_masks[indices]
        attention_labels = attention_labels[indices]
        attention_scores = attention_scores[indices]

        # during training, we need to rearrange the "things" in the input masks tensor according to the attention masks generated here
        if self.training or actual_masks is not None:
            rearranged_masks, chosen = self.rearrange_actual_masks_iou(actual_masks, attention_masks, attention_labels)

            matched_mask_indices = (chosen != self.max_int).nonzero(as_tuple=True)[0]
            matched_masks = rearranged_masks[matched_mask_indices]

            # Pad matched_masks if less than n_att
            num_matched = matched_masks.size(0)
            if num_matched < self.n_att:
                padding = torch.zeros((self.n_att - num_matched, matched_masks.size(1), matched_masks.size(2)), dtype=matched_masks.dtype, device=matched_masks.device)
                matched_masks = torch.cat([matched_masks, padding], dim=0)
            else:
                # Optionally trim if more than n_att
                matched_masks = matched_masks[:self.n_att]

            # unmatched: actual masks without any corresponding predicted mask
            unmatched_mask_indices = (chosen == self.max_int).nonzero(as_tuple=True)[0]
            unmatched_masks = rearranged_masks[unmatched_mask_indices]

            aggregated_unmatched_mask = torch.any(unmatched_masks.bool(), dim=0).to(unmatched_masks.dtype).unsqueeze(0)

            unlabeled_mask = self.get_unlabeled_mask(matched_masks, aggregated_unmatched_mask)

            # Append aggregated unmatched mask to matched_masks
            new_actual_masks = torch.cat([matched_masks, aggregated_unmatched_mask, unlabeled_mask], dim=0)

            mask = torch.zeros(attention_masks.size(0), dtype=torch.bool, device=attention_masks.device)
            mask[chosen[chosen != self.max_int].long()] = True


            # Zero out attention masks not in chosen
            attention_masks = attention_masks * mask.unsqueeze(1).unsqueeze(2).to(attention_masks.dtype)

            attention_masks *= self.c_att
            return (attention_masks, attention_labels, attention_scores, new_actual_masks)
        
        attention_masks *= self.c_att
        return (attention_masks, attention_labels, attention_scores, None)

    def forward(self, images, actual_masks=None, retinanet_targets=None):
        if self.training and retinanet_targets is not None:
            # When targets are present, retina_net's forward returns a dict of losses
            retina_output = self.retina_net(images, targets=retinanet_targets)
            
            # Get FPN features
            fp = self.fpn(images)
            fmaps = list(fp.values())
            
            # Generate detections (needed for the Panoptic Head's attention mask generation)
            detections = self.generate_detections(fmaps, images)
            detection_loss = retina_output

        # If targets are not provided (during inference or just the panoptic path)
        else:
            # Standard FPSNet forward: run FPN, then generate detections manually
            fp = self.fpn(images)
            fmaps = list(fp.values())
            detections = self.generate_detections(fmaps, images)
            detection_loss = {}

        aggregated_maps = self.to_single_fmap(fmaps) # scaling and adding the feature maps to get a single feature map

        panoptic_features = []
        panoptic_labels = []
        panoptic_scores = []

        h = images.shape[2]
        w = images.shape[3]

        h_mask_sf = aggregated_maps.shape[2] / h
        w_mask_sf = aggregated_maps.shape[3] / w

        modified_actual_masks = None
        if self.training or actual_masks is not None:
            modified_actual_masks = []

        for i in range(images.shape[0]):
            detection = detections[i]
            aggregated_map = aggregated_maps[i]
            
            actual_mask = None
            if self.training or actual_masks is not None:
                actual_mask = actual_masks[i]

            # "detection" represents boxes for one image in a batch
            # "aggregated_map" represents feature map for one image in a batch
            boxes = detection["boxes"]
            scores = detection["scores"]
            labels = detection["labels"]
            attention_masks, attention_labels, attention_scores, new_actual_mask = self.generate_attention_masks(boxes, scores, labels, aggregated_map.shape[1], aggregated_map.shape[2], h_mask_sf, w_mask_sf, actual_masks=actual_mask) # generating attention masks from bounding boxes

            if self.training or actual_masks is not None:
                modified_actual_masks.append(new_actual_mask)

            panoptic_feature_single = torch.cat([attention_masks, aggregated_map], dim=0) # generating panoptic features for one image

            # for batch processing multiple images at a time
            panoptic_features.append(panoptic_feature_single)
            panoptic_labels.append(attention_labels)
            panoptic_scores.append(attention_scores)

        panoptic_features = torch.stack(panoptic_features)
        if self.training or actual_masks is not None:
            modified_actual_masks = torch.stack(modified_actual_masks)

        panoptic_features = self.pre_panoptic_head(panoptic_features)
        output = self.panoptic_head(panoptic_features)

        upscaled_output = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)

        result = {
            "values": upscaled_output,
            "labels": panoptic_labels,
            "scores": panoptic_scores,
            "boxes": [detection["boxes"] for detection in detections],
            "mod_masks": modified_actual_masks,
            "detections": detections
        }
        result.update(detection_loss)

        return result
    
    def train_model(self, train_dataloader, epochs):
        params_to_optimize = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params_to_optimize, lr=0.01)
        self.train()

        print(f"Starting training on device: {self.device}")

        for epoch in range(epochs):
            total_loss = 0
            
            # Dataloader yields (images, actual_masks_tensor)
            for batch_idx, (images, actual_masks_tensor) in enumerate(train_dataloader):
                # 1. Prepare Data
                images = images.to(self.device)
                actual_masks_tensor = actual_masks_tensor.to(self.device) 
                
                # --- DYNAMIC TARGET GENERATION ---
                # Generate the List[Dict] format detection targets from the ground truth masks
                targets_list = masks_to_targets(actual_masks_tensor, self.device)
                # --- END DYNAMIC TARGET GENERATION ---

                # 2. Forward Pass (Calculates both detection outputs and panoptic features)
                output = self(images, actual_masks=actual_masks_tensor, retinanet_targets=targets_list)
                
                # 3. Calculate Total Loss
                # A. Panoptic Cross-Entropy Loss (L_PCE)
                mod_masks_batch = output['mod_masks']
                gt_pan_map = create_gt_panoptic_map(mod_masks_batch, self.n_att, 0)
                
                loss_panoptic = F.cross_entropy(output['values'], gt_pan_map.to(torch.long))
                
                # B. Detection Loss (L_DET) - Calculated internally by RetinaNet
                loss_detection_cls = output['classification']
                loss_detection_box = output['bbox_regression']
                
                loss_detection = loss_detection_cls + loss_detection_box

                # C. Total Loss 
                loss = loss_panoptic + 0.5 * loss_detection
                # 4. Backward Pass and Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}")
                    print(f"  Total Loss: {loss.item():.4f} | PCE: {loss_panoptic.item():.4f} | CLS: {loss_detection_cls.item():.4f} | BOX: {loss_detection_box.item():.4f}")

            avg_loss = total_loss / len(train_dataloader)
            print(f"--- Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} ---")

        self.eval()

        print(f"Training complete")

    def test_model(self, test_dataloader):
        # Specify class IDs for 'things' (Crater is ID 1). Stuff is empty.
        crater_class_id = 1
        things_classes = {crater_class_id}
        stuffs_classes = {0} 

        # Initialize metric (set ignore index to 0 for background/unlabeled)
        pq_metric = PanopticQuality(
            things=things_classes, 
            stuffs=stuffs_classes,
        ).to(self.device)

        with torch.inference_mode():
            total_pq = 0 
            for image_batch, mask_batch in test_dataloader:
                image_batch = image_batch.to(self.device)
                mask_batch = mask_batch.to(self.device)
                output = self(image_batch, mask_batch)
                mod_masks_batch = output["mod_masks"]
                gt_pan_map = create_gt_panoptic_map(mod_masks_batch, self.n_att, 0)
                preds_pq = convert_preds_to_pq_format(output['values'], self.n_att, crater_class_id)
                target_pq = convert_gt_to_pq_format(gt_pan_map, self.n_att, crater_class_id)
                pq_results = pq_metric(preds_pq, target_pq)
                total_pq += pq_results
        
        avg_pq = total_pq / len(test_dataloader)

        return [
            {
                "name": "Panoptic Quality",
                "value": avg_pq
            }
        ]