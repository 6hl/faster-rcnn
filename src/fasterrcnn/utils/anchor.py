import torch
import torch.nn as nn
import torchvision


class Anchor(nn.Module):
    """ Class makes anchors for input image

    Args:
        n_anchors (int): number of anchors per location
        anchor_ratios (list): ratio of anchor sizes for each location
        scales (list): scales of anchors for each location
        anchor_threshold (list): [upper_threshold, lower_threshold]
                                1 > upper_threshold > lower_threshold > 0
        batch_size (list): Batch sizes for detection regions
    """
    def __init__(
        self,
        n_anchors: int = 9, 
        anchor_ratios: list = [0.5, 1, 2], 
        scales: list = [8, 16, 32], 
        anchor_threshold: list = [0.5, 0.1], 
        batch_size: list = [256, 64],
    ):
        super().__init__()
        self.n_anchors = torch.tensor(n_anchors)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.anchor_threshold = anchor_threshold
        self.scales = torch.tensor(scales)
        self.train_nms_filter = [12000, 2000]
        self.test_nms_filter = [6000, 300]
        self.batch_size = batch_size
        self.roi_anchor_threshold = [0.5, 0.0]
    
    def _ratio_anchors(self, base_anchor: torch.Tensor, ratios: torch.Tensor) -> torch.Tensor:
        """Helper function to generate ratio anchors
        Args:
            base_anchor (torch.Tensor): initial anchor location
            ratios (torch.Tensor): ratios for anchors
        Returns:
            torch.Tensor: bounding boxes (len(ratios), 4)
        """
        
        yolo_anchor = self._voc_to_yolo(base_anchor.reshape(-1,4))[0]
        wr = torch.round(torch.sqrt(yolo_anchor[2]*yolo_anchor[3]/ratios))
        hr = torch.round(wr*ratios)
        return self._anchor_set(
            [
                yolo_anchor[0],
                yolo_anchor[1],
                wr,
                hr
            ]
        )
    
    def _anchor_set(self, yolo_anchor: torch.Tensor) -> torch.Tensor:
        """Helper function to generate anchors
        Args:
            yolo_anchor (torch.Tensor): (x_center, y_center, width, height)
        Returns:
            torch.Tensor: (n,4) set of (x1,y1,x2,y2) cords
        """

        return torch.stack(
            (
                yolo_anchor[0] - 0.5 * (yolo_anchor[2]-1),
                yolo_anchor[1] - 0.5 * (yolo_anchor[3]-1),
                yolo_anchor[0] + 0.5 * (yolo_anchor[2]-1),
                yolo_anchor[1] + 0.5 * (yolo_anchor[3]-1),
            ), 
            dim=1
        ) 

    # TODO: Remove for torchvision.ops.box_convert
    def _voc_to_yolo(self, bbx: torch.Tensor) -> torch.Tensor:
        """Helper function that returns yolo labeling for bounding box
        Args:
            bbx (torch.Tensor): [N, 4] (x1, y1, x2, y2)
        Returns:
            torch.Tensor: (x_center, y_center, width, height)
        """

        return torch.stack(
            (
                bbx[:, 0] + 0.5*(bbx[:, 3]-1), 
                bbx[:, 1] + 0.5*(bbx[:, 2]-1), 
                bbx[:, 3] - bbx[:, 1] + 1,
                bbx[:, 2] - bbx[:, 0] + 1
            ), dim=1
        )

    def _scale_ratio_anchors(self, anchor: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """Helper function to scale the ratio anchors
        Args:
            anchor (torch.Tensor): (x_center, y_center, width, height)
            scales (torch.Tensor): scales for anchors
        """

        yolo_anchor = self._voc_to_yolo(anchor.reshape(-1,4))[0]
        return self._anchor_set(
            [
                yolo_anchor[0], 
                yolo_anchor[1], 
                yolo_anchor[2] * scales, 
                yolo_anchor[3] * scales
            ]
        )    
 
    # TODO: Make anchor script dynamically adjust scales if needed
    def forward(self, images: torch.Tensor, feature_maps: torch.Tensor) -> torch.Tensor:
        """Function generates anchor maps for given image and feature maps
        Args:   
            images (torch.Tensor): input image
            feature_maps (torch.Tensor): backbone feature maps
        Returns:
            torch.Tensor[feature_maps*anchors, 4]
        """

        h_img, w_img = images.shape[2], images.shape[3]
        h_fmap, w_fmap = feature_maps.shape[2], feature_maps.shape[3]
        n_fmap = h_fmap*w_fmap

        # TODO: Adjust for batchsize > 1
        h_stride, w_stride = h_img/h_fmap, w_img/h_fmap
        base_anchor_local = torch.tensor([0, 0, w_stride-1, h_stride-1])
        ratio_anchors_local = self._ratio_anchors(base_anchor_local, self.anchor_ratios)
        local_anchors = torch.stack([
            self._scale_ratio_anchors(ratio_anchors_local[i,:], self.scales) for i in range(ratio_anchors_local.shape[0])
        ], dim=0).reshape(1, -1, 4)
        mesh_x, mesh_y = torch.meshgrid(
            (
                torch.arange(0, w_fmap) * w_stride,
                torch.arange(0, h_fmap) * h_stride
            ),
            indexing="xy"
        )
        anchor_shifts = torch.stack(
            (
                mesh_x.flatten(),
                mesh_y.flatten(),
                mesh_x.flatten(),
                mesh_y.flatten()
            ), 
            dim=0
        ).transpose(0,1).reshape(1, n_fmap, 4).view(-1, 1, 4)

        anchor_mesh = (local_anchors + anchor_shifts).reshape(-1,4)
        self.anchors = torchvision.ops.clip_boxes_to_image(anchor_mesh, (h_img, w_img)).to(feature_maps.device)
        return self.anchors
