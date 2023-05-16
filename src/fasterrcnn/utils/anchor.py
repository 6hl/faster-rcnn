import torch
import torchvision


class Anchor(object):
    """ Class makes anchors for input image

    Args:
        n_anchors (int): number of anchors per location
        anchor_ratios (list): ratio of anchor sizes for each location
        scales (list): scales of anchors for each location
        anchor_threshold (list): [upper_threshold, lower_threshold]
                                1 > upper_threshold > lower_threshold > 0    
    """
    def __init__(
        self,
        model,
        n_anchors=9, 
        anchor_ratios=[0.5, 1, 2], 
        scales=[8, 16, 32], 
        anchor_threshold=[0.5, 0.1], 
        batch_size=[256, 64],
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_anchors = torch.tensor(n_anchors)
        self.anchor_ratios = torch.tensor(anchor_ratios)
        self.anchor_threshold = anchor_threshold
        self.scales = torch.tensor(scales)
        self.border = 0
        self.train_nms_filter = [12000, 2000]
        self.test_nms_filter = [6000, 300]
        self.batch_size = batch_size
        self.roi_anchor_threshold = [0.5, 0.0]
    
    def _ratio_anchors(self, base_anchor, ratios):
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
    
    def _anchor_set(self, yolo_anchor):
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
    def _voc_to_yolo(self, bbx):
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

    def _scale_ratio_anchors(self, anchor, scales):
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

    def _parameterize(self, source_bxs, dst):
        """
        Args:
            source_bxs (torch.tensor[N,4]): source boxes x,y,w,h
            dst (torch.tensor[N,4]): ground_truth boxes x,y,w,h
        Returns:
            torch.tensor[N, 4]
        """
        source_bxs = torchvision.ops.box_convert(source_bxs, in_fmt="xyxy", out_fmt="cxcywh")
        dst = torchvision.ops.box_convert(dst, in_fmt="xyxy", out_fmt="cxcywh")
        return torch.stack(
            (
                (source_bxs[:,0] - dst[:,0]) / dst[:, 2],
                (source_bxs[:,1] - dst[:,1])/ dst[:, 3],
                torch.log(source_bxs[:,2]/dst[:,2]),
                torch.log(source_bxs[:,3]/dst[:,3])
            ), dim=1
        ).to(torch.float64)
    
    def _unparameterize(self, source_bxs, deltas):
        """
        Args: 
            source_bxs torch.tensor[N,4]: in (x1,y1,x2,y2) order
            deltas torch.tensor[N,4]: (delta_x, delta_y, delta_w, delta_h)
        Returns:
            torch.tensor[N,4]
        """
        source_bxs = torchvision.ops.box_convert(source_bxs, in_fmt="xyxy", out_fmt="cxcywh")
        return torchvision.ops.box_convert(
            torch.stack(
                (
                    deltas[:, 0] * source_bxs[:, 2] + source_bxs[:, 0],
                    deltas[:, 1] * source_bxs[:, 3] + source_bxs[:, 1],
                    torch.exp(deltas[:, 2]) * source_bxs[:, 2],
                    torch.exp(deltas[:, 3]) * source_bxs[:, 3]
                ), dim=1
            ),
            in_fmt="cxcywh",
            out_fmt="xyxy"
        ).to(torch.float64)
 
    # TODO: Make anchor script dynamically adjust scales if needed
    def generate_anchor_mesh(self, images, feature_maps):
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
        anchors = torchvision.ops.clip_boxes_to_image(anchor_mesh, (h_img, w_img))

        return anchors
    
    def generate_rpn_targets(self, anchors, true_bx):
        """ Generates RPN Targets

        Args:
            anchors (torch.tensor[N, 4]): generated anchors
            true_bx (torch.tensor[N, 4]): ground truth bbxs
        
        Returns:
            true_rpn_delta (torch.tensor[N, 4])
            true_rpn_targets (torch.tensor[N, 4])
            rpn_anchor_idx (torch.tensor[N, 4])
        """
        true_bx = true_bx.reshape(-1,4)
        anchor_iou = torchvision.ops.box_iou(true_bx, anchors)
        max_values, max_idx = anchor_iou.max(dim=0)
        true_anchor_bxs = torch.stack(
            [true_bx[m.item()] for m in max_idx],
            dim=0
        )

        # Find fg/bg anchor idx
        fg_idx = (max_values >= self.anchor_threshold[0]).nonzero().ravel()
        bg_idx = (max_values <= self.anchor_threshold[1]).nonzero().ravel()
        bg_bool = True if len(bg_idx) <= int(self.batch_size[0]/2) else False

        # Create batch from fg/bg idx
        if len(fg_idx) > 0:
            fg_idx = fg_idx[
                torch.ones(len(fg_idx)).multinomial(
                    min(int(self.batch_size[0]/2), len(fg_idx)),
                    replacement=False)
                ]
        bg_idx = bg_idx[
            torch.ones(len(bg_idx)).multinomial(
                self.batch_size[0]-len(fg_idx),
                replacement=bg_bool)
            ]
        
        rpn_anchor_idx = torch.cat((fg_idx, bg_idx), dim=0)
        true_rpn_targets = torch.tensor([1]*len(fg_idx) + [0]*len(bg_idx))
        true_rpn_delta = self._parameterize(anchors[rpn_anchor_idx, :], true_anchor_bxs[rpn_anchor_idx, :])
        return true_rpn_delta, true_rpn_targets, rpn_anchor_idx

    def generate_roi(
            self, 
            anchors, 
            rpn_delta, 
            rpn_targets, 
            true_bx, 
            true_targets, 
            img_shape
        ):
        """ Generates ROI
        Args:
            anchors (torch.tensor[N, 4]): generated anchors
            rpn_delta (torch.tensor[1, N, 4]): RPN anchor deltas
            rpn_targets (torch.tensor[1, N, C]): RPN targets
            true_bx (torch.tensor[N, 4]): ground truth bbxs
            true_targets (torch.tensor[N, C]): ground truth targets
            img_shape (torch.tensor[1,3,H,W]): input image shape
        Returns:
            batch_roi (torch.tensor) 
            true_roi_delta (torch.tensor)
            true_roi_targets (torch.tensor)
        """
        # TODO: Ensure GPU support
        if self.model.training:
            nms_filter = self.train_nms_filter
        else:
            nms_filter = self.test_nms_filter

        rpn_un_param = self._unparameterize(anchors, rpn_delta[0].clone().detach())
        rpn_anchors = torchvision.ops.clip_boxes_to_image(rpn_un_param, (img_shape[2], img_shape[3]))
        rpn_anchor_idx = torchvision.ops.remove_small_boxes(rpn_anchors, 16.0)
        rpn_anchors = rpn_anchors[rpn_anchor_idx, :]
        rpn_targets = rpn_targets[0][rpn_anchor_idx].clone().detach().view(-1, 2)[:,1]

        top_scores_idx = rpn_targets.argsort()[:nms_filter[0]]
        rpn_anchors = rpn_anchors[top_scores_idx, :].to(torch.float64)
        rpn_targets = rpn_targets[top_scores_idx].to(torch.float64)

        nms_idx = torchvision.ops.nms(rpn_anchors, rpn_targets, iou_threshold=0.7)
        nms_idx = nms_idx[:nms_filter[1]]
        rpn_anchors = rpn_anchors[nms_idx, :]
        rpn_targets = rpn_targets[nms_idx]

        if not self.model.training:
            return rpn_anchors, None, None

        anchor_iou = torchvision.ops.box_iou(true_bx.reshape(-1,4), rpn_anchors)
        max_values, max_idx = anchor_iou.max(dim=0)
        true_roi_targets = torch.zeros(self.batch_size)
        
        # Find fg/bg anchor idx
        fg_idx = (max_values >= self.anchor_threshold[0]).nonzero().ravel()
        bg_idx = ((max_values < self.roi_anchor_threshold[0]) &
                (max_values >= self.roi_anchor_threshold[1])).nonzero().ravel()

        # Create batch from fg/bg idx, consider repeated values
        if len(fg_idx) > 0:
            fg_idx = fg_idx[
                torch.ones(len(fg_idx)).multinomial(
                    min(int(self.batch_size[0]/2), len(fg_idx)), 
                    replacement=False
                )
            ]
        bg_idx = bg_idx[
            torch.ones(len(bg_idx)).multinomial(
                self.batch_size[0]-len(fg_idx), 
                replacement=True if len(bg_idx) <= self.batch_size[0]-len(fg_idx) else False
            )
        ]
        
        batch_rpn_idx = torch.cat((fg_idx, bg_idx), dim=0)
        batch_roi = rpn_anchors[batch_rpn_idx]
        true_roi_targets = torch.cat((true_targets[max_idx[fg_idx]], torch.zeros(len(bg_idx)))).long()
        true_roi_delta = self._parameterize(batch_roi, anchors[batch_rpn_idx])
        return batch_roi, true_roi_delta, true_roi_targets
