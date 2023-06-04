import torch
import torchvision


def parameterize(source_bxs, dst) -> torch.Tensor:
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

def unparameterize(source_bxs, deltas) -> torch.Tensor:
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