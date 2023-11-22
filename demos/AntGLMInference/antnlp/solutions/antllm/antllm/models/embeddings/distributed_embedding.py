import torch
import torch.distributed as dist


class EmbeddingGather(torch.autograd.Function):
    """
    Gather the embedding tensor along the first dimention for training.
    Maintain the grad for backward.
    """
    @staticmethod
    def forward(ctx, x: torch.tensor):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def embedding_gather(x: torch.tensor):
    """
    Gather the embedding tensor along the first dimention.
    """
    if not dist.is_initialized():
        return x
    x_gather = EmbeddingGather.apply(x)
    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


@torch.no_grad()
def embedding_gather_nograd(x: torch.tensor):
    if not dist.is_initialized():
        return x
    x_gather = [torch.ones_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(x_gather, x, async_op=False)

    x_gather = torch.cat(x_gather, dim=0)
    return x_gather


@torch.no_grad()
def varsize_gather_nograd(x: torch.Tensor):
    """
    Gather tensors of different sizes along the first dimension
    """
    if not dist.is_initialized():
        return x

    # determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    max_size = max([size.cpu().max() for size in allsizes])

    padded = torch.empty(max_size, *x.shape[1:], dtype=x.dtype, device=x.device)
    padded[: x.shape[0]] = x
    output = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(output, padded)

    output = [tensor[: allsizes[k]] for k, tensor in enumerate(output)]
    output = torch.cat(output, dim=0)

    return output


@torch.no_grad()
def get_varsize(x: torch.Tensor):
    """gather tensors of different sizes along the first dimension"""
    if not dist.is_initialized():
        return [x.shape[0]]

    # determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    allsizes = torch.cat(allsizes)
    return allsizes


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main():
    return get_rank() == 0


def get_world_size():
    if not dist.is_initialized():
        return 1
    else:
        return dist.get_world_size()
