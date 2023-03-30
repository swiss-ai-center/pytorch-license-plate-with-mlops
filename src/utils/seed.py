import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set seed for reproducibility.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # This make it reproducible on GPU but much slower
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # TODO: This is not supported on CUDA >= 10.2
    # torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
