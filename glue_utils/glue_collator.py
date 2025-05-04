import torch
from transformers import BatchEncoding

# for debugging
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


class GlueClassificationCollator:
    def __init__(self):
        pass

    def __call__(self, examples):
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])
        return batch