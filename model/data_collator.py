"""
reference: https://github.com/gucci-j/light-transformer-emnlp2021/blob/master/src/model/data_collator.py
"""

from transformers import BatchEncoding, PreTrainedTokenizerBase
import torch
import re
import random

# for debugging
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

class DataCollatorForMaskedLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_prob: float=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob


    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])

        # create inputs and targets for MLM
        inputs, labels = self.mask_tokens(batch["input_ids"])
        return {"input_ids": inputs, 
                "attention_mask": batch["attention_mask"], 
                "labels": labels}


    def mask_tokens(self, inputs: torch.Tensor):
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone() # this serves as a target

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_prob defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # By passing -100, nn.CrossEntropyLoss will ignore such labels when computing loss.
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.get_vocab()[self.tokenizer.mask_token]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.get_vocab()), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class DataCollatorForMaskedLanguageModelingWithWholeTokenPrediction:
    """
    Data collator used for language modeling.
    - uses the whole tokens as the target (predicts the full word of the masked token)
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling

    Reference:
        https://github.com/huggingface/transformers/blob/f744b81572e533af5a8469c2fba661c5972f2b66/src/transformers/data/data_collator.py#L118
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mlm_prob: float=0.15):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob


    def __call__(self, examples):
        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(examples[0], (dict, BatchEncoding)):
            examples = [vars(f) for f in examples]
        first = examples[0]

        # Handling of all possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        batch = {}
        for k, v in first.items():
            if v is not None and k != 'text':
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in examples])
                else:
                    batch[k] = torch.tensor([f[k] for f in examples])
        batch['text'] = [f['text'] for f in examples]            
        
        # create inputs and targets for MLM
        inputs, labels = self.mask_tokens(batch["input_ids"], batch["text"])
        return {"input_ids": inputs, 
                "attention_mask": batch["attention_mask"], 
                "labels": labels}


    def mask_tokens(self, inputs: torch.Tensor, text: str):
        """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        tokenized_whole = self.tokenizer(text, max_length=512, padding="max_length", truncation="longest_first")
        labels = torch.tensor(tokenized_whole['input_ids']) 

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_prob defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer.pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # By passing -100, nn.CrossEntropyLoss will ignore such labels when computing loss.
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.get_vocab()[self.tokenizer.mask_token]

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer.get_vocab()), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

