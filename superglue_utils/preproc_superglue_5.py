from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
import pandas as pd

def tokenize_example(task_name, tokenizer: PreTrainedTokenizerBase, example, max_len):
    if task_name in ["boolq"]:
        batched = list(map(lambda x, y: (x, y), example["question"], example["passage"]))
        tokenized_example = tokenizer(batched, max_length=max_len, padding="max_length", truncation="longest_first")
    elif task_name in ["cb", "rte", "axg"]:
        batched = list(map(lambda x, y: (x, y), example["premise"], example["hypothesis"]))
        tokenized_example = tokenizer(batched, max_length=max_len, padding="max_length", truncation="longest_first")
    elif task_name in ["wic"]:
        batched = list(map(lambda x, y, z: (x+ " </s> </s> " +y, z),
                       example["word"], example["sentence1"], example["sentence2"]))
        tokenized_example = tokenizer(batched, max_length=max_len, padding="max_length", truncation="longest_first")
    elif task_name in ["multirc"]:
        batched = list(map(lambda x, y, z: truncate_ori(x, y, z, tokenizer, max_length=max_len),
                                                        example["paragraph"], example["question"], example["answer"]))
        tokenized_example = tokenizer(batched, max_length=max_len, padding="max_length", truncation="longest_first")
    elif task_name in ["axb"]:
        batched = list(map(lambda x, y: (x, y), example["sentence1"], example["sentence2"]))
        tokenized_example = tokenizer(batched, max_length=max_len, padding="max_length", truncation="longest_first")
    else:
        raise NotImplementedError
    return tokenized_example


def truncate_ori(paragraph, question, answer, tokenizer, max_length):
    question_len = len(tokenizer.tokenize(question))
    answer_len = len(tokenizer.tokenize(answer))
    paragraph = " ".join(tokenizer.tokenize(paragraph)[:max_length-2-question_len-answer_len])
    return paragraph + " " + question + " </s> </s> " + answer


def preproc_superglue_5(task_name, tokenizer, max_len):
    train_dataset = load_dataset("super_glue", task_name)["train"]
    train_dataset = train_dataset.map(lambda example: tokenize_example(task_name, tokenizer, example, max_len),
                                      batched=True, batch_size=5000, num_proc=6, load_from_cache_file=True)
    test_dataset = load_dataset("super_glue", task_name)["validation"]
    test_dataset = test_dataset.map(lambda example: tokenize_example(task_name, tokenizer, example, max_len),
                                    batched=True, batch_size=5000, num_proc=6, load_from_cache_file=True)
    return train_dataset, test_dataset





