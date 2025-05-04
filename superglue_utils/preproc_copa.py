from transformers import PreTrainedTokenizerBase
from datasets import load_dataset, load_from_disk


_QUESTION_DICT = {
    "cause": "What was the cause of this?",
    "effect": "What happened as a result?"
}


def tokenize_example_copa(example, tokenizer: PreTrainedTokenizerBase, max_len):
    prompt = example['premise'] + " " + _QUESTION_DICT[example['question']]
    paired_text = ([(prompt, example['choice1']), (prompt, example['choice2'])])
    tokenized_example = tokenizer(paired_text, max_length=max_len, padding="max_length", truncation="longest_first")
    example.update(tokenized_example)
    return example

def preproc_superglue_copa(tokenizer, max_len):
    train_dataset = load_dataset("super_glue", "copa")['train']
    train_dataset = train_dataset.map(lambda example: tokenize_example_copa(example, tokenizer, max_len),
                                      batched=False, num_proc=6, load_from_cache_file=True)

    test_dataset = load_dataset("super_glue", "copa")['validation']
    test_dataset = test_dataset.map(lambda example: tokenize_example_copa(example, tokenizer, max_len),
                                    batched=False, num_proc=6, load_from_cache_file=True)
    return train_dataset, test_dataset