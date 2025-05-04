import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import f1_score

from transformers import AutoConfig, EvalPrediction, AutoModelForSequenceClassification

from trainer import Trainer

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    set_seed,
)

import transformers
transformers.logging.set_verbosity_debug()

from torch.utils.tensorboard import SummaryWriter
from trainer import EarlyStoppingCallback
from superglue_utils import preproc_superglue_5
from glue_utils import GlueClassificationCollator

from utils.tokenizers import (
    Tokenizer_f,
    Tokenizer_m,
    Tokenizer_l,
    Tokenizer_ff,
    Tokenizer_ll,
    Tokenizer_fl,
    Tokenizer_fff,
    Tokenizer_lll,
    Tokenizer_fml,
    Tokenizer_full_token,
    Tokenizer_full_token_vowels,
    Tokenizer_full_token_consonants
)

import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    patience: int = field(
        default=5, metadata={"help": "Patience value for early stopping."}
    )
    checkpoint_model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to checkpoint."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_pretrained_weights: Optional[bool] = field(
        default=False, metadata={"help": "If `True`, pre-trained weights will not be fine-tuned."}
    )
    tokenizer_name: str = field(
        default=None,
        metadata={"help": "The name of the tokenizer. Choose from: f, m, l, ff, fl, ll, fff, " 
                                            + "fml, lll, full_token, vowels, consonants"}
    )


class SuperGlueTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            self.optimizer = transformers.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(num_training_steps * 0.06), num_training_steps=num_training_steps,
                num_cycles=5
            )


def each_tr(seed):
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.output_dir = training_args.output_dir + "--" + str(seed)
    training_args.logging_dir = training_args.logging_dir + "--" + str(seed)

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(seed)

    fine_tuning_task_dict = {
        'boolq': 'boolq',
        'cb': 'cb',
        'rte': 'rte',
        'wic': 'rte',
        'multirc': 'mrpc'
    }

    convert_task_dict = {
        'boolq': 'rte',
        'cb': 'cb',
        'rte': 'rte',
        'wic': 'rte',
        'multirc': 'mrpc'
    }

    report_metrics_dict = {
        'boolq': 'eval_acc',
        'cb': 'eval_f1',
        'rte': 'eval_acc',
        'wic': 'eval_acc',
        'multirc': 'eval_f1'
    }

    steps_dict = {
        'boolq': 177,
        'cb': 5,
        'rte': 47,
        'wic': 102,
        'multirc': 512
    }

    superglue_tasks_num_labels = {
        'boolq': 2,
        'cb': 3,
        'rte': 2,
        'wic': 2,
        'multirc': 2
    }


    try:
        num_labels = superglue_tasks_num_labels[data_args.task_name]
        output_mode = "classification"
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    try:
        report_metrics = report_metrics_dict[data_args.task_name]
    except KeyError:
        raise ValueError("Metrics not found: %s" % (data_args.task_name))

    set_seed(seed)

 
    config = AutoConfig.from_pretrained(model_args.model_path,
                                        num_labels=num_labels,
                                        finetuning_task=fine_tuning_task_dict[data_args.task_name])
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_path, config=config)
   


    tokenizers = {
        'f':Tokenizer_f,
        'm':Tokenizer_m,
        'l':Tokenizer_l,
        'ff':Tokenizer_ff,
        'fl':Tokenizer_fl,
        'll':Tokenizer_ll,
        'fff':Tokenizer_fff,
        'fml':Tokenizer_fml,
        'lll':Tokenizer_lll,
        'full_token':Tokenizer_full_token,
        'vowels':Tokenizer_full_token_vowels,
        'consonants':Tokenizer_full_token_consonants
    }

    if model_args.tokenizer_name in tokenizers:
        tokenizer = tokenizers[model_args.tokenizer_name]()
    else:
        raise ValueError(
            "The tokenizer name is not recognised."
            "Please choose from f, m, l, ff, fl, ll, fff, fml, lll, full_token, vowels, consonants"
            )


    if data_args.task_name == 'multirc':
        train_dataset, test_dataset = preproc_superglue_5(task_name=data_args.task_name, tokenizer=tokenizer,max_len=512)
    else:
        train_dataset, test_dataset = preproc_superglue_5(task_name=data_args.task_name, tokenizer=tokenizer,max_len=256)
    train_dataset.rename_column_('label', 'labels')
    test_dataset.rename_column_('label', 'labels')
    eval_dataset = test_dataset

    data_collator = GlueClassificationCollator()


    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            if output_mode == "classification":
                preds = np.argmax(preds, axis=1)
                if task_name == 'cb':
                    acc = (preds == p.label_ids).mean()
                    f11 = f1_score(y_true=p.label_ids == 0, y_pred=preds == 0)
                    f12 = f1_score(y_true=p.label_ids == 1, y_pred=preds == 1)
                    f13 = f1_score(y_true=p.label_ids == 2, y_pred=preds == 2)
                    f1 = (f11 + f12 + f13) / 3.0
                    return {"acc": acc, "f1": f1}
                else:
                    preds = np.squeeze(preds)
                return glue_compute_metrics(convert_task_dict[task_name], preds, p.label_ids)
        return compute_metrics_fn

    # Initialize our Trainer
    tb_writer = SummaryWriter(training_args.logging_dir)
    trainer = SuperGlueTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tb_writer=tb_writer,
        compute_metrics=build_compute_metrics_fn(data_args.task_name),
        data_collator=data_collator
    )
    trainer.add_callback(
        EarlyStoppingCallback(
            patience=model_args.patience,
            metric_name=report_metrics_dict[data_args.task_name],
            objective_type="maximize"
        )
    )

    if training_args.do_train:
        trainer.train(model_path=None)
        trainer.save_model()

    if model_args.freeze_pretrained_weights:
        logger.info("Fine-tuned only a linear layer!")
    else:
        logger.info("Fine-tuned all layers!")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total # of trainable params: {total_params}")

    eval_results = {}
    logger.info("*** Evaluate ***")
    trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name)
    eval_result = trainer.evaluate(eval_dataset=eval_dataset)

    output_eval_file = os.path.join(
        training_args.output_dir, f"eval_results_{data_args.task_name}--{seed}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(data_args.task_name))
            for key, value in eval_result.items():
                logger.info(" %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
    eval_results.update(eval_result)

    predict_results = {}
    logger.info("*** Predict ***")
    trainer.compute_metrics = build_compute_metrics_fn(data_args.task_name)
    predict_result = trainer.evaluate(eval_dataset=test_dataset)
    output_eval_file = os.path.join(
        training_args.output_dir, f"test_results_{data_args.task_name}--{seed}.txt"
    )
    if trainer.is_world_master():
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results {} *****".format(data_args.task_name))
            for key, value in predict_result.items():
                logger.info(" %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
    predict_results.update(predict_result)

    output_all_file = os.path.join(
        training_args.output_dir.split('--')[0], f"all_results_{data_args.task_name}--{seed}.txt"
    )

    return training_args.output_dir.split('--')[0], output_all_file, \
        eval_results[report_metrics_dict[data_args.task_name]], \
        predict_results[report_metrics_dict[data_args.task_name]]


if __name__ == "__main__":
    results_table = []
    # seeds
    for i in [1 , 12, 19, 29, 42]:
        output_dir, output_all_file, eval_results, predict_results = each_tr(i)
        results_table.append((eval_results, predict_results))
    res = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_all_file, "w") as writer:
        for j in results_table:
            res.append(j[1])
            writer.write("%s , %s\n" % j)
        writer.write("%.1f\n" % round(np.mean(res)*100, 2))
        writer.write("%.2f\n" % round(np.std(res)*100, 2))






