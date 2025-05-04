# Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance?


This repo contains the models and the implementation code for the EMNLP 2023 Main paper [Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance?](https://aclanthology.org/2023.emnlp-main.563/).


## Pre-training
### 1. Required packages installation
The required packages can be installed using the following command:
```
pip install -r requirements.txt
```
The `en_core_web_sm` model from spaCy is required for preprocessing and can be installed by running the follwoing `python -m spacy download en_core_web_sm`.

### 2. Datasets pre-processing
For each model type, the BookCorpus and the English Wikipedia pre-training datasets need to be pre-processed using the model's tokenizer. The following command pre-process the pre-training datasets: 
```
cd ./utils
python preprocess_roberta.py --tokenizer_name=<tokenizer_name> --path=/path/to/save/data/
```
The <tokenizer_name> can be one of the following:
```
'f': Tokenizer for words first character
'm': Tokenizer for words middle character
'l': Tokenizer for words last character
'ff': Tokenizer for words first two characters
'fl': Tokenizer for words first and last characters
'll': Tokenizer for words last two characters
'fff': Tokenizer for words first three characters
'fml': Tokenizer for words first, middle and last characters
'lll': Tokenizer for words last three characters
'full_token': Tokenizer for words full characters
'vowels': Tokenizer for words vowel characters
'consonants': Tokenizer for words consonant characters  
```
### 3. Models pre-training
The following command can be used to pre-train the models. You need to specify the path to the pre-processed pre-training dataset and the name of the tokenizer for each model. 
```
python pretrainer.py \
--data_dir=/path/to/dataset/ \
--tokenizer_name=<tokenizer_name> \
--do_train \
--learning_rate=1e-4 \
--hidden_size=768 \
--intermediate_size=3072 \
--num_attention_heads=12 \
--num_hidden_layers=12 \
--weight_decay=0.01 \
--adam_epsilon=1e-8 \
--max_grad_norm=1.0 \
--num_train_epochs=20 \
--warmup_steps=10000 \
--save_steps=50000 \
--save_interval=100000 \
--seed=42 \
--per_device_train_batch_size=16 \
--logging_steps=100 \
--output_dir=/path/to/save/weights/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=True \
--prediction_loss_only \
--fp16
```


#### Distributed pre-training  
The pre-training process can be distributed using the following command:  
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 \
pretrainer.py \
--data_dir=/path/to/dataset/ \
--tokenizer_name=<tokenizer_name> \
--do_train \
--learning_rate=1e-4 \
--hidden_size=768 \
--intermediate_size=3072 \
--num_attention_heads=12 \
--num_hidden_layers=12 \
--weight_decay=0.01 \
--adam_epsilon=1e-8 \
--max_grad_norm=1.0 \
--num_train_epochs=20 \
--warmup_steps=10000 \
--save_steps=50000 \
--save_interval=100000 \
--seed=42 \
--per_device_train_batch_size=16 \
--logging_steps=100 \
--output_dir=/path/to/save/weights/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=True \
--prediction_loss_only \
--fp16
```

### 4. Fine-tunining on GLUE
First, the GLUE data needs to be downloaded:
```
git clone https://github.com/huggingface/transformers
python transformers/utils/download_glue_data.py
```

Then, the model can be fine-tuned using the following command:
```
python run_glue.py \
--model_name_or_path=/path/to/pre-trained/weights/ \
--tokenizer_name=<tokenizer_name> \
--task_name=<task> \
--do_train \
--do_eval \
--do_predict \
--data_dir=/path/to/task/dataset/ \
--max_seq_length=128 \
--learning_rate=2e-5 \
--num_train_epochs=3 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=128 \
--logging_steps=500 \
--logging_first_step \
--save_steps=1000 \
--save_total_limit=2 \
--evaluate_during_training=true \
--output_dir=/path/to/save/models/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--fp16 \
--patience=5 \
--disable_tqdm=true
```
> For `task_name` and `data_dir`, you can choose one of the follwoing: CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE, and WNLI.

### 4. Fine-tunining on SuperGLUE
Similar to fine-tuning on GLUE, the model can be fine-tuned on SuperGLUE using the following command:
```
python run_sg.py \
--model_path=/path/to/pre-trained/weights/ \
--task_name=<task> \
--tokenizer_name=<tokenizer_name> \
--do_train \
--do_eval \
--learning_rate=1e-5 \
--num_train_epochs=5 \
--warmup_steps=1000 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=128 \
--logging_steps=162 \
--weight_decay=0.01 \
--logging_first_step \
--evaluate_during_training \
--save_total_limit=1 \
--output_dir=/path/to/save/models/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=true \
--load_best_model_at_end \
--fp16 \
--patience=5
```
> For `task_name`, you can choose one of the follwoing: wic, boolq, rte, cb, multirc.

To fine-tuning the models on COPA task, you can run the following command:
```
python run_sg_copa.py \
--model_path=/path/to/pre-trained/weights/ \
--task_name=copa \
--tokenizer_name=<tokenizer_name> \
--do_train \
--do_eval \
--learning_rate=1e-5 \
--num_train_epochs=5 \
--warmup_steps=1000 \
--per_device_train_batch_size=32 \
--per_device_eval_batch_size=128 \
--logging_steps=162 \
--weight_decay=0.01 \
--logging_first_step \
--evaluate_during_training \
--save_total_limit=1 \
--output_dir=/path/to/save/models/ \
--overwrite_output_dir \
--logging_dir=/path/to/save/log/files/ \
--disable_tqdm=true \
--load_best_model_at_end \
--fp16 \
--patience=5
```

## 5. Probing 
First, download the probing tasks data:
```
git clone https://github.com/facebookresearch/SentEval.git
```
The data files for the probing tasks can be accessed in the `SentEval/data/probing/` directory.

Then, run the following command to extract models features for each probing task:
```
cd ./probing
python extract_features.py \
--data_file=/path/to/probing/task/data/file/ \
--output_file=extracted_features_file_name.json \
--output_dir=/path/to/save/extracted/features/ \
--model_path=/path/to/pre-trained/weights/ \
--tokenizer_name=<tokenizer_name>
```
Finally, run the following code to train the classifier for each probing task for a given model layer:
```
cd ../
python probe.py \
--labels_file /path/to/probing/task/data/file/ \
--feats_file /path/to/extracted/features/file.json \
--layer 1 \
--seed 42
```

## Citation  
```
@inproceedings{alajrami-etal-2023-understanding,
    title = "Understanding the Role of Input Token Characters in Language Models: How Does Information Loss Affect Performance?",
    author = "Alajrami, Ahmed  and
      Margatina, Katerina  and
      Aletras, Nikolaos",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.563/",
    doi = "10.18653/v1/2023.emnlp-main.563",
    pages = "9085--9108",
    abstract = "Understanding how and what pre-trained language models (PLMs) learn about language is an open challenge in natural language processing. Previous work has focused on identifying whether they capture semantic and syntactic information, and how the data or the pre-training objective affects their performance. However, to the best of our knowledge, no previous work has specifically examined how information loss in input token characters affects the performance of PLMs. In this study, we address this gap by pre-training language models using small subsets of characters from individual tokens. Surprisingly, we find that pre-training even under extreme settings, i.e. using only one character of each token, the performance retention in standard NLU benchmarks and probing tasks compared to full-token models is high. For instance, a model pre-trained only on single first characters from tokens achieves performance retention of approximately 90{\%} and 77{\%} of the full-token model in SuperGLUE and GLUE tasks, respectively."
}
```

## License
[MIT License](./LICENSE)