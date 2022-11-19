
# Code Search

1. Download the dataset


```
mkdir dataset && cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```


2. Fine-tuning the multlingual adapter

```
python -m torch.distributed.launch --nproc_per_node 4 run_multilin_adapter.py --output_dir=./saved_models/ --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=microsoft/codebert-base --do_train --train_data_file=dataset/CSN --eval_data_file=dataset/CSN --test_data_file=dataset/CSN --codebase_file=dataset/CSN --num_train_epochs 10 --code_length 256 --nl_length 128 --train_batch_size 32 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 2>&1| tee saved_models/multilin_adapter_train.log

```

3. Evaluation
```
python -m torch.distributed.launch --nproc_per_node 4 run_multilin_adapter.py --output_dir=./saved_models/ --config_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base --tokenizer_name=microsoft/codebert-base --do_test --train_data_file=dataset/CSN --eval_data_file=dataset/CSN --test_data_file=dataset/CSN --codebase_file=dataset/CSN --num_train_epochs 10 --code_length 256 --nl_length 128 --train_batch_size 32 --eval_batch_size 64 --learning_rate 2e-5 --seed 123456 2>&1| tee saved_models/multilin_adapter_test.log
```
