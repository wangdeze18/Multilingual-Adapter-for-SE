# Code Search

1. Download the dataset


```
mkdir dataset && cd dataset
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
cd ../..
```


2. Fine-tune pre-trained models

   You can also skip this step by downloading the [pre-trained model]() directly.
```
python run_curri.py --output_dir=./saved_models/java --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_train --train_data_file=dataset/java/large_train_test.jsonl --eval_data_file=dataset/java/valid.jsonl --test_data_file=dataset/java/test.jsonl --codebase_file=dataset/java/codebase.jsonl --num_train_epochs 20 --code_length 256 --data_flow_length 64 --nl_length 128 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --pacing_function linear --seed 123456 2>&1| tee saved_models/java/large_train.log;
```

3. Evaluate pre-trained models on the original dataset
```
python run_curri.py --output_dir=./saved_models/java --config_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base --tokenizer_name=microsoft/graphcodebert-base --lang=java --do_test --train_data_file=dataset/java/large_train_test.jsonl --eval_data_file=dataset/java/valid.jsonl --test_data_file=dataset/java/test.jsonl --codebase_file=dataset/java/codebase.jsonl --num_train_epochs 10 --code_length 256 --data_flow_length 64 --nl_length 128 --train_batch_size 128 --eval_batch_size 256 --learning_rate 2e-5 --pacing_function linear --seed 123456 2>&1| tee saved_models/java/large_test.log;
```
