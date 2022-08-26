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
python  run_multilin_adapter.py     --output_dir saved_models     --model_name_or_path microsoft/unixcoder-base      --do_train     --train_data_dir dataset/CSN  --test_data_dir dataset/CSN    --eval_data_dir dataset/CSN     --codebase_file dataset/CSN     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 64     --eval_batch_size 64     --learning_rate 5e-5     --seed 123456  2>&1| tee saved_models/multilin_adapter_train.log
```

3. Evaluation
```
python   run_multilin_adapter.py     --output_dir saved_models     --model_name_or_path microsoft/unixcoder-base      --do_test     --train_data_dir dataset/CSN     --test_data_dir dataset/CSN     --codebase_file dataset/CSN     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 64     --eval_batch_size 64     --learning_rate 1e-4     --seed 123456  2>&1| tee saved_models/multilin_adapter_test.log
```

4. Other files
```
+ ./run_adapter.py  # Fine-tune the pre-trained model in a general curriculum learning strategy
+ ./run_multilin.py     # Fine-tune the pre-trained model in the class-based curriculum learning strategy (ours)
```
