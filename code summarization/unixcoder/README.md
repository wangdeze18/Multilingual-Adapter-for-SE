# Code Summarization

1. Download the dataset

```
wget https://github.com/microsoft/CodeXGLUE/raw/main/Code-Text/code-to-text/dataset.zip
unzip dataset.zip
rm dataset.zip
cd dataset
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/python.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/ruby.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/javascript.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/go.zip
wget https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..
```


2. Fine-tuning the multlingual adapter

   You can also skip this step by downloading the [pre-trained model]() directly.
```
python  run_multilin_adapter.py     --output_dir saved_models     --model_name_or_path microsoft/unixcoder-base      --do_train     --train_data_dir dataset/CSN  --test_data_dir dataset/CSN    --eval_data_dir dataset/CSN     --codebase_file dataset/CSN     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 64     --eval_batch_size 64     --learning_rate 5e-5     --seed 123456  2>&1| tee saved_models/multilin_adapter_train.log
```

3. Evaluation
```
python   run_multilin_adapter.py     --output_dir saved_models     --model_name_or_path microsoft/unixcoder-base      --do_test     --train_data_dir dataset/CSN     --test_data_dir dataset/CSN     --codebase_file dataset/CSN     --num_train_epochs 10     --code_length 256     --nl_length 128     --train_batch_size 64     --eval_batch_size 64     --learning_rate 1e-4     --seed 123456  2>&1| tee saved_models/multilin_adapter_test.log
```

4. Other baselines for check
```
./run_adapter.py  # Fine-tuning the monolingual model with adapter
./run_multilin.py  # Fine-tuning the multilingual model with the full model
```
