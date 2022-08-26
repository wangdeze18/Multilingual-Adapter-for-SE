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
python  run_multilin_adapter.py --do_train --do_eval --model_name_or_path microsoft/unixcoder-base --train_dir dataset --dev_dir dataset --output_dir saved_models/$lang --max_source_length 256 --max_target_length 128 --beam_size 10 --train_batch_size 48 --eval_batch_size 48 --learning_rate 5e-5 --gradient_accumulation_steps 2 --num_train_epochs 10  2>&1| tee saved_models/multilin_adapter_train.log
```

3. Evaluation
```
python  run_multilin_adapter.py --do_test --model_name_or_path microsoft/unixcoder-base --test_dir dataset --output_dir saved_models/$lang --max_source_length 256 --max_target_length 128 --beam_size 10 --train_batch_size 48 --eval_batch_size 48 --learning_rate 5e-5 --gradient_accumulation_steps 2 --num_train_epochs 10  2>&1| tee saved_models/multilin_adapter_test.log
```

4. Other baselines for check
```
./run_adapter.py  # Fine-tuning the monolingual model with adapter
./run_multilin.py  # Fine-tuning the multilingual model with the full model
```
