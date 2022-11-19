lr=5e-5
batch_size=32
beam_size=10
source_length=256
target_length=128
data_dir=../dataset
output_dir=model/
train_file=$data_dir/
dev_file=$data_dir/
epochs=10
pretrained_model=microsoft/codebert-base #Roberta: roberta-base

python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs




batch_size=64

test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

test_file=$data_dir/java/test.jsonl
python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model  --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
test_file=$data_dir/php/test.jsonl
python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model  --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

test_file=$data_dir/go/test.jsonl
python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model  --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

test_file=$data_dir/ruby/test.jsonl

python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model  --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

test_file=$data_dir/python/test.jsonl
python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model  --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

test_file=$data_dir/javascript/test.jsonl
python -m torch.distributed.launch --nproc_per_node 4  run_multilin_adapter.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model  --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size
