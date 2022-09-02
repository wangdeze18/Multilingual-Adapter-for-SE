
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import sys
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset

logger = logging.getLogger(__name__)
#

import torch.distributed as dis
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                             RobertaConfig, RobertaModel, RobertaTokenizer)

from torch.utils.data.distributed import DistributedSampler

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data.distributed import Sampler, Dataset
import torch.distributed as dist
from contextlib import nullcontext


from opendelta import AdapterModel
from opendelta import Visualization
#__all__ = ["DistributedSampler_overr", ]

T_co = TypeVar('T_co', covariant=True)


class DistributedSampler_overr(Sampler[T_co]):


    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,bs: int = 0,epoch: int = 10,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = epoch+1
        self.drop_last = drop_last
        self.bs = bs
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        indices_all = []
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            #print("epoch = ",self.epoch)
            for i in range(6):
                lang_len = self.dataset.len_list[i]
                #print(i,lang_len)
            #for lang_len in self.dataset.len_list:
                indices = torch.randperm(lang_len, generator=g).tolist()  # type: ignore[arg-type]

                indices = [i + len(indices_all) for i in indices]
                indices_all.extend(indices)

        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        len_sum = len(indices_all)
        indices_all = indices_all[self.rank:len_sum:self.num_replicas]
        #assert len(indices) == self.num_samples

        return iter(indices_all)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


language = ['java', 'ruby', 'go', 'php', 'javascript', 'python']

class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,

                 ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url


def convert_examples_to_features(js, tokenizer, args):
    """convert examples to token ids"""
    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code_tokens'].split())
    code_tokens = tokenizer.tokenize(code)[:args.code_length - 4]
    code_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.code_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id] * padding_length

    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:args.nl_length - 4]
    nl_tokens = [tokenizer.cls_token, "<encoder-only>", tokenizer.sep_token] + nl_tokens + [tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.nl_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, js['url'] if "url" in js else js["retrieval_idx"])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_dir=None, istest=False, lang=None,epoch=10):
        self.examples = []
        self.len_list = []
        data = []
        if (istest):
            with open(file_dir) as f:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    code_tokens = []
                    lang_str = '<' + lang + '>'
                    code_tokens.append(lang_str)
                    code_tokens.extend(js['code_tokens'])
                    js['code_tokens'] = code_tokens

                    data.append(js)
                self.len_list.append(len(data))
        else:
            pre_sum =0 
            for i in range(min(len(language),epoch+1)):
                lang_i = language[i]
            #for lang_i in language:

                train_file = os.path.join(file_dir, lang_i, 'train.jsonl')
                with open(train_file) as f:
                    for line in f:
                        line = line.strip()
                        js = json.loads(line)
                        if 'function_tokens' in js:
                            js['code_tokens'] = js['function_tokens']
                        code_tokens = []
                        lang_str = '<' + lang_i + '>'
                        code_tokens.append(lang_str)
                        code_tokens.extend(js['code_tokens'])
                        js['code_tokens'] = code_tokens

                        data.append(js)
                self.len_list.append(len(data)-pre_sum)
                pre_sum = len(data)

        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return (torch.tensor(self.examples[i].code_ids), torch.tensor(self.examples[i].nl_ids))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, model, tokenizer):
    """ Train the model """
    # get training dataset
    train_dataset = TextDataset(tokenizer, args, args.train_data_dir)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      num_workers=4)
    else:

        train_sampler = DistributedSampler_overr(train_dataset,bs=args.train_batch_size)
        #train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size // args.gradient_accumulation_steps,
                                      num_workers=4)
    # get optimizer and scheduler

    params = []

    for n ,p in model.named_parameters():
        if 'adapter' in n :#or 'Norm' in n
            #print(n)
            params.append(p)


    optimizer = AdamW(params, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * args.num_train_epochs)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", len(train_dataloader) * args.num_train_epochs)

    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()

    model.train()
    tr_num, tr_loss, best_mrr = 0, 0, 0
    losses = []
    for idx in range(args.num_train_epochs):
        
        if args.local_rank!= -1:
            train_dataloader.sampler.set_epoch(idx)
        for step, batch in enumerate(train_dataloader):
            with torch.autograd.set_detect_anomaly(True):
                #my_context = model.no_sync if args.local_rank != -1 and step % args.gradient_accumulation_steps != 0 else nullcontext
                #with my_context():
                    # get inputs
                nl_input = batch[1].to(args.device)
                code_input = batch[0].to(args.device)

                # get code and nl vectors
                
                nl_vec = model(nl_inputs=nl_input)

                code_vec = model(code_inputs=code_input)

               
                #print("moe loss", moe_loss.item())
                # calculate scores and loss
                scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(scores * 20, torch.arange(code_input.size(0), device=scores.device))
                #print("loss",loss.item())
                #loss += moe_loss
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                losses.append(loss.item())
                # report loss
                tr_loss += loss.item()
                #tr_loss += moe_loss.item()
                tr_num += 1

                    # backward
                loss.backward()
                #if len(losses) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                if len(losses) // args.gradient_accumulation_steps % 100 == 0:
                    logger.info("epoch {} step {} loss {}".format(idx, len(losses) // args.gradient_accumulation_steps, round(tr_loss / tr_num, 5)))
                    tr_loss = 0
                    tr_num = 0

            # evaluate
        # todo: result = multi_lingual_average

        result = 0.0
        for lang in language:
            eval_data_file = os.path.join(args.eval_data_dir, lang, 'valid.jsonl')
            results = evaluate(args, model, tokenizer, eval_data_file, lang, eval_when_training=True)
            result += results['eval_mrr']  ##todo
        results = {
            "eval_mrr": float(result / len(language))
        }
        '''
        results = evaluate(args, model, tokenizer,args.eval_data_dir, eval_when_training=True)
        results = {
        "eval_mrr":results['eval_mrr']
        }
        '''
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

            # save best model
        if results['eval_mrr'] > best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  " + "*" * 20)
            logger.info("  Best mrr:%s", round(best_mrr, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = 'checkpoint-best-mrr'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            output_dir1 = os.path.join(output_dir, '{}'.format('multilin_adapter_model.bin'))
            torch.save(model_to_save.state_dict(), output_dir1)
            logger.info("Saving model checkpoint to %s", output_dir1)


def evaluate(args, model, tokenizer, eval_data_file ,lang, eval_when_training=False):
    
    codebase_name = os.path.join(args.eval_data_dir, lang, 'codebase.jsonl')
    query_dataset = TextDataset(tokenizer, args, eval_data_file, lang=lang, istest=True)
    # query_sampler = SequentialDistributedSampler(query_dataset)
    # query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    code_dataset = TextDataset(tokenizer, args, codebase_name, lang=lang, istest=True)
    # code_sampler = SequentialDistributedSampler(code_dataset)
    # code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # query_dataset = TextDataset(tokenizer, args, file_name)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # code_dataset = TextDataset(tokenizer, args, args.codebase_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in query_dataloader:
        nl_inputs = batch[1].to(args.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs)
            nl_vecs.append(nl_vec.cpu().numpy())

    for batch in code_dataloader:
        code_inputs = batch[0].to(args.device)
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())
    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)

    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }

    return result

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_dir", default=None, type=str,
                        help="The input training data dir.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_data_dir", default=None, type=str,
                        help="An optional input evaluation data dir to evaluate the MRR.")
    parser.add_argument("--test_data_dir", default=None, type=str,
                        help="An optional input test data file to test the MRR(a josnl file).")
    parser.add_argument("--codebase_file", default=None, type=str,
                        help="An optional input test data file to codebase .")

    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_zero_shot", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_F2_norm", action='store_true',
                        help="Whether to run eval on the test set.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--local_rank', default=-1 ,type=int,
                        help="random seed for initialization")
    parser.add_argument("--apply_adapter", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--apply_expert_soup", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument('--adapter_size', type=int, default=16,
                        help="random seed for initialization")
    parser.add_argument('--num_experts', type=int, default=4,
                        help="random seed for initialization")
    parser.add_argument('--inference_level', type=int, default=1,
                        help="random routing")
    parser.add_argument('--use_consistency_loss', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--sharing_down', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--sharing_up', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument("--adapter_type", default="houlsby", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    #set_seed(args.seed)
    
    if args.local_rank == -1:
        set_seed(args.seed)
    else:
        rank = torch.distributed.get_rank()
        # logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
        set_seed(args.seed + rank)
    
    # build model
    tokenizer = RobertaTokenizer.from_pretrained('./unixcoder', local_files_only=True)
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    model = RobertaModel.from_pretrained('./unixcoder', local_files_only=True)

    delta_model = AdapterModel(backbone_model=model, modified_modules=['attention', '[r](\d)+\.output'], bottleneck_dim=128)
    # delta_model.freeze_module(set_state_dict = True)
    delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
    # Visualization(model).structure_graph()

    # add spec tokens to tokenizer
    special_tokens_dict = {'additional_special_tokens': ['<java>', '<ruby>', '<go>', '<php>', '<javascript>', '<python>']}

    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    model.resize_token_embeddings(len(
        tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

    # assert '<java>' in tokenizer.lang_token
    #print(model)
    model = Model(model)
    logger.info("Training/evaluation parameters %s", args)

    model.to(args.device)
    if args.local_rank != -1:
        model = DDP(model,broadcast_buffers=False,find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

        # Training
    if args.do_train:
        train(args, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/multilin_adapter_model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        for lang in language:
            result = evaluate(args, model, tokenizer, args.eval_data_dir, lang)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key], 3)))

    if args.do_test:
        if args.do_zero_shot is False:
            checkpoint_prefix = 'checkpoint-best-mrr/multilin_adapter_model_copy.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            model_to_load = model.module if hasattr(model, 'module') else model
            model_to_load.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        for lang in language:
            eval_data_file = os.path.join(args.eval_data_dir, lang, 'test.jsonl')
            result = evaluate(args, model, tokenizer,eval_data_file,lang)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key], 3)))


if __name__ == "__main__":
    main()
