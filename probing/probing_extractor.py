
import os
import sys
import json
import torch
import pickle
import collections

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers import BartTokenizer, AutoModelForSeq2SeqLM, BartConfig
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import T5Config,T5ForConditionalGeneration

from transformers import AutoModelForMaskedLM, RobertaForMaskedLM
from opendelta import AdapterModel
from opendelta import Visualization


class InputExample(object):
    def __init__(self, text, unique_id):
        self.text = text
        self.unique_id = unique_id


class InputFeatures(object):
    def __init__(self, tokens, unique_id, input_ids, input_mask, input_type_ids):
        self.tokens = tokens
        self.unique_id = unique_id

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def read_examples(text_file):
    examples = []
    unique_id = 0

    with open(text_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line: break

            text = line.strip().split('\t')[-1]
            examples.append(InputExample(text=text, unique_id=unique_id))
            unique_id += 1
    return examples


def convert_examples_to_features(examples, seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        cand_tokens = tokenizer.tokenize(example.text)
        if len(cand_tokens) > seq_length - 2:
            ## Account for [CLS] and [SEP] with "- 2"
            cand_tokens = cand_tokens[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []

        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in cand_tokens:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(tokens=tokens, unique_id=example.unique_id, input_ids=input_ids, input_mask=input_mask,
                          input_type_ids=input_type_ids))
    return features


def get_max_seq_length(samples, tokenizer):
    max_seq_len = -1
    for sample in samples:
        cand_tokens = tokenizer.tokenize((sample.text))
        cur_len = len(cand_tokens)
        if cur_len > max_seq_len:
            max_seq_len = cur_len

    # *************************************
    if max_seq_len > model_max_seq_length:
        max_seq_len = model_max_seq_length
    # *************************************

    return max_seq_len


def save_features(model, tokenizer, device):
    # convert data to ids
    examples = read_examples(text_dataset)
    features = convert_examples_to_features(examples=examples, seq_length=(get_max_seq_length(examples, tokenizer)),
                                            tokenizer=tokenizer)

    # extract and write features
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_example_indices = torch.arange(all_input_ids.size(0),
                                       dtype=torch.long)  # gives => tensor([0,1, 2, ... (num_samples - 1) ])
    eval_dataset = TensorDataset(all_input_ids, all_input_mask, all_example_indices)
    eval_dataloader = DataLoader(eval_dataset, sampler=SequentialSampler(eval_dataset), batch_size=batchsize)

    pbar = tqdm(total=len(examples) // batchsize)
    with open(json_features, "w") as writer:
        with torch.no_grad():
            for input_ids, input_mask, example_indices in eval_dataloader:  # batch_sized input_ids, input_mask, example_indices tensor
                input_ids = input_ids.to(device)  # batch_sized input_ids tensor
                input_mask = input_mask.to(device)  # batch_sized input_mask tensor
                if "plbart" in model.__dict__["config"]._name_or_path:
                    all_outputs = model(input_ids=input_ids)  # , token_type_ids=None, attention_mask=input_mask)
                    enc_layers = all_outputs.encoder_hidden_states
                elif "codet5" in model.__dict__["config"]._name_or_path:
                    all_outputs = model(input_ids=input_ids,
                                        decoder_input_ids=input_ids)  # , token_type_ids=None, attention_mask=input_mask)
                    enc_layers = all_outputs.encoder_hidden_states
                else:
                    all_outputs = model(input_ids=input_ids, token_type_ids=None, attention_mask=input_mask,output_hidden_states=True)
                    #print("all_outputs", all_outputs)
                    enc_layers = all_outputs.hidden_states
                    # print("***************************************************")

                #print(enc_layers)
                print(model_checkpoint, " => Num layers:", len(enc_layers))
                print("***************************************************")

                for iter_index, example_index in enumerate(example_indices):
                    # for every feature in batch => tokens, input_ids, input_mask => features[example_index.item()]
                    feature = features[example_index.item()]  # example_indices are i,j,k, ... till batch_size
                    unique_id = int(feature.unique_id)

                    all_output_features = []
                    for (token_index, token) in enumerate(feature.tokens):
                        all_layers = []
                        for layer_index in range(len(enc_layers)):
                            layer_output = enc_layers[
                                int(layer_index)]  # layer   layer_index (#0, #1, #2 ... max_layers)
                            layer_feat_output = layer_output[iter_index]  # feature iter_index

                            layers = collections.OrderedDict()
                            layers["index"] = layer_index
                            layers["values"] = [round(hidden_unit.item(), 6) for hidden_unit in layer_feat_output[
                                token_index]]  # layer layer_index, feature iter_index, token token_index
                            all_layers.append(layers)

                        out_features = collections.OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_output_features.append(out_features)
                        break  # if breaking only [CLS] token will be considered for classification

                    output_json = collections.OrderedDict()
                    output_json["linex_index"] = unique_id
                    output_json["features"] = all_output_features
                    writer.write(json.dumps(output_json) + "\n")

                pbar.update(1)
    pbar.close()
    print('written features to %s' % (json_features))


if __name__ == '__main__':

    task_codes = ['CPX', 'LEN','TYP']  # ['AST', 'CPX', 'CSC', 'IDF', 'IDT', 'JBL', 'JFT', 'JMB', 'LEN', 'MXN', 'NML', 'NMS', 'NPT', 'OCT', 'OCU', 'REA', 'SCK', 'SRI', 'SRK', 'TAN', 'TYP', 'VCT', 'VCU']
    shuffle_kinds = ['ORIG']
    label_counts = ['10k']  # , '10k'

    model_checkpoints = {

        "BERT": "bert-base-uncased",
        "BERT_adapter": "BERT_adapter",
        "CodeBERT": "microsoft/codebert-base",
        "CodeBERT_adapter": "microsoft/codebert-base",
        "CodeBERTa": "huggingface/CodeBERTa-small-v1",
        "CodeBERTa_adapter": "huggingface/CodeBERTa-small-v1",
        "GraphCodeBERT": "microsoft/graphcodebert-base",
        "GraphCodeBERT_adapter": "microsoft/graphcodebert-base",

    }

    model_checkpoints_full = {
        "BERT": "bert-base-uncased",
        "BERT_adapter": "BERT_adapter",
        "CodeBERT": "microsoft/codebert-base",
        "CodeBERT_adapter": "microsoft/codebert-base",
        "CodeBERTa": "huggingface/CodeBERTa-small-v1",
        "CodeBERTa_adapter": "huggingface/CodeBERTa-small-v1",
        "GraphCodeBERT": "microsoft/graphcodebert-base",
        "GraphCodeBERT_adapter": "microsoft/graphcodebert-base",
        "CodeT5": "Salesforce/codet5-base",
        "CodeT5_adapter": "Salesforce/codet5-base",
        "CodeT5_multilin_adapter": "Salesforce/codet5-base",
        "CodeT5_multlin": "Salesforce/codet5-base",
        "JavaBERT-mini": "anjandash/JavaBERT-mini",
        "JavaBERT-mini_adapter": "anjandash/JavaBERT-mini",
        "PLBART-mtjava": "uclanlp/plbart-multi_task-java",
        "PLBART-mtjava_adapter": "uclanlp/plbart-multi_task-java",
        "PLBART": "uclanlp/plbart-base",
        "PLBART_adapter": "uclanlp/plbart-base",
        "PLBART-large": "uclanlp/plbart-large",
        "PLBART-large_adapter": "uclanlp/plbart-large",
        "t5": "t5-base",
        "BERT_finetune": "bert-base-uncased",
        "BERT_finetune_adapter": "bert-base-uncased",
        "BERT_finetune_init_adapter": "bert-base-uncased",
    }
    '''
    model_max_seq_lengths = {
        "BERT": 512,
        "BERT_adapter": 512,
        "CodeBERT": 512,
        "CodeBERT_adapter": 512,
        "CodeBERTa": 512,
        "CodeBERTa_adapter": 512,
        "GraphCodeBERT": 512,
        "GraphCodeBERT_adapter": 512,
        "CodeT5": 512,
        "CodeT5_adapter": 512,
        "CodeT5_multilin_adapter": 512,
        "CodeT5_multlin": 512,
        "JavaBERT-mini": 512,
        "JavaBERT-mini_adapter": 512,
        "PLBART": 1024,
        "PLBART_adapter": 1024,
        "PLBART-mtjava": 1024,
        "PLBART-mtjava_adapter": 1024,
        "PLBART-large": 1024,
        "PLBART-large_adapter": 1024,
        "t5": 512,
        "BERT_finetune": 512,
        "BERT_finetune_adapter": 512,
        "BERT_finetune_init_adapter": 512,
    }
    '''
    for task_code in task_codes:
        for shuffle_kind in shuffle_kinds:
            for model_checkpoint in list(model_checkpoints.keys()):
                for label_count in label_counts:
                    print("********")
                    print(f"Processing for task >> {task_code} >> {shuffle_kind}:{model_checkpoint} for {label_count}")
                    print("********")

                    text_dataset = sys.path[
                                       0] + '/data/datasets_' + task_code + '/' + task_code + '_' + shuffle_kind + '_' + label_count + '.txt'
                    json_features = sys.path[
                                        0] + '/data/datasets_' + task_code + '/' + shuffle_kind + '/' + model_checkpoint + '_features_' + label_count + '.json'

                    if not os.path.exists(json_features):
                        path = Path(json_features)
                        path.parent.mkdir(parents=True, exist_ok=True)

                        # *******************************************************

                    modelname = model_checkpoints.get(model_checkpoint, None)
                    #model_max_seq_length = model_max_seq_lengths.get(model_checkpoint, None)
                    model_max_seq_length = 512

                    device = torch.device("cpu")
                    batchsize = 8  # 8 for 512 tokens 4 for 1024 tokens

                    if model_checkpoint == "BERT":

                        config = BertConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=True, cache_dir="~/tmp")
                        model = BertModel.from_pretrained(modelname, config=config, cache_dir="~/tmp")

                    elif model_checkpoint in ["CodeBERT", "CodeBERTa", "GraphCodeBERT"]:

                        config = RobertaConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = RobertaTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model = RobertaForSequenceClassification.from_pretrained(modelname, config=config,
                                                                                 cache_dir="~/tmp")
                    elif model_checkpoint == "BERT_adapter":

                        config = BertConfig.from_pretrained(modelname)
                        tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=True,
                                                                  cache_dir="~/tmp")
                        # model = BertModel.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        model = BertForMaskedLM.from_pretrained(modelname, config=config, cache_dir="~/tmp")

                        delta_model = AdapterModel(backbone_model=model,
                                                   modified_modules=['attention', '[r](\d)+\.output'],
                                                   bottleneck_dim=128)  # This will apply adapter to the self-attn and feed-forward layer.
                        # delta_model.freeze_module(set_state_dict=True)
                        if 'adapter' in model_checkpoint:
                            path = './saved_models/bert/pytorch_model.bin'
                            pretrained_dict = torch.load(path)
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                if k in pretrained_dict:
                                    model_dict[k] = pretrained_dict[k]
                            model.load_state_dict(model_dict)
                        # delta_model.log()
                    elif model_checkpoint == "BERT_finetune":

                        config = BertConfig.from_pretrained(modelname)
                        tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=True,
                                                                  cache_dir="~/tmp")
                        model = BertModel.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        # model = BertForMaskedLM.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        path = './saved_models/bert/bert_model.bin'
                        pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                        model_dict = model.state_dict()
                        # for k,v in model_dict.items():
                        #    print(k)
                        # for k,v in pretrained_dict.items():
                        #    print(k)

                        for k, v in model_dict.items():
                            pretrained_k = 'encoder.' + k
                            print("k = ", k)
                            if pretrained_k in pretrained_dict:
                                print("k in ,", pretrained_k)
                                model_dict[k] = pretrained_dict[pretrained_k]
                        model.load_state_dict(model_dict)
                        # delta_model.log()
                    elif model_checkpoint == "BERT_finetune_adapter":

                        config = BertConfig.from_pretrained(modelname)
                        tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=True,
                                                                  cache_dir="~/tmp")
                        # model = BertModel.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        model = BertForMaskedLM.from_pretrained(modelname, config=config, cache_dir="~/tmp")

                        delta_model = AdapterModel(backbone_model=model,
                                                   modified_modules=['attention', '[r](\d)+\.output'],
                                                   bottleneck_dim=128)  # This will apply adapter to the self-attn and feed-forward layer.
                        # delta_model.freeze_module(set_state_dict=True)
                        if 'adapter' in model_checkpoint:
                            path = './saved_models/bert_adapter/model.bin'
                            pretrained_dict = torch.load(path)
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                pretrained_k = 'encoder.' + k[5:]
                                if pretrained_k in pretrained_dict:
                                    # print("k in ,", pretrained_k)
                                    model_dict[k] = pretrained_dict[pretrained_k]
                            model.load_state_dict(model_dict)
                        # delta_model.log()
                    elif model_checkpoint == "BERT_finetune_init_adapter":

                        config = BertConfig.from_pretrained(modelname)
                        tokenizer = BertTokenizer.from_pretrained(modelname, do_lower_case=True,
                                                                  cache_dir="~/tmp")
                        # model = BertModel.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        model = BertForMaskedLM.from_pretrained(modelname, config=config, cache_dir="~/tmp")

                        delta_model = AdapterModel(backbone_model=model,
                                                   modified_modules=['attention', '[r](\d)+\.output'],
                                                   bottleneck_dim=128)  # This will apply adapter to the self-attn and feed-forward layer.
                        # delta_model.freeze_module(set_state_dict=True)
                        if 'adapter' in model_checkpoint:
                            path = './saved_models/bert_init_adapter/model.bin'
                            pretrained_dict = torch.load(path)
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                pretrained_k = 'encoder.' + k[5:]
                                if pretrained_k in pretrained_dict:
                                    # print("k in ,", pretrained_k)
                                    model_dict[k] = pretrained_dict[pretrained_k]
                            model.load_state_dict(model_dict)
                        # delta_model.log()
                    elif model_checkpoint in ["CodeBERT_adapter", "CodeBERTa_adapter", "GraphCodeBERT_adapter"]:
                        config = RobertaConfig.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = RobertaTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model = RobertaForSequenceClassification.from_pretrained(modelname, config=config,
                                                                                 cache_dir="~/tmp")
                        # model = RobertaForSequenceClassification.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        # Visualization(model).structure_graph()
                        delta_model = AdapterModel(backbone_model=model,
                                                   modified_modules=['attention', '[r](\d)+\.output'],
                                                   bottleneck_dim=128)  # This will apply adapter to the self-attn and feed-forward layer.
                        # delta_model.freeze_module(set_state_dict=True)
                        if 'adapter' in model_checkpoint:
                            path = './PretrainingBERT-main/model/' + model_checkpoint + '/' + 'pytorch_model.bin'
                            pretrained_dict = torch.load(path)
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                if k in pretrained_dict:
                                    model_dict[k] = pretrained_dict[k]
                            model.load_state_dict(model_dict)
                        # delta_model.log()


                    elif model_checkpoint in ["CodeT5_multilin_adapter","CodeT5_multilin"]:  #######################
                        config = T5Config.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = RobertaTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model = T5ForConditionalGeneration.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        Visualization(model).structure_graph()

                        # delta_model.freeze_module(set_state_dict=True)
                        if 'adapter' in model_checkpoint:
                            delta_model = AdapterModel(backbone_model=model,
                                                       modified_modules=['layer.0', 'layer.2',
                                                                         '[r]encoder\.block\.(\d)+\.layer\.[01]'],
                                                       # modified_modules=['T5LayerSelfAttention', 'T5LayerFF'],
                                                       bottleneck_dim=128)  # This will apply adapter to the self-attn and feed-forward layer.
                            path = './saved_models/code_summarization/model_codet5/multilin_models/multilin_adapter_model.bin'
                            pretrained_dict = torch.load(path,map_location=torch.device('cpu'))
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                if k in pretrained_dict:
                                    #print(k)
                                    model_dict[k] = pretrained_dict[k]
                                else:
                                    print("not in ",k)
                            model.load_state_dict(model_dict)
                        else:

                            path = './saved_models/code_summarization/model_codet5/multilin_models/multilin_model.bin'
                            pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                if k in pretrained_dict:
                                    # print(k)
                                    model_dict[k] = pretrained_dict[k]
                                else:
                                    print("not in ", k)
                            model.load_state_dict(model_dict)
                        #    path = '/saved_models/code_summarization/mode_codet5/multilin_models/multilin_adapter_model.bin'
                        # delta_model.log()
                    elif model_checkpoint in ["codet5_java","codet5_ruby","codet5_python","codet5_php","codet5_go","codet5_javascript",
                                              "codet5_adapter_java","codet5_adapter_ruby","codet5_adapter_python","codet5_adapter_php","codet5_adapter_go","codet5_adapter_javascript"]:

                        language = model_checkpoint.split('_')[1]
                        config = T5Config.from_pretrained(modelname, output_hidden_states=True)
                        tokenizer = RobertaTokenizer.from_pretrained(modelname, cache_dir="~/tmp")
                        model = T5ForConditionalGeneration.from_pretrained(modelname, config=config, cache_dir="~/tmp")
                        #Visualization(model).structure_graph()


                        if 'adapter' in model_checkpoint:
                            delta_model = AdapterModel(backbone_model=model,
                                                       modified_modules=['layer.0', 'layer.2',  ###### may change!!!
                                                                         '[r]encoder\.block\.(\d)+\.layer\.[01]'],
                                                       # modified_modules=['T5LayerSelfAttention', 'T5LayerFF'],
                                                       bottleneck_dim=128)  # This will apply adapter to the self-attn and feed-forward layer.
                            path = './saved_models/code_summarization/model_codet5/separate_models/'+language+'/checkpoint-best-bleu/adapter_model.bin'
                            pretrained_dict = torch.load(path,map_location=torch.device('cpu'))
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                if k in pretrained_dict:
                                    #print(k)
                                    model_dict[k] = pretrained_dict[k]
                                else:
                                    print("not in ",k)
                            model.load_state_dict(model_dict)
                        else:
                            path = './saved_models/code_summarization/model_codet5/separate_models/'+language+'/checkpoint-best-bleu/model.bin'
                            pretrained_dict = torch.load(path, map_location=torch.device('cpu'))
                            model_dict = model.state_dict()
                            for k, v in model_dict.items():
                                if k in pretrained_dict:
                                    # print(k)
                                    model_dict[k] = pretrained_dict[k]
                                else:
                                    print("not in ", k)
                            model.load_state_dict(model_dict)

                    print("-----")
                    #print("Vocabulary  Size:\t", model.config.vocab_size)
                    #print("Tokenizer Length:\t", len(tokenizer))
                    print("-----")
                    model.to(device)
                    model.eval()
                    save_features(model, tokenizer, device)
                    print("********")
