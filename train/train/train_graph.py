# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import numpy as np
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
from torch.utils.data import Dataset
from graphgpt.train.graphchat_trainer import GraphChatTrainer

from graphgpt import conversation as conversation_lib
from graphgpt.model import *

from PIL import Image
import torch.nn as nn
from torch_geometric.data import Data

import logging
import time
# TODO: import and use code from ../data/dataset.py

from graphgpt.utils.utils import *
from graphgpt.utils.util_data import *
from graphgpt.utils.util_neighbor import *
from graphgpt.utils.util_negative import *
from collections import Counter

IGNORE_INDEX = -100
GRAPH_TOKEN_INDEX = -200
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_graph_mlp_adapter: bool = field(default=False)
    graph_tower: Optional[str] = field(default=None)
    graph_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_graph_mlp_adapter: Optional[str] = field(default=None)
    use_graph_start_end: bool = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_graph: bool = False
    sep_graph_conv_front: bool = False
    graph_token_len: int = 0
    graph_content: Optional[str] = field(default=None)
    graph_data_path: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    sample_neighbor_strategy: str = field(default='recent')
    sample_neighbor_size: Optional[int] = field(default=-1)
    time_scaling_factor: float = field(default=1e-6)
    threshold: float = field(default=0.5)
    use_dataset:Optional[str] = field(default="arxiv")
    negative_sample: str = field(default='entity')

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_graph_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool =False


def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_graph(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len = cur_token_len
    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
            if graph_cfg['use_graph_start_end']:
                replace_token = DEFAULT_G_START_TOKEN + replace_token + DEFAULT_G_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    return sources

def preprocess_graph_LP(
    sources: Sequence[str],
    graph_cfg: dict,
    cur_token_len_1: int,
    cur_token_len_2: int,
) -> Dict:
    is_graph = graph_cfg['is_graph']
    # image_token_len = multimodal_cfg['image_token_len']
    graph_token_len_1 = cur_token_len_1
    graph_token_len_2 = cur_token_len_2

    if not is_graph:
        return sources

    for source in sources:
        if graph_cfg['sep_graph_conv_front']:
            assert DEFAULT_GRAPH_TOKEN in source[0]['value']
            source[0]['value'] = source[0]['value'].replace(DEFAULT_GRAPH_TOKEN, '').strip()
            source[0]['value'] = DEFAULT_GRAPH_TOKEN + conversation_lib.default_conversation.sep + conversation_lib.default_conversation.roles[0] + ": " + source[0]['value']
        for sentence in source:
            replace_token_1 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_1
            replace_token_2 = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len_2
            if graph_cfg['use_graph_start_end']:
                replace_token_1 = DEFAULT_G_START_TOKEN + replace_token_1 + DEFAULT_G_END_TOKEN
                replace_token_2 = DEFAULT_G_START_TOKEN + replace_token_2 + DEFAULT_G_END_TOKEN

            if DEFAULT_GRAPH_TOKEN in sentence["value"]:
                first_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:first_index] + replace_token_1 + sentence["value"][first_index+len(DEFAULT_GRAPH_TOKEN):]

                # 替换第二个<graph>为B
                second_index = sentence["value"].find(DEFAULT_GRAPH_TOKEN)
                sentence["value"] = sentence["value"][:second_index] + replace_token_2 + sentence["value"][second_index+len(DEFAULT_GRAPH_TOKEN):]


            # sentence["value"] = sentence["value"].replace(DEFAULT_GRAPH_TOKEN, replace_token)

    # print(sources)

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    logging.info(conversations)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_graph: bool = True
) -> Dict:


    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    # print(conversations)



    if has_graph:
        input_ids = torch.stack([tokenizer_graph_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # print(input_ids)
    
    # Mask targets
    for conversation, target in zip(conversations, targets):
        total_len = target.shape[0]

        rounds = conversation.split('<|eot_id|>')

        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
         
            header_start_idx = rou.find('<|start_header_id|>')
            header_end_idx = rou.find('<|end_header_id|>')
            start_idx = header_start_idx + len('<|start_header_id|>')
            end_idx = header_end_idx
            role_name = rou[start_idx:end_idx]

            rou += '<|eot_id|>'
            if role_name == 'SYSTEM':      
                round_len = len(tokenizer(rou).input_ids)     
                target[:round_len] = IGNORE_INDEX
                cur_len += round_len
            elif role_name == conv.roles[0]:
                if has_graph:
                    round_len = len(tokenizer_graph_token(rou, tokenizer))-1
                else:
                    round_len = len(tokenizer(rou).input_ids)-1
                target[cur_len: cur_len+round_len] = IGNORE_INDEX
                cur_len += round_len
            elif role_name == conv.roles[1]:
                newline_idx = rou.find('\n\n')
                instruction = rou[:newline_idx+2]
                instruction_len = len(tokenizer(instruction).input_ids)-1
                target[cur_len: cur_len+instruction_len] = IGNORE_INDEX
                round_len = len(tokenizer(rou).input_ids)-1
                cur_len += round_len
            else:
                raise ValueError("unknown role")
        
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    # print(target)
    # exit()


    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        sources = [example["conversations"] for example in list_data_dict]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict, 
                 **kwargs,):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        self.use_dataset = graph_cfg['use_dataset']
        self.data_path = data_path
        self.full_data, self.train_data, _,_,_,_, self.cat_num = get_temporal_data(self.data_path, self.use_dataset,0.15,0.15)
        self.tokenizer = tokenizer

        self.sample_neighbor_size = graph_cfg['sample_neighbor_size']
        self.threshold = graph_cfg['threshold'] / 10.0
        self.seed = graph_cfg['seed']
        self.sample_neighbor_strategy = graph_cfg['sample_neighbor_strategy']
        self.time_scaling_factor = graph_cfg['time_scaling_factor']
        self.negative_sample = graph_cfg['negative_sample']

        self.neighbor_sampler = get_neighbor_sampler(data=self.train_data, sample_neighbor_strategy=self.sample_neighbor_strategy,
                                                time_scaling_factor=self.time_scaling_factor, seed=self.seed)
       
        self.interaction_index = np.arange(self.train_data.num_interactions)
        # self.interaction_index = np.random.choice(self.interaction_index,10000,replace=False)
        self.interaction_index = np.random.default_rng(self.seed).choice(self.interaction_index,10000,replace=False)
        self.interaction_index = np.sort(self.interaction_index)


    def __len__(self):
        return len(self.interaction_index)
    

    def truncate_str(self,content,max_length=10):

        contents = content.split(' ')
        contents = contents[:max_length]
        new_str = ' '.join(contents)
        if new_str[-1] == '.':
            new_str = new_str[:-1]
        
        return new_str

    def process_text(self,id,text):
        if self.use_dataset == 'Enron':
            text = text.split('@')[0]
        elif self.use_dataset == 'Googlemap_CT':
            text = text.split('.')[0]
        elif self.use_dataset == 'ICEWS1819':
            text = text.split(':')[0]
        elif self.use_dataset == 'Amazon_movies':
            if id >= 233460:
                text = text.split('.')[0]
                text = text.split(':')[1].strip()
        else:
            pass

        return text

    # def get_source(self, i):    


    #     src_id = self.train_data.src_node_ids[self.interaction_index[i]]
    #     src_id_text = self.train_data.entity_texts[src_id]
    #     src_id_text = self.process_text(src_id,src_id_text)


    #     tgt_id = self.train_data.dst_node_ids[self.interaction_index[i]]
    #     tgt_id_text = self.train_data.entity_texts[tgt_id]
    #     tgt_id_text = self.process_text(tgt_id,tgt_id_text)


    #     interaction_time = self.train_data.node_interact_times[self.interaction_index[i]]
    #     final_interaction_time = self.train_data.node_interact_times[-1]
    #     src_nodes_neighbor_ids, _, src_nodes_neighbor_times = self.neighbor_sampler.get_all_first_hop_neighbors([src_id],[final_interaction_time+1])
    #     # tgt_nodes_neighbor_ids, _, tgt_nodes_neighbor_times = self.neighbor_sampler.get_all_first_hop_neighbors([tgt_id],[final_interaction_time+1])


    #     positive_flag = np.random.rand() >= self.threshold
    #     if not positive_flag:
    #         if self.negative_sample == 'temporal':
    #             mask = src_nodes_neighbor_ids[0] == tgt_id 
    #             interaction_time = np.setdiff1d(self.train_data.node_interact_times,src_nodes_neighbor_times[0][mask])
            
                
    #             logging.info(f'len {src_nodes_neighbor_times[0][mask].shape}')
    #             logging.info(f'len total {self.train_data.node_interact_times.shape}')
    #             logging.info(f'random {interaction_time.shape}')
    #             interaction_time = np.random.choice(interaction_time)
    #         elif self.negative_sample == 'entity':
    #             mask = src_nodes_neighbor_times[0] == interaction_time
    #             tgt_id = set(self.train_data.dst_node_ids.tolist()) - set([src_id]) - set(src_nodes_neighbor_ids[0][mask].tolist())
    #             tgt_id = np.random.choice(np.array(list(tgt_id)))
    #             tgt_id_text = self.train_data.entity_texts[tgt_id]
    #             tgt_id_text = self.process_text(tgt_id,tgt_id_text)  
    #             # tgt_nodes_neighbor_ids, _, tgt_nodes_neighbor_times = self.neighbor_sampler.get_all_first_hop_neighbors([tgt_id],[final_interaction_time+1])
    #         else:
    #             raise ValueError('unknown negative sample')

    #     graph_token_id_1 = []
    #     graph_token_id_2 = []
    #     graph_timestamp = []

    #     graph_timestamp += [interaction_time]

    #     if self.use_dataset == 'Enron':
    #         conversations = [{"from": "human", "value": f"This temporal graph represents the email communications between employees. Before timestamp {int(interaction_time)}, the employees who recently exchanged emails with {src_id_text} are represented by the following sequence of graph tokens:"}]
    #     elif self.use_dataset == 'GDELT':
    #         conversations= [{"from": "human", "value": f"This temporal graph represents the political behaviors between political entities. Before timestamp {int(interaction_time)}, the political entities who recently engaged in political relationship with {src_id_text} are represented by the following sequence of graph tokens:"}]
    #     elif self.use_dataset == 'ICEWS1819':
    #         conversations= [{"from": "human", "value": f"This temporal graph represents the political behaviors between political entities. Before timestamp {int(interaction_time)}, the political entities who recently engaged in political relationship with {src_id_text} are represented by the following sequence of graph tokens:"}]
    #     elif self.use_dataset == 'Googlemap_CT':
    #         conversations= [{"from": "human", "value": f"This temporal graph represents the reviews from users on business entities. Before timestamp {int(interaction_time)}, the business entities which were recently reviewed by {src_id_text} are represented by the following sequence of graph tokens:"}]
    #     elif self.use_dataset == 'Amazon_movies':
    #         conversations= [{"from": "human", "value": f"This temporal graph represents the reviews from users on the moives. Before timestamp {int(interaction_time)}, The moives which were recently reviewed by {src_id_text} are represented by the following sequence of graph tokens:"}]
    #     else:
    #         pass

    #     graph_token_id_1 += [src_id]

    #     # selected_time_interval = src_nodes_neighbor_times[0] < interaction_time
    #     # src_nodes_neighbor_ids_part = src_nodes_neighbor_ids[0][selected_time_interval]         

    #     # if src_nodes_neighbor_ids_part.shape[0] > 0:
    #     #     src_neighbors_part_counter = Counter(src_nodes_neighbor_ids_part) 
    #     #     most_common_neighbors = src_neighbors_part_counter.most_common(self.sample_neighbor_size)
           
    #     #     # for (neighbor_id,_) in most_common_neighbors:
    #     #     for neighbor_id in set(src_nodes_neighbor_ids_part[-self.sample_neighbor_size:].tolist()):
    #     #         graph_token_id_1 += [neighbor_id]
         
   
    #     if self.use_dataset == 'Enron':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {src_id_text}, while the remaining tokens represent other employees."
    #     elif self.use_dataset == 'GDELT':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {src_id_text}, while the remaining tokens represent other political entities."
    #     elif self.use_dataset == 'ICEWS1819':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {src_id_text}, while the remaining tokens represent other political entities."
    #     elif self.use_dataset == 'Googlemap_CT':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {src_id_text}, while the remaining tokens represent business entities."
    #     elif self.use_dataset == 'Amazon_movies':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {src_id_text}, while the remaining tokens represent moives."
    #     else:
    #         pass



    #     if self.use_dataset == 'Enron':
    #         conversations[0]['value'] += f" Those who recently communicated with {tgt_id_text} are represented by the following sequence of graph tokens:"
    #     elif self.use_dataset == 'GDELT':
    #         conversations[0]['value'] += f" The political entities who recently engaged in political relationship with {tgt_id_text} are represented by the following sequence of graph tokens:"
    #     elif self.use_dataset == 'ICEWS1819':
    #         conversations[0]['value'] += f" The political entities who recently engaged in political relationship with {tgt_id_text} are represented by the following sequence of graph tokens:"
    #     elif self.use_dataset == 'Googlemap_CT':
    #         conversations[0]['value'] += f" The users who recently commented on {tgt_id_text} are represented by the following sequence of graph tokens:"
    #     elif self.use_dataset == 'Amazon_movies':
    #         conversations[0]['value'] += f" The users who recently commented on {tgt_id_text} are represented by the following sequence of graph tokens:"
    #     else:
    #         pass
       
    #     graph_token_id_2 += [tgt_id]


    #     # selected_time_interval = tgt_nodes_neighbor_times[0] < interaction_time
    #     # tgt_nodes_neighbor_ids_part = tgt_nodes_neighbor_ids[0][selected_time_interval]  

    #     # if tgt_nodes_neighbor_ids_part.shape[0] > 0:
    #     #     tgt_neighbors_part_counter = Counter(tgt_nodes_neighbor_ids_part) 
    #     #     most_common_neighbors = tgt_neighbors_part_counter.most_common(self.sample_neighbor_size)
            
    #     #     # for (neighbor_id,_) in most_common_neighbors:
    #     #     for neighbor_id in set(tgt_nodes_neighbor_ids_part[-self.sample_neighbor_size:].tolist()):
    #     #         graph_token_id_2 += [neighbor_id]

    #     # conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*len(graph_token_id_2)}\nThe first token corresponds to {tgt_id_text}, while the remaining tokens represent other employees."
        
    #     if self.use_dataset == 'Enron':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {tgt_id_text}, while the remaining tokens represent other employees."
    #     elif self.use_dataset == 'GDELT':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {tgt_id_text}, while the remaining tokens represent other political entities."
    #     elif self.use_dataset == 'ICEWS1819':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {tgt_id_text}, while the remaining tokens represent other political entities."
    #     elif self.use_dataset == 'Googlemap_CT':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {tgt_id_text}, while the remaining tokens represent users."
    #     elif self.use_dataset == 'Amazon_movies':
    #         conversations[0]['value'] += f"\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\nThe first token corresponds to {tgt_id_text}, while the remaining tokens represent users."
    #     else:
    #         pass



    #     if self.use_dataset == 'Enron':
    #         conversations[0]['value'] += f" Based on the information, determine whether there is an email communication between {src_id_text} and {tgt_id_text} at timestamp {int(interaction_time)}."
    #     elif self.use_dataset == 'GDELT':
    #         conversations[0]['value'] += f" Based on the information, determine whether there is a political relationship between {src_id_text} and {tgt_id_text} at timestamp {int(interaction_time)}."
    #     elif self.use_dataset == 'ICEWS1819':
    #         conversations[0]['value'] += f" Based on the information, determine whether there is a political relationship between {src_id_text} and {tgt_id_text} at timestamp {int(interaction_time)}."
    #     elif self.use_dataset == 'Googlemap_CT':
    #         conversations[0]['value'] += f" Based on the information, determine whether there is a comment between {src_id_text} and {tgt_id_text} at timestamp {int(interaction_time)}."
    #     elif self.use_dataset == 'Amazon_movies':
    #         conversations[0]['value'] += f" Based on the information, determine whether there is a comment between {src_id_text} and {tgt_id_text} at timestamp {int(interaction_time)}."
    #     else:
    #         pass



    #     conversations.append({"from": "gpt", "value": f""})
    #     if positive_flag: 
    #         conversations[1]['value'] += f'Yes, there is.'
    #     else:
    #         conversations[1]['value'] += f'No, there is not.'


    #     sources = {}
    #     sources['conversations'] = conversations
    #     sources['graph_token_id_1'] = graph_token_id_1
    #     sources['graph_token_id_2'] = graph_token_id_2
    #     sources['graph_timestamp'] = graph_timestamp

    #     return sources


    def get_source(self, i):    
        src_id = self.train_data.src_node_ids[self.interaction_index[i]]
        src_id_text = self.train_data.entity_texts[src_id]
        src_id_text = self.process_text(src_id,src_id_text)


        tgt_id = self.train_data.dst_node_ids[self.interaction_index[i]]
        tgt_id_text = self.train_data.entity_texts[tgt_id]
        tgt_id_text = self.process_text(tgt_id,tgt_id_text)


        interaction_time = self.train_data.node_interact_times[self.interaction_index[i]]
        final_interaction_time = self.train_data.node_interact_times[-1]
        src_nodes_neighbor_ids, _, src_nodes_neighbor_times = self.neighbor_sampler.get_all_first_hop_neighbors([src_id],[final_interaction_time+1])
        # tgt_nodes_neighbor_ids, _, tgt_nodes_neighbor_times = self.neighbor_sampler.get_all_first_hop_neighbors([tgt_id],[final_interaction_time+1])


        positive_flag = np.random.rand() >= self.threshold
        if not positive_flag:
            if self.negative_sample == 'temporal':
                mask = src_nodes_neighbor_ids[0] == tgt_id 
                interaction_time = np.setdiff1d(self.train_data.node_interact_times,src_nodes_neighbor_times[0][mask])
            
                
                logging.info(f'len {src_nodes_neighbor_times[0][mask].shape}')
                logging.info(f'len total {self.train_data.node_interact_times.shape}')
                logging.info(f'random {interaction_time.shape}')
                interaction_time = np.random.choice(interaction_time)
            elif self.negative_sample == 'entity':
                mask = src_nodes_neighbor_times[0] == interaction_time
                tgt_id = set(self.train_data.dst_node_ids.tolist()) - set([src_id]) - set(src_nodes_neighbor_ids[0][mask].tolist())
                tgt_id = np.random.choice(np.array(list(tgt_id)))
                tgt_id_text = self.train_data.entity_texts[tgt_id]
                tgt_id_text = self.process_text(tgt_id,tgt_id_text)  
                # tgt_nodes_neighbor_ids, _, tgt_nodes_neighbor_times = self.neighbor_sampler.get_all_first_hop_neighbors([tgt_id],[final_interaction_time+1])
            else:
                raise ValueError('unknown negative sample')

        graph_token_id_1 = []
        graph_token_id_2 = []
        graph_timestamp = []

        graph_timestamp += [interaction_time]

        if self.use_dataset == 'Enron':
            conversations = [{"from": "human", "value": f"This temporal graph captures email communications among employees. Each sequence of graph tokens begins with a specific employee, followed by tokens representing others they recently communicated with before timestamp {int(interaction_time)}."}]
        elif self.use_dataset == 'GDELT':
            conversations= [{"from": "human", "value": f"This temporal graph models political relationships between entities. Each sequence of graph tokens begins with a target entity, followed by entities they recently interacted with before timestamp {int(interaction_time)}."}]
        elif self.use_dataset == 'ICEWS1819':
            conversations= [{"from": "human", "value": f"This temporal graph models political relationships between entities. Each sequence of graph tokens begins with a target entity, followed by entities they recently interacted with before timestamp {int(interaction_time)}."}]
        elif self.use_dataset == 'Googlemap_CT':
            conversations= [{"from": "human", "value": f"This temporal graph captures user reviews on businesses. Each token sequence starts with an entity, followed by those they recently interacted with before timestamp {int(interaction_time)}."}]
        elif self.use_dataset == 'Amazon_movies':
            conversations= [{"from": "human", "value": f"This temporal graph represents user reviews on movies. Each sequence begins with an entity, followed by recent interactions before timestamp {int(interaction_time)}."}]
        else:
            pass


   
        if self.use_dataset == 'Enron':
            conversations[0]['value'] += f"\nGiven the sequences for {src_id_text} and {tgt_id_text}, determine whether there is an email communication between them at timestamp {int(interaction_time)}."
        elif self.use_dataset == 'GDELT':
            conversations[0]['value'] += f"\nGiven the sequences for {src_id_text} and {tgt_id_text}, determine whether a political relationship exists between them at timestamp {int(interaction_time)}."
        elif self.use_dataset == 'ICEWS1819':
            conversations[0]['value'] += f"\nGiven the sequences for {src_id_text} and {tgt_id_text}, determine whether a political relationship exists between them at timestamp {int(interaction_time)}."
        elif self.use_dataset == 'Googlemap_CT':
            conversations[0]['value'] += f"\nGiven the sequences for {src_id_text} (user) and {tgt_id_text} (business), determine whether {src_id_text} commented on {tgt_id_text} at timestamp {int(interaction_time)}."
        elif self.use_dataset == 'Amazon_movies':
            conversations[0]['value'] += f"\nGiven the sequences for {src_id_text} (user) and {tgt_id_text} (movie), determine whether {src_id_text} reviewed {tgt_id_text} at timestamp {int(interaction_time)}."
        else:
            pass


        conversations[0]['value'] += f"\n\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\n{DEFAULT_GRAPH_TOKEN*self.sample_neighbor_size}\n\n"


        graph_token_id_1 += [src_id]
        graph_token_id_2 += [tgt_id]

        conversations.append({"from": "gpt", "value": f""})
        if positive_flag: 
            conversations[1]['value'] += f'Yes.'
        else:
            conversations[1]['value'] += f'No.'


        sources = {}
        sources['conversations'] = conversations
        sources['graph_token_id_1'] = graph_token_id_1
        sources['graph_token_id_2'] = graph_token_id_2
        sources['graph_timestamp'] = graph_timestamp

        return sources


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.get_source(i)
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        graph_token_id_1 = sources[0]['graph_token_id_1']
        graph_token_id_2 = sources[0]['graph_token_id_2']
        graph_timestamp = sources[0]['graph_timestamp']

        # cur_token_len_1 = len(graph_token_id_1)
        # cur_token_len_2 = len(graph_token_id_2)

        # sources = preprocess_graph_LP(
        #     copy.deepcopy([e["conversations"] for e in sources]),
        #     self.graph_cfg, cur_token_len_1, cur_token_len_2)

        sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
      
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             graph_token_id_1=graph_token_id_1,
                             graph_token_id_2=graph_token_id_2,
                             graph_timestamp=graph_timestamp)

        return data_dict
    
class LazySupervisedDataset_back(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 graph_cfg: dict, 
                 **kwargs,):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.graph_cfg = graph_cfg
        graph_data_path = kwargs.get('graph_data_path')
        self.graph_data_all = torch.load(graph_data_path)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'graph' in sources[0]:
            graph_dict = self.list_data_dict[i]['graph']
            graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
            graph_node_list = copy.deepcopy(graph_dict['node_list'])
            target_node = copy.deepcopy(graph_dict['node_idx'])
            graph_type = copy.deepcopy(self.list_data_dict[i]['id']).split('_')[0]
            graph_node_rep = self.graph_data_all[graph_type].x[graph_node_list] ## 
            
            cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size
            sources = preprocess_graph(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.graph_cfg, cur_token_len)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'graph' in self.list_data_dict[i]:
            # data_dict['graph_node'] = graph_node_rep
            # data_dict['graph_edge'] = graph_edge_index
            # data_dict['target_node'] = target_node
            data_dict['graph_data'] = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))

        elif self.graph_cfg['is_graph']:
            # image does not exist in the data, but the model is multimodal
            node_feas = self.graph_cfg['graph_processor'].node_feas
            data_dict['graph_data'] = Data(graph_node = torch.zeros(3, node_feas), edge_index=torch.zeros(2, 3), target_node = torch.tensor([0]))
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


        graph_token_id_1 = [instance['graph_token_id_1'] for instance in instances]
        graph_token_id_2 = [instance['graph_token_id_2'] for instance in instances]
        graph_timestamp = [instance['graph_timestamp'] for instance in instances]
        batch['graph_token_id_1'] = graph_token_id_1
        batch['graph_token_id_2'] = graph_token_id_2
        batch['graph_timestamp'] = graph_timestamp

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                graph_cfg=dict(
                                    is_graph=data_args.is_graph,
                                    sep_graph_conv_front=data_args.sep_graph_conv_front,
                                    graph_token_len=data_args.graph_token_len,
                                    graph_content=data_args.graph_content,
                                    use_graph_start_end=getattr(data_args, 'use_graph_start_end', False),
                                    sample_neighbor_strategy=data_args.sample_neighbor_strategy,
                                    sample_neighbor_size=data_args.sample_neighbor_size,
                                    time_scaling_factor=data_args.time_scaling_factor,
                                    threshold=data_args.threshold,
                                    use_dataset=data_args.use_dataset,
                                    seed=data_args.seed,
                                    negative_sample=data_args.negative_sample
                                    ), 
                                    graph_data_path = data_args.graph_data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    set_random_seed(training_args.seed)
    data_args.seed = training_args.seed

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}

    ## load 4 8 bit 
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        from peft import prepare_model_for_int8_training
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.graph_tower is not None:
        model = GraphLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            ) ## TODO: add real Graph Llama model 
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

    # model.config.pretrain_graph_model_path = '/project/SDS/research/sds-rise/weili/Project_2/GraphGPT/Arxiv-PubMed-GraphCLIP-GT'
    # model.config.pretrain_graph_model_path = model.config.pretrain_graph_model_path + model_args.graph_tower
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_int8_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing and model_args.graph_tower is None:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logging.warning("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        # tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id = 128010
        # conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]


    raw_bert_entities_embedding = np.load(f'{data_args.graph_data_path}/{data_args.use_dataset}_e_feat.npy')
    raw_bert_edges_embedding = np.load(f'{data_args.graph_data_path}/{data_args.use_dataset}_r_feat.npy')

    if model_args.graph_tower is not None:
        model_graph_dict = model.get_model().initialize_graph_modules(
            graph_tower=model_args.graph_tower,
            graph_select_layer=model_args.graph_select_layer,
            entities_embedding=raw_bert_entities_embedding,
            edges_embedding=raw_bert_edges_embedding,
            pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter,
            fsdp=training_args.fsdp
        )
        model.get_graph_tower().to(dtype=compute_dtype, device=training_args.device)
        model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)

        # model.get_graph_tower().set_num_neighbors(data_args.sample_neighbor_size)
        model.get_graph_tower().set_max_input_sequence_length(data_args.sample_neighbor_size)
        # graph_config = model_graph_dict['graph_config']

        # data_args.graph_token_len = model_graph_dict['graph_token_len']
        # data_args.graph_processor = model_graph_dict['graph_processor']
        data_args.is_graph = True

        model.config.tune_graph_mlp_adapter = training_args.tune_graph_mlp_adapter = model_args.tune_graph_mlp_adapter
        if model_args.tune_graph_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = True
            for p in model.get_graph_tower().parameters():
                p.requires_grad = True

        model.config.freeze_graph_mlp_adapter = training_args.freeze_graph_mlp_adapter
        if training_args.freeze_graph_mlp_adapter:
            for p in model.get_model().graph_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().graph_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.use_graph_start_end = data_args.use_graph_start_end = model_args.use_graph_start_end
        # graph_config.use_graph_start_end = training_args.use_graph_start_end = model_args.use_graph_start_end
        training_args.use_graph_start_end = model_args.use_graph_start_end
        model.config.sep_graph_conv_front = data_args.sep_graph_conv_front
        # model.initialize_graph_tokenizer(use_graph_start_end=model_args.use_graph_start_end, tokenizer=tokenizer, device=training_args.device,
        #                                   tune_graph_mlp_adapter=model_args.tune_graph_mlp_adapter, pretrain_graph_mlp_adapter=model_args.pretrain_graph_mlp_adapter)

        params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
                else:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)
                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    tuned_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    model.get_graph_tower().set_neighbor_sampler(data_module['train_dataset'].neighbor_sampler)

    trainer = GraphChatTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    
    print('************************** parameters: #', sum(p.numel() for p in model.parameters() if p.requires_grad))
    tuned_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            tuned_params.append(name)
    print(tuned_params)

    s_time = time.time()

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    logging.info(f'elapsed time {time.time()-s_time}')

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
