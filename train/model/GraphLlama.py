#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from graphgpt.model.graph_layers import MPNN, GNN, CLIP, graph_transformer
from torch_geometric.data import Data
import json
import os.path as osp
import glob

from transformers import modeling_utils

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", "rowwise"]



GRAPH_TOKEN_INDEX = -200
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

import logging
from .model_base import TemporalGraph
import numpy as np



class GraphLlamaConfig(LlamaConfig):
    model_type = "GraphLlama"

class GraphPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path): 
    # load conig json
    
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = GraphPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys(): 
        state_dict.pop('logit_scale')
    print('loading graph pre train model')
    model.load_state_dict(state_dict)


    return model, args
def transfer_param_tograph(clip_graph, gnn):
    
    print(clip_graph)
    gnn_state_dict = clip_graph.gnn.state_dict()
    gnn.load_state_dict(gnn_state_dict)
    return gnn


class GraphLlamaModel(LlamaModel):
    config_class = GraphLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(GraphLlamaModel, self).__init__(config)


    def get_graph_tower(self):
        graph_tower = getattr(self, 'graph_tower', None)
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def initialize_graph_modules(self, graph_tower, graph_select_layer,entities_embedding,edges_embedding,
                                  pretrain_graph_mlp_adapter=None, fsdp=None): # TODO: modify this function
        self.config.graph_tower = graph_tower

        self.config.graph_hidden_size = entities_embedding.shape[-1]
        if not hasattr(self, 'graph_tower'):
            self.graph_tower = TemporalGraph(entities_embedding,edges_embedding)
  
        # graph_tower.requires_grad_(False)

        # if fsdp is not None and len(fsdp) > 0:
        #     self.graph_tower = [graph_tower]
        # else:
        #     self.graph_tower = graph_tower

        

        self.config.use_graph_proj = True
        self.config.graph_select_layer = graph_select_layer

        if not hasattr(self, 'graph_projector'):
            self.graph_projector = nn.Linear(self.config.graph_hidden_size, self.config.hidden_size)

        if pretrain_graph_mlp_adapter is not None:
            graph_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
            self.graph_projector.load_state_dict({k.split('.')[-1]: v for k, v in graph_projector_weights.items()})





    # def forward(
    #     self,
    #     input_ids: torch.LongTensor = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[List[torch.FloatTensor]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     # graph_node_reps: Optional[torch.FloatTensor] = None,
    #     # edge_index_reps: Optional[torch.FloatTensor] = None,
    #     graph_token_id_1: Optional[List[List]] = None,
    #     graph_token_id_2: Optional[List[List]] = None,
    #     graph_timestamp: Optional[List[List]] = None,
    #     return_dict: Optional[bool] = None,
    # ) -> Union[Tuple, BaseModelOutputWithPast]:

    #     graph_tower = self.graph_tower

    #     new_input_embeds = []
    #     # for batch_idx, (cur_input_ids, cur_input_embeds) in enumerate(zip(input_ids, inputs_embeds)):
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         # assert (cur_input_ids == self.config.graph_patch_token).sum() != 0
    #         # assert self.config.use_graph_start_end
    #         # assert (cur_input_ids == self.config.graph_start_token).sum() == (cur_input_ids == self.config.graph_end_token).sum()
               
    #         cur_graph_token_id_1 = graph_token_id_1[batch_idx]
    #         cur_graph_token_id_2 = graph_token_id_2[batch_idx]
    #         cur_graph_timestamp = graph_timestamp[batch_idx]
            

    #         graph_feature_1 = graph_tower.compute_src_dst_node_temporal_embeddings(np.array(cur_graph_token_id_1),np.array(cur_graph_timestamp*len(cur_graph_token_id_1)))
    #         graph_feature_2 = graph_tower.compute_src_dst_node_temporal_embeddings(np.array(cur_graph_token_id_2),np.array(cur_graph_timestamp*len(cur_graph_token_id_2)))
    #         # if batch_idx == 0:
    #         #     print(f'aa {torch.norm(graph_feature_1,dim=-1)}')
    #         #     print(f'aa {torch.norm(graph_feature_2,dim=-1)}')
    #         graph_feature_1 = self.graph_projector(graph_feature_1)
    #         graph_feature_2 = self.graph_projector(graph_feature_2)
    #         # if batch_idx == 0:
    #         #     print(f'bb {torch.norm(graph_feature_1,dim=-1)}')
    #         #     print(f'bb {torch.norm(graph_feature_2,dim=-1)}')
   

    #         # graph_feature = [graph_feature_1,graph_feature_2]

    #         # cur_new_input_embeds = []
    #         # previous_idx = 0
    #         # graph_start_tokens = torch.where(cur_input_ids == self.config.graph_start_token)[0]
    #         # for i,graph_start_token_pos in enumerate(graph_start_tokens):
    #         #     cur_graph_features = graph_feature[i].to(device=cur_input_embeds.device)
    #         #     num_patches = cur_graph_features.shape[0]
    #         #     assert cur_input_ids[graph_start_token_pos + num_patches + 1] == self.config.graph_end_token
    #         #     cur_new_input_embeds.append(cur_input_embeds[previous_idx:graph_start_token_pos+1])
    #         #     cur_new_input_embeds.append(cur_graph_features)
    #         #     previous_idx = graph_start_token_pos + num_patches + 1


    #         graph_features = torch.cat((graph_feature_1,graph_feature_2),dim=0)
    #         graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]

    #         assert graph_features.shape[0] == len(graph_token_indices)

    #         cur_new_input_embeds = []
    #         previous_idx = 0

    #         for i, graph_token_index in enumerate(graph_token_indices):
    #             graph_token_feature = graph_features[i]
               
    #             if graph_token_index > previous_idx:
    #                 cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[previous_idx:graph_token_index]))
    #             cur_new_input_embeds.append(graph_token_feature.unsqueeze(dim=0))
    #             previous_idx = graph_token_index + 1


    #         cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[previous_idx:]))
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
    #         new_input_embeds.append(cur_new_input_embeds)

    #     inputs_embeds = torch.stack(new_input_embeds, dim=0)

    #     return super(GraphLlamaModel, self).forward(
    #         input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds, use_cache=use_cache,
    #         output_attentions=output_attentions, output_hidden_states=output_hidden_states,
    #         return_dict=return_dict
    #     )









    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_token_id_1: Optional[List[List]] = None,
        graph_token_id_2: Optional[List[List]] = None,
        graph_timestamp: Optional[List[List]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        graph_tower = self.graph_tower

        new_input_embeds = []
        # for batch_idx, (cur_input_ids, cur_input_embeds) in enumerate(zip(input_ids, inputs_embeds)):
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # assert (cur_input_ids == self.config.graph_patch_token).sum() != 0
            # assert self.config.use_graph_start_end
            # assert (cur_input_ids == self.config.graph_start_token).sum() == (cur_input_ids == self.config.graph_end_token).sum()
               
            cur_graph_token_id_1 = graph_token_id_1[batch_idx]
            cur_graph_token_id_2 = graph_token_id_2[batch_idx]
            cur_graph_timestamp = graph_timestamp[batch_idx]
            cur_attention_mask = attention_mask[batch_idx]


            graph_feature_1,graph_feature_2,src_padded_ids,tgt_padded_ids = graph_tower.compute_src_dst_node_temporal_embeddings(np.array(cur_graph_token_id_1),np.array(cur_graph_token_id_2),np.array(cur_graph_timestamp*len(cur_graph_token_id_1)))
            graph_features = torch.cat((graph_feature_1.squeeze(),graph_feature_2.squeeze()),dim=0)
            padded_ids = torch.cat((src_padded_ids.squeeze(),tgt_padded_ids.squeeze()),dim=-1)

            if batch_idx == 0:
                print(f'aa {torch.norm(graph_features,dim=-1)}')
            graph_features = self.graph_projector(graph_features)
            if batch_idx == 0:
                print(f'bb {torch.norm(graph_features,dim=-1)}')
   
            graph_features[padded_ids==0] = 0
            graph_token_indices = torch.where(cur_input_ids == GRAPH_TOKEN_INDEX)[0]

            assert graph_features.shape[0] == len(graph_token_indices)

            cur_new_input_embeds = []
            previous_idx = 0

            for i, graph_token_index in enumerate(graph_token_indices):
                graph_token_feature = graph_features[i]

                if padded_ids[i] == 0:
                    cur_attention_mask[graph_token_index] = False
               
                if graph_token_index > previous_idx:
                    cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[previous_idx:graph_token_index]))
                cur_new_input_embeds.append(graph_token_feature.unsqueeze(dim=0))
                previous_idx = graph_token_index + 1


            cur_new_input_embeds.append(self.embed_tokens(cur_input_ids[previous_idx:]))
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)



        inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(GraphLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )





class GraphLlamaForCausalLM(LlamaForCausalLM):
    config_class = GraphLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = GraphLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_graph_tower(self):
        return self.get_model().get_graph_tower()

    def get_vision_tower(self):
        model = self.get_model()
        graph_tower = model.graph_tower
        if type(graph_tower) is list:
            graph_tower = graph_tower[0]
        return graph_tower

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        # graph_node_reps: Optional[torch.FloatTensor] = None,
        # edge_index_reps: Optional[torch.FloatTensor] = None,
        graph_token_id_1: Optional[List[List]] = None,
        graph_token_id_2: Optional[List[List]] = None,
        graph_timestamp: Optional[List[List]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # graph_node_reps=graph_node_reps, 
            # edge_index_reps=edge_index_reps
            graph_token_id_1 = graph_token_id_1,
            graph_token_id_2 = graph_token_id_2,
            graph_timestamp = graph_timestamp
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": [kwargs.get("graph_data", None)],
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        return model_inputs

    def initialize_graph_tokenizer(self, use_graph_start_end, tokenizer, device,
                                    tune_graph_mlp_adapter=False, pretrain_graph_mlp_adapter=None):
        vision_config = self.get_graph_tower().config
        vision_config.use_graph_start_end = use_graph_start_end
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.graph_start_token, vision_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_graph_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_graph_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_graph_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]

AutoConfig.register("GraphLlama", GraphLlamaConfig)
AutoModelForCausalLM.register(GraphLlamaConfig, GraphLlamaForCausalLM)
