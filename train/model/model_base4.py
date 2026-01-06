import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('/project/SDS/research/sds-rise/weili/Project_2/LLMTemporal2')

from graphgpt.utils.util_neighbor import NeighborSampler


class TimeEncoder(nn.Module):

    def __init__(self, time_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(TimeEncoder, self).__init__()

        self.time_dim = time_dim
        # trainable parameters for time encoding
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)

        timestamps = timestamps.unsqueeze(dim=2)
        timestamps = timestamps.to(self.w.weight.device)
        timestamps = timestamps.to(self.w.weight.dtype)
   
        # Tensor, shape (batch_size, seq_len, time_dim)
        output = torch.cos(self.w(timestamps))

        return output
    



class TemporalGraph(nn.Module):

    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler = None,query_type:str = 'None',
                 time_feat_dim: int = 100, num_layers: int = 2, num_heads: int = 2, dropout: float = 0.1):
        """
        TCL model.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param num_layers: int, number of transformer layers
        :param num_heads: int, number of attention heads
        :param num_depths: int, number of depths, identical to the number of sampled neighbors plus 1 (involving the target node)
        :param dropout: float, dropout rate
        :param device: str, device
        """
        super(TemporalGraph, self).__init__()

        #self.node_raw_features = nn.Parameter(torch.from_numpy(node_raw_features.astype(np.float32)), requires_grad = True).to(device)
        #self.edge_raw_features = nn.Parameter(torch.from_numpy(edge_raw_features.astype(np.float32)), requires_grad = True).to(device)

        self.register_buffer("node_raw_features" ,torch.from_numpy(node_raw_features.astype(np.float32)))
        self.register_buffer("edge_raw_features" ,torch.from_numpy(edge_raw_features.astype(np.float32)))

        self.node_raw_features = F.normalize(self.node_raw_features,dim=-1)
        self.edge_raw_features = F.normalize(self.edge_raw_features,dim=-1)

        self.neighbor_sampler = neighbor_sampler
        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.freq_feat_dim = 100
        self.num_channels = 4
        self.channel_embedding_dim = 50
        self.num_neighbors = 23
        self.hidden_size = self.node_feat_dim

        self.time_encoder = TimeEncoder(time_dim=time_feat_dim)
        self.neighbor_co_occurrence_encoder = NeighborCooccurrenceEncoder(self.num_neighbors,self.freq_feat_dim)

        self.projection_layer = nn.ModuleDict({
            'node': nn.Linear(in_features=self.node_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'edge': nn.Linear(in_features=self.edge_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'time': nn.Linear(in_features=self.time_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            'neighbor_co_occurrence': nn.Linear(in_features=self.freq_feat_dim, out_features=self.channel_embedding_dim, bias=True),
            #  'neighbor_co_occurrence': nn.Identity(),
        })

        self.transformers = nn.ModuleList([
            TransformerEncoder(attention_dim=self.channel_embedding_dim*self.num_channels, num_heads=self.num_heads, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])


        self.output_layer = nn.Linear(in_features=self.channel_embedding_dim*self.num_channels, out_features=self.node_feat_dim, bias=True)


    def set_neighborhood_sampler(self,neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    
    def set_num_neighbors(self,num_neighbors:int):
        self.num_neighbors = num_neighbors-1
  

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, num_neighbors: int = 35):
        """
        compute source and destination node temporal embeddings
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        # get temporal neighbors of source nodes, including neighbor ids, edge ids and time information
        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=src_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.num_neighbors)

        # get temporal neighbors of destination nodes, including neighbor ids, edge ids and time information
        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors)
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=dst_node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.num_neighbors)

        # src_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_node_ids = np.concatenate((src_node_ids[:, np.newaxis], src_neighbor_node_ids), axis=1)
        # src_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_edge_ids = np.concatenate((np.zeros((len(src_node_ids), 1)).astype(np.longlong), src_neighbor_edge_ids), axis=1)
        # src_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        src_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], src_neighbor_times), axis=1)

        # dst_neighbor_node_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_node_ids = np.concatenate((dst_node_ids[:, np.newaxis], dst_neighbor_node_ids), axis=1)
        # dst_neighbor_edge_ids, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_edge_ids = np.concatenate((np.zeros((len(dst_node_ids), 1)).astype(np.longlong), dst_neighbor_edge_ids), axis=1)
        # dst_neighbor_times, ndarray, shape (batch_size, num_neighbors + 1)
        dst_neighbor_times = np.concatenate((node_interact_times[:, np.newaxis], dst_neighbor_times), axis=1)


        src_neighbor_co_occurrence_raw_features, dst_neighbor_co_occurrence_raw_features = \
            self.neighbor_co_occurrence_encoder(src_padded_nodes_neighbor_ids=src_neighbor_node_ids,
                                                dst_padded_nodes_neighbor_ids=dst_neighbor_node_ids)

        # pad the features of the sequence of source and destination nodes
        # src_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # src_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # src_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # src_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=src_neighbor_node_ids,
                              nodes_edge_ids=src_neighbor_edge_ids, nodes_neighbor_times=src_neighbor_times, time_encoder=self.time_encoder)

        # dst_nodes_neighbor_node_raw_features, Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        # dst_nodes_edge_raw_features, Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        # dst_nodes_neighbor_time_features, Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        # dst_nodes_neighbor_depth_features, Tensor, shape (num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features = \
            self.get_features(node_interact_times=node_interact_times, nodes_neighbor_ids=dst_neighbor_node_ids,
                              nodes_edge_ids=dst_neighbor_edge_ids, nodes_neighbor_times=dst_neighbor_times, time_encoder=self.time_encoder)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        src_nodes_neighbor_node_raw_features = self.projection_layer['node'](src_nodes_neighbor_node_raw_features)
        src_nodes_edge_raw_features = self.projection_layer['edge'](src_nodes_edge_raw_features)
        src_nodes_neighbor_time_features = self.projection_layer['time'](src_nodes_neighbor_time_features)
        src_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](src_neighbor_co_occurrence_raw_features)

        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        dst_nodes_neighbor_node_raw_features = self.projection_layer['node'](dst_nodes_neighbor_node_raw_features)
        dst_nodes_edge_raw_features = self.projection_layer['edge'](dst_nodes_edge_raw_features)
        dst_nodes_neighbor_time_features = self.projection_layer['time'](dst_nodes_neighbor_time_features)
        dst_neighbor_co_occurrence_features = self.projection_layer['neighbor_co_occurrence'](dst_neighbor_co_occurrence_raw_features)


        src_node_features = torch.cat((src_nodes_neighbor_node_raw_features, src_nodes_edge_raw_features, src_nodes_neighbor_time_features, src_neighbor_co_occurrence_features), dim=-1)
        dst_node_features = torch.cat((dst_nodes_neighbor_node_raw_features, dst_nodes_edge_raw_features, dst_nodes_neighbor_time_features, dst_neighbor_co_occurrence_features), dim=-1)

        for i,transformer in enumerate(self.transformers):
            # self-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_features = transformer(inputs_query=src_node_features, inputs_key=src_node_features,
                                            inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_features = transformer(inputs_query=dst_node_features, inputs_key=dst_node_features,
                                            inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            
            # cross-attention block
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            src_node_embeddings = transformer(inputs_query=src_node_features, inputs_key=dst_node_features,
                                            inputs_value=dst_node_features, neighbor_masks=dst_neighbor_node_ids)
            # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
            dst_node_embeddings = transformer(inputs_query=dst_node_features, inputs_key=src_node_features,
                                            inputs_value=src_node_features, neighbor_masks=src_neighbor_node_ids)

            src_node_features, dst_node_features = src_node_embeddings, dst_node_embeddings



        # retrieve the embedding of the corresponding target node, which is at the first position of the sequence

        # src_node_embeddings = self.output_layer(src_node_features.mean(dim=1))
        # # Tensor, shape (batch_size, node_feat_dim)
        # dst_node_embeddings = self.output_layer(dst_node_features.mean(dim=1))

        # # Tensor, shape (batch_size, node_feat_dim)
        # src_node_embeddings = self.output_layer(src_node_features[:, 0, :])
        # # Tensor, shape (batch_size, node_feat_dim)
        # dst_node_embeddings = self.output_layer(dst_node_features[:, 0, :])

        
        # Tensor, shape (batch_size, node_feat_dim)
        src_node_embeddings = self.output_layer(src_node_features)
        # Tensor, shape (batch_size, node_feat_dim)
        dst_node_embeddings = self.output_layer(dst_node_features)


        return src_node_embeddings, dst_node_embeddings,torch.from_numpy(src_neighbor_node_ids), torch.from_numpy(dst_neighbor_node_ids)

    def get_features(self, node_interact_times: np.ndarray, nodes_neighbor_ids: np.ndarray, nodes_edge_ids: np.ndarray,
                     nodes_neighbor_times: np.ndarray, time_encoder: TimeEncoder):
        """
        get node, edge, time and depth features
        :param node_interact_times: ndarray, shape (batch_size, )
        :param nodes_neighbor_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_edge_ids: ndarray, shape (batch_size, num_neighbors + 1)
        :param nodes_neighbor_times: ndarray, shape (batch_size, num_neighbors + 1)
        :param time_encoder: TimeEncoder, time encoder
        :return:
        """
        # Tensor, shape (batch_size, num_neighbors + 1, node_feat_dim)
        nodes_neighbor_node_raw_features = self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, edge_feat_dim)
        nodes_edge_raw_features = self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
        # Tensor, shape (batch_size, num_neighbors + 1, time_feat_dim)
        nodes_neighbor_time_features = time_encoder(timestamps=torch.from_numpy(node_interact_times[:, np.newaxis] - nodes_neighbor_times).float())
       
        nodes_neighbor_time_features[torch.from_numpy(nodes_neighbor_ids == 0)] = 0.0

        return nodes_neighbor_node_raw_features, nodes_edge_raw_features, nodes_neighbor_time_features

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        self.neighbor_sampler = neighbor_sampler
        if self.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
            assert self.neighbor_sampler.seed is not None
            self.neighbor_sampler.reset_random_state()



class NeighborCooccurrenceEncoder(nn.Module):

    def __init__(self, num_neighbors:int, neighbor_co_occurrence_feat_dim: int):
        """
        Neighbor co-occurrence encoder.
        :param neighbor_co_occurrence_feat_dim: int, dimension of neighbor co-occurrence features (encodings)
        :param device: str, device
        """
        super(NeighborCooccurrenceEncoder, self).__init__()
        self.neighbor_co_occurrence_feat_dim = neighbor_co_occurrence_feat_dim
        self.num_neighbors = num_neighbors


        self.neighbor_co_occurrence_encode_layer = FrequencyEncoder(fre_dim=self.neighbor_co_occurrence_feat_dim,parameter_requires_grad=False)

    def count_nodes_appearances(self, src_padded_nodes_neighbor_ids_all: np.ndarray, dst_padded_nodes_neighbor_ids_all: np.ndarray):
        """
        count the appearances of nodes in the sequences of source and destination nodes
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # two lists to store the appearances of source and destination nodes
        src_padded_nodes_appearances, dst_padded_nodes_appearances = [], []
        # src_padded_node_neighbor_ids, ndarray, shape (src_max_seq_length, )
        # dst_padded_node_neighbor_ids, ndarray, shape (dst_max_seq_length, )
        for src_padded_node_neighbor_ids, dst_padded_node_neighbor_ids in zip(src_padded_nodes_neighbor_ids_all, dst_padded_nodes_neighbor_ids_all):

            # src_unique_keys, ndarray, shape (num_src_unique_keys, )
            # src_inverse_indices, ndarray, shape (src_max_seq_length, )
            # src_counts, ndarray, shape (num_src_unique_keys, )
            # we can use src_unique_keys[src_inverse_indices] to reconstruct the original input, and use src_counts[src_inverse_indices] to get counts of the original input
            src_unique_keys, src_inverse_indices, src_counts = np.unique(src_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_src = torch.from_numpy(src_counts[src_inverse_indices]).float()
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the source node
            src_mapping_dict = dict(zip(src_unique_keys, src_counts))

            # dst_unique_keys, ndarray, shape (num_dst_unique_keys, )
            # dst_inverse_indices, ndarray, shape (dst_max_seq_length, )
            # dst_counts, ndarray, shape (num_dst_unique_keys, )
            # we can use dst_unique_keys[dst_inverse_indices] to reconstruct the original input, and use dst_counts[dst_inverse_indices] to get counts of the original input
            dst_unique_keys, dst_inverse_indices, dst_counts = np.unique(dst_padded_node_neighbor_ids, return_inverse=True, return_counts=True)
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_dst = torch.from_numpy(dst_counts[dst_inverse_indices]).float()
            # dictionary, store the mapping relation from unique neighbor id to its appearances for the destination node
            dst_mapping_dict = dict(zip(dst_unique_keys, dst_counts))

            # we need to use copy() to avoid the modification of src_padded_node_neighbor_ids
            # Tensor, shape (src_max_seq_length, )
            src_padded_node_neighbor_counts_in_dst = torch.from_numpy(src_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: dst_mapping_dict.get(neighbor_id, 0.0)).float()
            # Tensor, shape (src_max_seq_length, 2)
            src_padded_nodes_appearances.append(torch.stack([src_padded_node_neighbor_counts_in_src, src_padded_node_neighbor_counts_in_dst], dim=1))

            # we need to use copy() to avoid the modification of dst_padded_node_neighbor_ids
            # Tensor, shape (dst_max_seq_length, )
            dst_padded_node_neighbor_counts_in_src = torch.from_numpy(dst_padded_node_neighbor_ids.copy()).apply_(lambda neighbor_id: src_mapping_dict.get(neighbor_id, 0.0)).float()
            # Tensor, shape (dst_max_seq_length, 2)
            dst_padded_nodes_appearances.append(torch.stack([dst_padded_node_neighbor_counts_in_src, dst_padded_node_neighbor_counts_in_dst], dim=1))

        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances = torch.stack(src_padded_nodes_appearances, dim=0)
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances = torch.stack(dst_padded_nodes_appearances, dim=0)

        # set the appearances of the padded node (with zero index) to zeros
        # Tensor, shape (batch_size, src_max_seq_length, 2)
        src_padded_nodes_appearances[torch.from_numpy(src_padded_nodes_neighbor_ids_all == 0)] = 0.0
        # Tensor, shape (batch_size, dst_max_seq_length, 2)
        dst_padded_nodes_appearances[torch.from_numpy(dst_padded_nodes_neighbor_ids_all == 0)] = 0.0

        return src_padded_nodes_appearances, dst_padded_nodes_appearances

   

    def forward(self, src_padded_nodes_neighbor_ids: np.ndarray, dst_padded_nodes_neighbor_ids: np.ndarray):
        """
        compute the neighbor co-occurrence features of nodes in src_padded_nodes_neighbor_ids and dst_padded_nodes_neighbor_ids
        :param src_padded_nodes_neighbor_ids: ndarray, shape (batch_size, src_max_seq_length)
        :param dst_padded_nodes_neighbor_ids:: ndarray, shape (batch_size, dst_max_seq_length)
        :return:
        """
        # src_padded_nodes_appearances, Tensor, shape (batch_size, src_max_seq_length, 2)
        # dst_padded_nodes_appearances, Tensor, shape (batch_size, dst_max_seq_length, 2)
        src_padded_nodes_appearances, dst_padded_nodes_appearances = self.count_nodes_appearances(src_padded_nodes_neighbor_ids_all=src_padded_nodes_neighbor_ids,
                                                                                                  dst_padded_nodes_neighbor_ids_all=dst_padded_nodes_neighbor_ids)


        src_padded_nodes_appearances[:,0,:] = 0
        dst_padded_nodes_appearances[:,0,:] = 0

        # src_padded_nodes_appearances_total = src_padded_nodes_appearances.sum(dim=-1)
        # dst_padded_nodes_appearances_total = dst_padded_nodes_appearances.sum(dim=-1)
        # src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances_total.float()/(src_padded_nodes_appearances_total.sum(dim=-1,keepdim=True)))
        # dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances_total.float()/(dst_padded_nodes_appearances_total.sum(dim=-1,keepdim=True)))
       

        # src_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,0].float()/(src_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True)))
        # dst_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,1].float()/(dst_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True)))
        # src_padded_nodes_neighbor_co_occurrence_features = src_padded_nodes_neighbor_co_occurrence_features_1
        # dst_padded_nodes_neighbor_co_occurrence_features = dst_padded_nodes_neighbor_co_occurrence_features_1     


        # src_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,0].float()/(1*self.num_neighbors))
        # src_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,1].float()/(1*self.num_neighbors))
        # dst_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,0].float()/(1*self.num_neighbors))
        # dst_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,1].float()/(1*self.num_neighbors))
        src_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,0].float()/(src_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True)+1e-5))
        src_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,1].float()/(src_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True)+1e-5))
        dst_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,0].float()/(dst_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True)+1e-5))
        dst_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,1].float()/(dst_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True)+1e-5))
        src_padded_nodes_neighbor_co_occurrence_features = src_padded_nodes_neighbor_co_occurrence_features_1 + src_padded_nodes_neighbor_co_occurrence_features_2
        dst_padded_nodes_neighbor_co_occurrence_features = dst_padded_nodes_neighbor_co_occurrence_features_1 + dst_padded_nodes_neighbor_co_occurrence_features_2     


        # src_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,0].float()/(src_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True)+1e-5))
        # src_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,1].float()/(src_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True)+1e-5))
        # dst_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,0].float()/(dst_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True)+1e-5))
        # dst_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,1].float()/(dst_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True)+1e-5))
        # src_padded_nodes_neighbor_co_occurrence_features = torch.cat((src_padded_nodes_neighbor_co_occurrence_features_1,src_padded_nodes_neighbor_co_occurrence_features_2),dim=-1)
        # dst_padded_nodes_neighbor_co_occurrence_features = torch.cat((dst_padded_nodes_neighbor_co_occurrence_features_1,dst_padded_nodes_neighbor_co_occurrence_features_2),dim=-1)  




        # src_padded_nodes_appearances_total = src_padded_nodes_appearances.sum(dim=-1)
        # dst_padded_nodes_appearances_total = dst_padded_nodes_appearances.sum(dim=-1)
        # src_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances_total.float()/(src_padded_nodes_appearances_total.sum(dim=-1,keepdim=True)))
        # dst_padded_nodes_neighbor_co_occurrence_features_1 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances_total.float()/(dst_padded_nodes_appearances_total.sum(dim=-1,keepdim=True)))
        # src_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances[:,:,0].float()/(src_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True)))
        # dst_padded_nodes_neighbor_co_occurrence_features_2 = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances[:,:,1].float()/(dst_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True)))
        # src_padded_nodes_neighbor_co_occurrence_features = src_padded_nodes_neighbor_co_occurrence_features_1 + src_padded_nodes_neighbor_co_occurrence_features_2
        # dst_padded_nodes_neighbor_co_occurrence_features = dst_padded_nodes_neighbor_co_occurrence_features_1 + dst_padded_nodes_neighbor_co_occurrence_features_2


        # src_padded_nodes_appearances_total = src_padded_nodes_appearances.sum(dim=-1)
        # dst_padded_nodes_appearances_total = dst_padded_nodes_appearances.sum(dim=-1)
        # src_padded_nodes_appearances_1 = src_padded_nodes_appearances_total.float()/(src_padded_nodes_appearances_total.sum(dim=-1,keepdim=True))
        # dst_padded_nodes_appearances_1 = dst_padded_nodes_appearances_total.float()/(dst_padded_nodes_appearances_total.sum(dim=-1,keepdim=True))
        # src_padded_nodes_appearances_2 = src_padded_nodes_appearances[:,:,0].float()/(src_padded_nodes_appearances[:,:,0].sum(dim=-1,keepdim=True))
        # dst_padded_nodes_appearances_2 = dst_padded_nodes_appearances[:,:,1].float()/(dst_padded_nodes_appearances[:,:,1].sum(dim=-1,keepdim=True))
        # src_padded_nodes_appearances = torch.stack((src_padded_nodes_appearances_1,src_padded_nodes_appearances_2),dim=-1)
        # dst_padded_nodes_appearances = torch.stack((dst_padded_nodes_appearances_1,dst_padded_nodes_appearances_2),dim=-1)
        # src_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(src_padded_nodes_appearances)
        # dst_padded_nodes_neighbor_co_occurrence_features = self.neighbor_co_occurrence_encode_layer(dst_padded_nodes_appearances)

       # src_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, src_max_seq_length, neighbor_co_occurrence_feat_dim)
        # dst_padded_nodes_neighbor_co_occurrence_features, Tensor, shape (batch_size, dst_max_seq_length, neighbor_co_occurrence_feat_dim)
        return src_padded_nodes_neighbor_co_occurrence_features, dst_padded_nodes_neighbor_co_occurrence_features



class FrequencyEncoder(nn.Module):

    def __init__(self, fre_dim: int, parameter_requires_grad: bool = True):
        """
        Time encoder.
        :param time_dim: int, dimension of time encodings
        :param parameter_requires_grad: boolean, whether the parameter in TimeEncoder needs gradient
        """
        super(FrequencyEncoder, self).__init__()

        self.fre_dim = fre_dim

        # trainable parameters for time encoding


        # self.w = nn.Sequential(
        #     nn.Linear(in_features=1, out_features=self.fre_dim),
        #     nn.ReLU(),
        #     nn.Linear(in_features=self.fre_dim, out_features=self.fre_dim),
        #     )
        
        w2 = torch.tensor(2*math.pi*np.arange(1, fre_dim+1)).unsqueeze(dim=0)
        self.register_buffer('w2',w2)
     
        self.w3 = nn.parameter.Parameter(torch.ones(1,fre_dim))


    def forward(self, frequency: torch.Tensor):
        """
        compute time encodings of time in timestamps
        :param timestamps: Tensor, shape (batch_size, seq_len)
        :return:
        """
        # Tensor, shape (batch_size, seq_len, 1)
        frequency = frequency.unsqueeze(dim=2)
        frequency = frequency.to(self.w3.device)
        frequency = frequency.to(self.w3.dtype)

        frequency = frequency*self.w3.expand(frequency.shape[1],-1)
        output = torch.cos(frequency * self.w2.expand(frequency.shape[1],-1))
        # output = self.w(frequency)


        return output



class TransformerEncoder(nn.Module):

    def __init__(self, attention_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Transformer encoder.
        :param attention_dim: int, dimension of the attention vector
        :param num_heads: int, number of attention heads
        :param dropout: float, dropout rate
        """
        super(TransformerEncoder, self).__init__()
        # use the MultiheadAttention implemented by PyTorch
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=attention_dim, num_heads=num_heads, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.linear_layers = nn.ModuleList([
            nn.Linear(in_features=attention_dim, out_features=4 * attention_dim),
            nn.Linear(in_features=4 * attention_dim, out_features=attention_dim)
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(attention_dim),
            nn.LayerNorm(attention_dim)
        ])

    def forward(self, inputs_query: torch.Tensor, inputs_key: torch.Tensor = None, inputs_value: torch.Tensor = None,
                neighbor_masks: np.ndarray = None):
        """
        encode the inputs by Transformer encoder
        :param inputs_query: Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        :param inputs_key: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param inputs_value: Tensor, shape (batch_size, source_seq_length, self.attention_dim)
        :param neighbor_masks: ndarray, shape (batch_size, source_seq_length), used to create mask of neighbors for nodes in the batch
        :return:
        """
        if inputs_key is None or inputs_value is None:
            assert inputs_key is None and inputs_value is None
            inputs_key = inputs_value = inputs_query
        # note that the MultiheadAttention module accept input data with shape (seq_length, batch_size, input_dim), so we need to transpose the input
        # transposed_inputs_query, Tensor, shape (target_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_key, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        # transposed_inputs_value, Tensor, shape (source_seq_length, batch_size, self.attention_dim)
        transposed_inputs_query, transposed_inputs_key, transposed_inputs_value = inputs_query.transpose(0, 1), inputs_key.transpose(0, 1), inputs_value.transpose(0, 1)

        if neighbor_masks is not None:
            # Tensor, shape (batch_size, source_seq_length)
            neighbor_masks = torch.from_numpy(neighbor_masks).to(inputs_query.device) == 0

        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.multi_head_attention(query=transposed_inputs_query, key=transposed_inputs_key,
                                                  value=transposed_inputs_value, key_padding_mask=neighbor_masks)[0].transpose(0, 1)
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[0](inputs_query + self.dropout(hidden_states))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        hidden_states = self.linear_layers[1](self.dropout(F.relu(self.linear_layers[0](outputs))))
        # Tensor, shape (batch_size, target_seq_length, self.attention_dim)
        outputs = self.norm_layers[1](outputs + self.dropout(hidden_states))

        return outputs
