from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd


class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, 
                edge_ids: np.ndarray, labels: np.ndarray, entity_texts, relation_texts):

        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.entity_texts = entity_texts
        self.relation_texts = relation_texts






def get_temporal_data(data_path,dataset_name, val_ratio, test_ratio, args=None):
    entity_df = pd.read_csv(f'{data_path}/datasets/{dataset_name}/entity_text.csv')
    entity_texts = entity_df.text.values
    entity_texts = list(entity_texts)
    relation_df = pd.read_csv(f'{data_path}/datasets/{dataset_name}/relation_text.csv')
    relation_texts = relation_df.text.values
    relation_texts = list(relation_texts)


    graph_df = pd.read_csv(f'{data_path}/datasets/{dataset_name}/edge_list.csv')
    node_num = max(graph_df['u'].max(), graph_df['i'].max()) + 1
    rel_num = graph_df['r'].max() + 1

    if graph_df['label'].min() != 0:
        graph_df.label -= 1
    if 'GDELT' in dataset_name:
        graph_df.ts = graph_df.ts//15
    cat_num = graph_df['label'].max() + 1

    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.r.values.astype(np.longlong)
    labels = graph_df.label.values

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, 
                    edge_ids=edge_ids, labels=labels, entity_texts=entity_texts, relation_texts=relation_texts)


    # the setting of seed follows previous works
    random.seed(2020)

    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)


    test_node_set = set(src_node_ids[node_interact_times > val_time]).union(set(dst_node_ids[node_interact_times > val_time]))
    new_test_node_set = set(random.sample(list(test_node_set), int(0.1 * num_total_unique_node_ids)))

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

    train_mask = np.logical_and(node_interact_times <= val_time, observed_edges_mask)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask],entity_texts=entity_texts, relation_texts=relation_texts)


    train_node_set = set(train_data.src_node_ids).union(train_data.dst_node_ids)
    assert len(train_node_set & new_test_node_set) == 0
    new_node_set = node_set - train_node_set

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    edge_contains_new_node_mask = np.array([(src_node_id in new_node_set or dst_node_id in new_node_set)
                                            for src_node_id, dst_node_id in zip(src_node_ids, dst_node_ids)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)


    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask],
                    entity_texts=entity_texts, relation_texts=relation_texts)

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask],
                     entity_texts=entity_texts, relation_texts=relation_texts)

    new_node_val_data = Data(src_node_ids=src_node_ids[new_node_val_mask], dst_node_ids=dst_node_ids[new_node_val_mask],
                             node_interact_times=node_interact_times[new_node_val_mask],
                             edge_ids=edge_ids[new_node_val_mask], labels=labels[new_node_val_mask],
                             entity_texts=entity_texts, relation_texts=relation_texts)

    new_node_test_data = Data(src_node_ids=src_node_ids[new_node_test_mask], dst_node_ids=dst_node_ids[new_node_test_mask],
                              node_interact_times=node_interact_times[new_node_test_mask],
                              edge_ids=edge_ids[new_node_test_mask], labels=labels[new_node_test_mask],
                              entity_texts=entity_texts, relation_texts=relation_texts)

    return full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, cat_num


if __name__ == '__main__':
    get_temporal_data('Enron', val_ratio=0.1, test_ratio=0.1, args=None)
