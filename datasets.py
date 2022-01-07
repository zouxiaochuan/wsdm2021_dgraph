
import torch
import torch.utils.data
import os
import tarfile
import numpy as np
import pickle
from common_structures import PointData, PointDataTest
import label_utils
import feat_utils
import constants
from tqdm import tqdm
import multiprocessing as mp
import point_data_utils
import common_utils
import random


def read_tar_names(param):
    filename = param
    with tarfile.open(filename, 'r') as fin:
        return fin.getnames()
    pass


class DygDataset(torch.utils.data.Dataset):

    def get_point_data_neg(self, eid) -> PointData:
        ts = self.timestamps[eid]
        current_sample = [
            self.src_nodes[eid], self.dst_nodes[eid], self.edge_types[eid], ts]

        while True:
            choose_change_idx = np.random.choice(3)
            choice_nums = [self.num_nodes, self.num_nodes, self.num_edge_types]
            current_sample[choose_change_idx] = np.random.choice(
                choice_nums[choose_change_idx])
        
            point_data = point_data_utils.get_point_data(
                current_sample, self.src_nodes, self.dst_nodes, self.edge_types,
                self.timestamps, self.config, self.triplet_index,
                self.triplet_index_bilateral, self.pair_index, self.node_index,
                self.trip_search_index, self.trip_search_index_map,
                self.pair_search_index, self.pair_search_index_map,
                self.src_node_search_index, self.src_node_search_index_map,
                self.dst_node_search_index, self.dst_node_search_index_map)

            if len(point_data.history_edges_triplet) > 0 or \
               len(point_data.history_edges_pair) > 0 or \
               len(point_data.history_edges_src) > 0 or \
               len(point_data.history_edges_dst) > 0:
                break
            pass
        return point_data

    def get_point_data_pos(self, eid) -> PointData:
        ts = self.timestamps[eid]
        current_sample = [
            self.src_nodes[eid], self.dst_nodes[eid], self.edge_types[eid], ts]
        
        point_data = point_data_utils.get_point_data(
            current_sample, self.src_nodes, self.dst_nodes, self.edge_types,
            self.timestamps, self.config, self.triplet_index,
            self.triplet_index_bilateral, self.pair_index, self.node_index,
            self.trip_search_index, self.trip_search_index_map,
            self.pair_search_index, self.pair_search_index_map,
            self.src_node_search_index, self.src_node_search_index_map,
            self.dst_node_search_index, self.dst_node_search_index_map)
        return point_data

    def get_start_end(self, data_folder):
        filenames = sorted(os.listdir(data_folder))

        with tarfile.open(os.path.join(data_folder, filenames[0]), 'r') as tar:
            first = int(tar.getnames()[0].split('.')[0])
        with tarfile.open(os.path.join(data_folder, filenames[-1]), 'r') as tar:
            last = int(tar.getnames()[-1].split('.')[0])
            pass

        return [first, last]

    def get_all_positive_eids(self,):

        # should not be new node
        past_nodes = set()
        is_new_node = np.zeros(len(self.src_nodes), dtype='bool')
        for eid, (snode, dnode) in enumerate(
                zip(self.src_nodes, self.dst_nodes)):
            if snode not in past_nodes or dnode not in past_nodes:
                is_new_node[eid] = True
                pass
            
            past_nodes.add(snode)
            past_nodes.add(dnode)
            pass

        is_before_train_start = self.timestamps < \
            common_utils.dt2ts(self.config['train_start'])

        eids = np.arange(len(self.src_nodes), dtype='int32')
        eids = eids[np.logical_not(is_new_node | is_before_train_start)]
        return eids
    
    def __init__(self, config, split, valid_percent=0.1, num=1000):
        folder = config['dataset_path']
        self.folder = folder
        self.data_folder = os.path.join(folder, 'data')
        self.config = config

        self.timestamps = np.load(os.path.join(folder, 'timestamps.npy'))
        self.src_nodes = np.load(os.path.join(folder, 'src_nodes.npy'))
        self.dst_nodes = np.load(os.path.join(folder, 'dst_nodes.npy'))
        self.edge_types = np.load(os.path.join(folder, 'edge_types.npy'))
        self.triplet_index = np.load(os.path.join(folder, 'triplet_index.npy'))
        self.triplet_index_bilateral = np.load(
            os.path.join(folder, 'triplet_index_bilateral.npy'))
        self.pair_index = np.load(os.path.join(folder, 'pair_index.npy'))
        self.node_index = np.load(os.path.join(folder, 'node_index.npy'))
        self.trip_search_index = np.load(os.path.join(
            folder, 'trip_search_index.npy'))
        self.trip_search_index_map = np.load(os.path.join(
            folder, 'trip_search_index_map.npy'))
        self.pair_search_index = np.load(os.path.join(
            folder, 'pair_search_index.npy'))
        self.pair_search_index_map = np.load(os.path.join(
            folder, 'pair_search_index_map.npy'))
        self.src_node_search_index = np.load(os.path.join(
            folder, 'src_node_search_index.npy'))
        self.src_node_search_index_map = np.load(os.path.join(
            folder, 'src_node_search_index_map.npy'))
        self.dst_node_search_index = np.load(os.path.join(
            folder, 'dst_node_search_index.npy'))
        self.dst_node_search_index_map = np.load(os.path.join(
            folder, 'dst_node_search_index_map.npy'))
        
        with open(os.path.join(folder, 'edge_type_map.pk'), 'rb') as fin:
            edge_type_map = pickle.load(fin)
            self.num_edge_types = len(edge_type_map)
            pass

        with open(os.path.join(folder, 'node_map.pk'), 'rb') as fin:
            node_map = pickle.load(fin)
            self.num_nodes = len(node_map)
            pass
        
        if 'node_feat_file' in config:
            # dataset a
            self.edge_type_feat = np.load(os.path.join(
                folder, 'edge_type_feat.npy'))
            self.node_feat = np.load(os.path.join(folder, 'node_feat.npy'))
            pass
        else:
            # dataset b
            self.edge_feat = np.load(os.path.join(folder, 'edge_feat.npy'))
            edge_feat_ids = np.load(os.path.join(folder, 'edge_feat_ids.npy'))
            self.edge_feat_idmap = {eid: i for i, eid in enumerate(edge_feat_ids)}
            pass

        self.all_positive_eids = self.get_all_positive_eids()
        
        num_train = len(self.all_positive_eids)
        num_valid_test = int(num_train * valid_percent)
        num_valid_train = num_train - num_valid_test
        
        if split == 'train':
            self.positive_eids = self.all_positive_eids
            pass
        elif split == 'valid_train':
            self.positive_eids = self.all_positive_eids[:num_valid_train]
            pass
        elif split == 'valid_test':
            self.positive_eids = self.all_positive_eids[num_valid_train:]
            pass
        else:
            raise RuntimeError(f'no recognize split: {split}')

        print(len(self.positive_eids))
        self.split = split

        self.num = num
        pass

    def encode_history_edges(self, edges, current_ts):
        edges_ts = self.timestamps[edges[:, 0]]

        edges_feat = feat_utils.time_encoding(current_ts, edges_ts)
        edges_direction = np.zeros(edges_feat.shape[0], dtype='int64')
        edges_direction = edges[:, 1]

        edges_feat = np.concatenate(
            (edges_feat, edges_direction.reshape(-1, 1)), axis=-1)
        
        return edges_feat

    def get_history_edge_feat_b(self, edges):
        feat = np.zeros(
            (len(edges), self.config['extra_feat_dim']),
            dtype='float32')

        if len(edges) > 0:
            edges = edges[:, 0]

            fids = []
            findex = []
            for i, e in enumerate(edges):
                fid = self.edge_feat_idmap.get(e)
                if fid is not None:
                    fids.append(int(fid))
                    findex.append(i)
                    pass
                pass

            fids = np.array(fids, dtype='int64')
            findex = np.array(findex, dtype='int64')

            feat[findex] = self.edge_feat[fids]
            pass
        
        return feat

    def get_edge_feat(self, src_node, dst_node, edge_type):
        if 'node_feat_file' in self.config:
            src_feat = self.node_feat[src_node] + 1 # 1 for -1
            dst_feat = self.node_feat[dst_node] + 1
            edge_type_feat = self.edge_type_feat[edge_type] + 1

            edge_feat = np.hstack((src_feat, dst_feat, edge_type_feat))
            edge_feat = feat_utils.merge_category(
                edge_feat, self.config['edge_feat_dim'])
            pass
        else:
            edge_feat = np.zeros((1, 1), dtype='int64')
            pass
        
        return edge_feat

    def get_pair_feat_extra(self, current_edge_type, pair_edges):
        
        if len(pair_edges) > 0:
            pair_edge_types = self.edge_types[pair_edges[:, 0]]
            is_predict_edge_type = np.array(pair_edge_types == current_edge_type, dtype='int32')
            pair_edge_types_feat = pair_edge_types[:, None]

            pair_feat = np.concatenate(
                (pair_edge_types_feat, is_predict_edge_type[:, None]), axis=-1)
        else:
            pair_feat = np.zeros((0, 2), dtype='int64')
            pass
        
        return pair_feat
    
    def remove_edges_before(self, edges, timestamps, ts):
        if len(edges) == 0 :
            return edges
        else:
            return edges[timestamps[edges[:, 0]] < ts]
        pass

    def get_max_history_ts(self, trip_edges, pair_edges, src_edges, dst_edges, min_label_ts):
        max_history_ts = None
        if len(trip_edges) > 0:
            max_history_ts = np.max(self.timestamps[trip_edges[:, 0]])
            pass

        for edges in (pair_edges, src_edges, dst_edges):
            if len(pair_edges) > 0:# and max_history_ts is None:
                max_history_ts_edges = np.max(self.timestamps[edges[:, 0]])
                if max_history_ts is not None:
                    max_history_ts = max(
                        max_history_ts,
                        max_history_ts_edges)
                else:
                    max_history_ts = max_history_ts_edges
                    pass
                pass
        
        if max_history_ts is None:
            return min_label_ts
        else:
            return max_history_ts
        pass

    def get_node_history_feat_extra(self, history_edges, current_edge_type, other_node):
        feat = np.zeros((len(history_edges), 4), dtype='int64')

        for i, (eid, direction) in enumerate(history_edges):
            feat[i, 1] = self.edge_types[eid]
            if direction == 0:
                feat[i, 0] = self.dst_nodes[eid]
                feat[i, 3] = int(self.dst_nodes[eid] == other_node)
            else:
                feat[i, 0] = self.src_nodes[eid]
                feat[i, 3] = int(self.src_nodes[eid] == other_node)

                pass
            feat[i, 2] = int(self.edge_types[eid] == current_edge_type)
            pass

        return feat
        pass
    
    def encode_node_history_edges(
            self, history_edges, max_history_ts, current_edge_type,
            other_node):
        feat = self.encode_history_edges(
            history_edges,
            max_history_ts)
        feat_extra = self.get_node_history_feat_extra(
            history_edges, current_edge_type, other_node)
        feat = np.concatenate(
            (feat, feat_extra),
            axis=-1)
        feat = feat_utils.merge_category(feat, self.config['node_history_feat_dim'])
        return feat
    
    def __getitem__(self, idx):
        eid = np.random.choice(self.positive_eids)
        if random.random() < self.config['neg_sample_proba']:
            point_data = self.get_point_data_neg(eid)
            pass
        else:
            point_data = self.get_point_data_pos(eid)
            pass
        
        config = self.config

        label_bins, label = label_utils.get_label(
            point_data.target_ts, point_data.max_history_ts,
            point_data.max_future_ts, self.config['neg_sample_num'],
            self.config['label_bin_size'])

        edge_feat = self.get_edge_feat(
            point_data.src_node, point_data.dst_node,
            point_data.edge_type)

        history_edges_trip = point_data.history_edges_triplet
        history_edges_pair = point_data.history_edges_pair
        history_edges_src = point_data.history_edges_src
        history_edges_dst = point_data.history_edges_dst

        max_history_ts = point_data.max_history_ts
        
        trip_feat = self.encode_history_edges(
            history_edges_trip,
            max_history_ts)
        
        trip_feat = feat_utils.merge_category(
            trip_feat, config['trip_feat_dim'])

        if 'node_feat_file' not in self.config:
            trip_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_trip)
            pair_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_pair)
            src_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_src)
            dst_feat_extra_b = self.get_history_edge_feat_b(
                history_edges_dst)
            pass
        else:
            trip_feat_extra_b = np.zeros((len(history_edges_trip), 1), dtype='float32')
            pair_feat_extra_b = np.zeros((len(history_edges_pair), 1), dtype='float32')
            src_feat_extra_b = np.zeros((len(history_edges_pair), 1), dtype='float32')
            dst_feat_extra_b = np.zeros((len(history_edges_pair), 1), dtype='float32')
            pass
                    
        pair_feat = self.encode_history_edges(
            history_edges_pair,
            max_history_ts)

        pair_feat_extra = self.get_pair_feat_extra(
            point_data.edge_type, history_edges_pair)

        pair_feat = np.concatenate(
            (pair_feat, pair_feat_extra),
            axis=-1)

        pair_feat = feat_utils.merge_category(
            pair_feat, config['pair_feat_dim'])

        src_feat = self.encode_node_history_edges(
            history_edges_src, max_history_ts, point_data.edge_type,
            point_data.dst_node)
        dst_feat = self.encode_node_history_edges(
            history_edges_dst, max_history_ts, point_data.edge_type,
            point_data.src_node)
        
        label_feat = feat_utils.get_label_feat(
            label_bins, config['label_bin_size'],
            max_history_ts,
            config['max_label_class']
        )
        label_feat = feat_utils.merge_category(
            label_feat, config['label_feat_dim'])

        return {
            'label': torch.from_numpy(label),
            'edge_feat': torch.from_numpy(edge_feat),
            'trip_feat': torch.from_numpy(trip_feat),
            'pair_feat': torch.from_numpy(pair_feat),
            'label_feat': torch.from_numpy(label_feat),
            'trip_feat_extra_b': torch.from_numpy(trip_feat_extra_b),
            'pair_feat_extra_b': torch.from_numpy(pair_feat_extra_b),
            'src_feat_extra_b': torch.from_numpy(src_feat_extra_b),
            'dst_feat_extra_b': torch.from_numpy(dst_feat_extra_b),
            'src_feat': torch.from_numpy(src_feat),
            'dst_feat': torch.from_numpy(dst_feat),
            'eid': idx
        }

    def __len__(self):
        # return len(self.positive_eids)
        return self.num
    pass


class DygDatasetTest(DygDataset):
    def __init__(self, config, split):
        super().__init__(config, 'train')

        folder = config['dataset_path']
        filename = split + '_data.tar'
        self.tar_file = tarfile.open(os.path.join(
            config['dataset_path'], filename), 'r')
        self.names = self.tar_file.getnames()

        self.test_src_nodes = np.load(
            os.path.join(folder, f'{split}_src_nodes.npy'))
        self.test_dst_nodes = np.load(
            os.path.join(folder, f'{split}_dst_nodes.npy'))
        self.test_edge_types = np.load(
            os.path.join(folder, f'{split}_edge_types.npy'))
        
        self.test_start_timestamps = np.load(
            os.path.join(folder, f'{split}_start_timestamps.npy'))
        self.test_end_timestamps = np.load(
            os.path.join(folder, f'{split}_end_timestamps.npy'))
        
        pass

    def get_point_data(self, idx) -> PointDataTest:
        name = self.names[idx]
        fin = self.tar_file.extractfile(name)
        return pickle.load(fin)

    def get_edge_feat(self, idx):
        if 'node_feat_file' in self.config:
            src_node = self.test_src_nodes[idx]
            dst_node = self.test_dst_nodes[idx]
            edge_type = self.test_edge_types[idx]
            src_feat = self.node_feat[src_node] + 1 # 1 for -1
            dst_feat = self.node_feat[dst_node] + 1
            edge_type_feat = self.edge_type_feat[edge_type] + 1

            edge_feat = np.hstack((src_feat, dst_feat, edge_type_feat))
            edge_feat = feat_utils.merge_category(
                edge_feat, self.config['edge_feat_dim'])
            pass
        else:
            edge_feat = np.zeros((1, 1), dtype='int64')
            pass
        
        return edge_feat
    
    def __getitem__(self, idx):        
        config = self.config

        point_data = self.get_point_data(idx)
        edge_feat = self.get_edge_feat(idx)
        history_edges_trip = point_data.history_edges_triplet
        history_edges_pair = point_data.history_edges_pair
        history_edges_src = point_data.history_edges_src
        history_edges_dst = point_data.history_edges_dst    
        
        start_ts = self.test_start_timestamps[idx]
        end_ts = self.test_end_timestamps[idx]

        max_history_ts = self.get_max_history_ts(
            history_edges_trip,
            history_edges_pair,
            history_edges_src,
            history_edges_dst,            
            start_ts)

        trip_feat = self.encode_history_edges(
            history_edges_trip,
            max_history_ts)
        trip_feat = feat_utils.merge_category(trip_feat, config['trip_feat_dim'])
        
        pair_feat = self.encode_history_edges(
            history_edges_pair,
            max_history_ts)

        pair_feat_extra = self.get_pair_feat_extra(
            self.test_edge_types[idx], history_edges_pair)

        pair_feat = np.concatenate(
            (pair_feat, pair_feat_extra),
            axis=-1)

        pair_feat = feat_utils.merge_category(pair_feat, config['pair_feat_dim'])

        if 'node_feat_file' not in self.config:
            trip_feat_extra_b = torch.from_numpy(self.get_history_edge_feat_b(history_edges_trip))
            pair_feat_extra_b = torch.from_numpy(self.get_history_edge_feat_b(history_edges_pair))
            pass
        else:
            trip_feat_extra_b = torch.zeros((len(history_edges_trip), 1))
            pair_feat_extra_b = torch.zeros((len(history_edges_pair), 1))
            pass

        src_feat = self.encode_node_history_edges(
            history_edges_src, max_history_ts, self.test_edge_types[idx],
            self.test_dst_nodes[idx])
        dst_feat = self.encode_node_history_edges(
            history_edges_dst, max_history_ts, self.test_edge_types[idx],
            self.test_src_nodes[idx])

        label_bins, label_weights = label_utils.get_predict_bins(
            start_ts, end_ts, config['label_bin_size'])
        label_feat = feat_utils.get_label_feat(
            label_bins, config['label_bin_size'],
            max_history_ts,
            config['max_label_class']
        )
        label_feat = feat_utils.merge_category(
            label_feat, config['label_feat_dim'])

        return {
            'label_bins': torch.from_numpy(label_bins),
            'label_weights': torch.from_numpy(label_weights),
            'edge_feat': torch.from_numpy(edge_feat),
            'trip_feat': torch.from_numpy(trip_feat),
            'pair_feat': torch.from_numpy(pair_feat),
            'label_feat': torch.from_numpy(label_feat),
            'trip_feat_extra_b': trip_feat_extra_b,
            'pair_feat_extra_b': pair_feat_extra_b,
            'src_feat_extra_b': src_feat_extra_b,
            'src_feat': torch.from_numpy(src_feat),
            'dst_feat': torch.from_numpy(dst_feat),

            'eid': idx
        }

    def __len__(self):
        # return len(self.names)
        return 1000
    pass


def collate_seq(feat_list):
    batch_size = len(feat_list)
    feat_max_len = np.max([feat.shape[0] for feat in feat_list])
    feat_dim = feat_list[0].shape[1]
    feat = torch.zeros(
        (batch_size, feat_max_len, feat_dim),
        dtype=feat_list[0].dtype)
    mask = torch.zeros((batch_size, feat_max_len))

    for i, ifeat in enumerate(feat_list):
        size = ifeat.shape[0]
        feat[i, :size, :] = ifeat
        mask[i, :size] = 1
        pass

    return feat, mask


def dyg_collate_fn(batch):
    edge_feat = torch.cat([b['edge_feat'] for b in batch], dim=0)
    label, label_mask = collate_seq(
        [b['label'][:, None] for b in batch])
    label = label.squeeze(-1)

    trip_feat, trip_mask = collate_seq([b['trip_feat'] for b in batch])
    pair_feat, pair_mask = collate_seq([b['pair_feat'] for b in batch])

    trip_feat_extra_b, _ = collate_seq([b['trip_feat_extra_b'] for b in batch])
    pair_feat_extra_b, _ = collate_seq([b['pair_feat_extra_b'] for b in batch])
    src_feat_extra_b, _ = collate_seq([b['src_feat_extra_b'] for b in batch])
    dst_feat_extra_b, _ = collate_seq([b['dst_feat_extra_b'] for b in batch])

    label_feat, label_mask = collate_seq([b['label_feat'] for b in batch])
    eids = [b['eid'] for b in batch]

    src_feat, src_mask = collate_seq([b['src_feat'] for b in batch])
    dst_feat, dst_mask = collate_seq([b['dst_feat'] for b in batch])
    
    return {
        'label': label,
        'edge_feat': edge_feat,
        'trip_feat': trip_feat,
        'trip_mask': trip_mask,
        'pair_feat': pair_feat,
        'pair_mask': pair_mask,
        'src_feat': src_feat,
        'dst_feat': dst_feat,
        'src_mask': src_mask,
        'dst_mask': dst_mask,
        'trip_feat_extra_b': trip_feat_extra_b,
        'pair_feat_extra_b': pair_feat_extra_b,
        'src_feat_extra_b': src_feat_extra_b,
        'dst_feat_extra_b': dst_feat_extra_b,
        'label_feat': label_feat,
        'label_mask': label_mask,
        'eid': eids
    }
    pass


def dyg_test_collate_fn(batch):
    edge_feat = torch.cat([b['edge_feat'] for b in batch], dim=0)
    label_bins, label_mask = collate_seq(
        [b['label_bins'][:, None] for b in batch])
    label_bins = label_bins.squeeze(-1)
    label_weights, label_mask = collate_seq(
        [b['label_weights'][:, None] for b in batch])
    label_weights = label_weights.squeeze(-1)

    trip_feat, trip_mask = collate_seq([b['trip_feat'] for b in batch])
    pair_feat, pair_mask = collate_seq([b['pair_feat'] for b in batch])

    trip_feat_extra_b, _ = collate_seq([b['trip_feat_extra_b'] for b in batch])
    pair_feat_extra_b, _ = collate_seq([b['pair_feat_extra_b'] for b in batch])

    src_feat, src_mask = collate_seq([b['src_feat'] for b in batch])
    dst_feat, dst_mask = collate_seq([b['dst_feat'] for b in batch])

    label_feat, label_mask = collate_seq([b['label_feat'] for b in batch])
    eids = [b['eid'] for b in batch]

    return {
        'label_bins': label_bins,
        'label_weights': label_weights,
        'edge_feat': edge_feat,
        'trip_feat': trip_feat,
        'trip_mask': trip_mask,
        'pair_feat': pair_feat,
        'pair_mask': pair_mask,
        'trip_feat_extra_b': trip_feat_extra_b,
        'pair_feat_extra_b': pair_feat_extra_b,
        'src_feat': src_feat,
        'dst_feat': dst_feat,
        'src_mask': src_mask,
        'dst_mask': dst_mask,
        'label_feat': label_feat,
        'label_mask': label_mask,
        'eid': eids
    }
    pass
    

class RandomDropSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially, always in the same order.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, dataset, drop_rate):
        self.dataset = dataset
        self.drop_rate = drop_rate
        self.drop_num = int(len(dataset) * drop_rate)

    def __iter__(self):
        arange = np.arange(len(self.dataset))
        np.random.shuffle(arange)
        indices = arange[: (1-self.drop_num)]
        return iter(np.sort(indices))
            
    def __len__(self):
        return len(self.dataset) - self.drop_num
