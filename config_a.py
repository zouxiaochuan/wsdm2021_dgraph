import constants
import numpy as np

node_feat_dim = np.array([32, 22, 15, 416, 62, 223, 19, 27]) + 2
edge_type_feat_dim = np.array([247, 19, 142]) + 2
edge_feat_dim = np.concatenate(
    (node_feat_dim, node_feat_dim, edge_type_feat_dim))

max_label_class = 800

config = {
    'train_file': '../data_set/edges_train_A.csv',
    'val_file': '../input_A_initial.csv',
    'test_file': '../input_A.csv',
    'node_feat_file': '../data_set/node_features.csv',
    'edge_type_feat_file': '../data_set/edge_type_features.csv',
    'dataset_path': '../dataset_A',
    'neg_sample_interval': 60 * 24 * 60 * 60,
    'neg_sample_num': 1,
    'train_start': '20150101',
    'max_train_ts': 1494705600,
    'label_bin_size': 24 * 60 * 60,
    'edge_feat_dim': edge_feat_dim.tolist(),
    'trip_feat_dim': constants.TE_FEAT_DIM.tolist(),
    'pair_feat_dim': np.concatenate((constants.TE_FEAT_DIM, edge_type_feat_dim, [2])).tolist(),
    'label_feat_dim': [max_label_class, constants.NUM_WEEKDAY],
    'max_label_class': max_label_class,
    'pair_edge_feat_num': len(edge_type_feat_dim),


    'hidden_size': 128,
    'num_attention_heads': 4,
    'num_hidden_layers_trip': 3,
    'num_hidden_layers_pair': 3,
    'num_hidden_layers2': 3,
    'num_hidden_layers_feat': 2,
    'batch_size': 128,
    'learning_rate': 0.0001,
    'num_data_workers': 4,
    'hidden_dropout_prob': 0,
    'attention_probs_dropout_prob': 0,

    'gpus': 8,
    'accelerator': 'ddp'
    
}

config['edge_feat_emb_num'] = int(np.sum(config['edge_feat_dim']))
config['trip_feat_emb_num'] = int(np.sum(config['trip_feat_dim']))
config['pair_feat_emb_num'] = int(np.sum(config['pair_feat_dim']))
config['label_feat_emb_num'] = int(np.sum(config['label_feat_dim']))
