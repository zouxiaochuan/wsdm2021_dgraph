from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
import json
import sys
import importlib


def ts2dt(ts):
    date = datetime.fromtimestamp(ts)
    return date.strftime('%Y%m%d')


def every_day_count(config):
    cnts = defaultdict(int)
    df = pd.read_csv(config['train_file'], header=None)
    data = df[[0,1,2,3]].values

    ts = data[:, -1]

    dt = list(map(ts2dt, ts))
    for d in tqdm(dt):
        cnts[d] += 1
        pass
    
    for dt, cnt in sorted(cnts.items()):
        print(f'{dt}:{cnt}')
        pass
    pass


def check_self2self(config):
    df = pd.read_csv(config['train_file'], header=None)
    data = df.values

    index = data[:, 0] == data[:, 1]
    print(index)
    print(data[index])
    print(len(data[index]))
    pass


def check_bilateral(config):
    df = pd.read_csv(config['train_file'], header=None)
    data = df[[0, 1, 2, 3]].values
    
    data = set([tuple(rec) for rec in data[:, :2]])

    count = 0
    for rec in data:
        if (rec[1], rec[0]) in data:
            count += 1
            pass
        pass

    print(count)
    print(count / len(data))


def test_dts(config):
    df = pd.read_csv(config['test_file'], header=None)
    data = df.values
    ts1 = data[:, -2]
    ts2 = data[:, -1]

    dt1 = list(map(ts2dt, ts1))
    dt2 = list(map(ts2dt, ts2))
    
    print(sorted(list(set(dt1+dt2))))
    pass


def test_interval(test_file):
    data = pd.read_csv(test_file).values

    minv = 999999999
    maxv = 0

    vs = []
    for rec in data:
        v = rec[4] - rec[3]
        vs.append(v)
        pass

    vs = np.array(vs)
    print(np.min(vs))
    print(np.max(vs))
    print(np.median(vs))
    print(np.sum(vs>(3*24*60*60)))
    pass


def train_interval(config):
    train_file = config['train_file']
    test_file = config['test_file']
    test_data = pd.read_csv(test_file, header=None).values
    data = pd.read_csv(train_file, header=None)[[0,1,2,3]].values

    test_keys = set([tuple(rec) for rec in test_data[:, :3]])
    data = data[-30000000:]
    last_ts = dict()
    tsdiffs = dict()

    for sid, did, etype, ts in tqdm(data):
        key = (sid, did, etype)
        if key not in test_keys:
            continue
        
        lts = last_ts.get(key)

        if lts is None:
            pass
        else:
            mts = tsdiffs.get(key)
            if mts is None:
                tsdiffs[key] = [ts - lts]
                pass
            else:
                tsdiffs[key].append(ts-lts)
                pass
            pass
        last_ts[key] = ts
        pass

    print(np.mean([np.mean(vs) for vs in tsdiffs.values()]))
    print(np.mean(np.array([np.max(vs) for vs in tsdiffs.values()])>800*24*60*60))

def test_train_interval(train_file, test_file):
    df = pd.read_csv(train_file)
    data = df.values

    train_max_ts = defaultdict(int)
    for rec in tqdm(data):
        key = tuple(rec[:3])
        train_max_ts[key] = max(train_max_ts[key], rec[3])
        pass

    df2 = pd.read_csv(test_file)
    data2 = df2.values

    test_min_ts = defaultdict(int)
    for rec in data2:
        key = tuple(rec[:3])
        val = test_min_ts.get(key)
        if val is None:
            val = rec[3]
        else:
            val = min(rec[3], val)
            pass
        test_min_ts[key] = val
        pass

    tsdiffs = []
    for key, ts in test_min_ts.items():
        ts_train = train_max_ts.get(key)

        if ts_train is not None:
            tsdiff = ts - ts_train
            tsdiffs.append(tsdiff)
        pass

    tsdiffs = np.asarray(tsdiffs)
    print(np.min(tsdiffs))
    print(np.max(tsdiffs))
    print(np.sum(tsdiffs>90*24*60*60))
    

def feat_dims(config):
    data_edge_type = pd.read_csv(config['edge_type_feat_file']).values[:, 1:]
    data_node = pd.read_csv(config['node_feat_file']).values[:, 1:]

    dims_edge_type = data_edge_type.max(axis=0)
    dims_node = data_node.max(axis=0)

    print(dims_edge_type)
    print(dims_node)
    
    pass


def hot_point_data(data):

    counts = defaultdict(int)

    for rec in data:
        counts[tuple(rec[:3])] += 1
        pass

    for i, (key, cnt) in enumerate(
            sorted(counts.items(), key=lambda x: x[1], reverse=True)):
        print(f'{key}: {cnt},{cnt/len(data)}')

        if i > 10:
            break
        pass
    pass


def hot_point(config):
    hot_point_data(pd.read_csv(config['train_file'], header=None).values)
    hot_point_data(pd.read_csv(config['test_file'], header=None).values)
    pass


def check_duplicate(config):
    data = pd.read_csv(config['train_file']).values

    counts = defaultdict(int)
    index = defaultdict(list)
    i = 0
    max_rec = None
    max_count = 0
    for rec in data:
        counts[tuple(rec)] += 1
        index[tuple(rec)].append(i)
        i += 1
        if counts[tuple(rec)] > max_count:
            max_count = counts[tuple(rec)]
            max_rec = tuple(rec)
        pass

    print(np.max(list(counts.values())))
    dup_counts = 0
    for value in counts.values():
        dup_counts += value-1
    print(dup_counts)
    print(index[max_rec])

    dts = [ts2dt(ts) for ts in data[:, 3]]

    counts2 = defaultdict(int)
    for rec, dt in zip(data, dts):
        counts2[tuple(rec[:2]) + (dt,)] += 1
        pass

    print(np.max(list(counts2.values())))


def check_test_new(config):
    data_train = pd.read_csv(config['train_file'], header=None)[[0, 1, 2, 3]].values
    data_val = pd.read_csv(config['val_file'], header=None).values
    data_test = pd.read_csv(config['test_file'], header=None).values

    train_nodes = set([v for v in np.hstack((data_train[:, 0], data_train[:, 1]))])

    val_nodes = set([v for v in np.hstack((data_val[:, 0], data_val[:, 1]))])
    test_nodes = set([v for v in np.hstack((data_test[:, 0], data_test[:, 1]))])

    for n in set.union(val_nodes, test_nodes):
        if n not in train_nodes:
            print(f'{n} not in train_nodes')
            pass
        pass

    train_edge_types = set(data_train[:, 2].tolist())
    val_edge_types = set(data_val[:, 2].tolist())
    test_edge_types = set(data_test[:, 2].tolist())

    for n in set.union(val_edge_types, test_edge_types):
        if n not in train_edge_types:
            print(f'{n} not in train edge types')
            pass
        pass
    pass


def check_ts(config):
    data_train = pd.read_csv(config['train_file'], header=None)[[0, 1, 2, 3]].values
    data_val = pd.read_csv(config['val_file'], header=None).values
    data_test = pd.read_csv(config['test_file'], header=None).values

    last_ts = 0
    for ts in data_train[:, 3]:
        if last_ts > ts:
            print(ts, last_ts)
            break
        last_ts = ts

    print("check finished!")


if __name__ == '__main__':
    config_file = sys.argv[1]
    config = importlib.import_module(config_file).config

    # every_day_count(config)
    # check_self2self(config)
    # test_dts(config)
    # test_train_interval(config['train_file'], config['test_file'])
    # train_interval(config)
    test_interval(config['test_file'])
    # feat_dims(config)
    # hot_point(config)
    # check_duplicate(config)
    # check_bilateral(config)
    # check_test_new(config)
    # check_ts(config)
    pass
