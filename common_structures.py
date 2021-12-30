# from collections import namedtuple

# SequenceStatus = namedtuple(
#     'SequenceStatus', ['current_index', 'seq', 'direction'],
#     defaults=[None, None, None])

class SequenceStatus(object):
    __slots__ = ('current_index', 'seq')

    def __init__(self,):
        self.current_index = 0
        self.seq = list()
        pass


class PointData(object):
    __slots__ = ('eid', 'target_ts', 'history_edges_triplet',
                 'history_edges_pair', 'history_edges_src', 'history_edges_dst')

    def __init__(self, eid, target_ts, history_edges_triplet,
                 history_edges_pair, history_edges_src, history_edges_dst):
        self.eid = eid
        self.target_ts = target_ts
        self.history_edges_triplet = history_edges_triplet
        self.history_edges_pair = history_edges_pair
        self.history_edges_src = history_edges_src
        self.history_edges_dst = history_edges_dst
        pass

    pass


class PointDataTest(object):
    __slots__ = ('history_edges_triplet', 'history_edges_pair',
                 'history_edges_src', 'history_edges_dst',
                 'start_timestamp', 'end_timestamp')

    def __init__(
            self,
            history_edges_triplet, history_edges_pair,
            history_edges_src, history_edges_dst,
            start_timestamp, end_timestamp):
        self.history_edges_triplet = history_edges_triplet
        self.history_edges_pair = history_edges_pair
        self.history_edges_src = history_edges_src
        self.history_edges_dst = history_edges_dst
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        pass

