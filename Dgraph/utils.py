import numpy as np
import torch

### Utility function and class
class EarlyStopMonitor(object):
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        self.epoch_count += 1
        
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        return self.num_round >= self.max_round

class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

# class RandNodeSampler(object):
#     def __init__(self, node_list):
#         self.node_list = np.unique(node_list)

#     def sample(self, size):
#         node_index = np.random.randint(0, len(self.node_list), size)
#         return self.node_list[node_index]

#     def contrast_sample(self, src_l_cut, size, sfeature):

#         node_indexs = []
#         src_sentives = sfeature[src_l_cut, 0]/100
#         dist = np.abs(src_sentives - np.expand_dims(src_sentives, axis=1))
#         # print(f'dist={dist}')
#         dist[np.isnan(dist)] = 0.5
#         # print(f'dist={dist}')
#         dist = dist / np.linalg.norm(dist, ord=1, axis=1, keepdims=True)
#         # print(f'dist={dist.sum(axis=1)}')

#         for i in range(size):
#             node_index = np.random.choice(size, size=1, replace=True, p=dist[i, :])[0]
#             node_indexs.append(src_l_cut[node_index])

#         return np.array(node_indexs)

class RandNodeSampler(object):
    def __init__(self, src):
        self.src = np.unique(src)
        self.size = len(self.src)

    def sample(self, size):
        node_index = np.random.randint(0, self.size, size)
        return self.src[node_index]

    def contrast_sample(self, num_samples, size):

        col_index = np.random.randint(0, num_samples, size)

        return col_index