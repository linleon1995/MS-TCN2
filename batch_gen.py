#!/usr/bin/python2.7
import random

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler



def read_data(vid_list_file):
    file_ptr = open(vid_list_file, 'r')
    list_of_samples = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    return list_of_samples


class VAS_Dataset(Dataset):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, list_of_samples):
        self.list_of_samples = list_of_samples
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate


    def __len__(self):
        return len(self.list_of_samples)

    def __getitem__(self, idx):
        vid = self.list_of_samples[idx]
        features = np.load(self.features_path + vid.split('.')[0] + '.npy')
        file_ptr = open(self.gt_path + vid, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(min(np.shape(features)[1], len(content)))
        for i in range(len(classes)):
            classes[i] = self.actions_dict[content[i]]
        input = features[:, ::self.sample_rate]
        target = classes[::self.sample_rate]
        seq_length = input.shape[1]
        mask = np.ones((self.num_classes, seq_length), np.float32)
        return {'input': input, 'target': target, 'mask': mask, 'seq_length': seq_length}



# class WeightedFixedBatchSampler(BatchSampler):
#     """
#     Ensures each batch contains a given class distribution.
#     The lists of indices for each class are shuffled at the start of each call to `__iter__`.
#     Parameters
#     ----------
#     class_samples_per_batch : `numpy.array(int)`
#         The number of samples of each class to include in each batch.
#     class_idxs : 2D list of ints
#         The indices that correspond to samples of each class.
#     n_batches : int
#         The number of batches to yield.
#     """

#     def __init__(self, sampler, batch_size, drop_last, list_of_samples):
#         super().__init__(sampler, batch_size, drop_last)
#         self.batch_size = batch_size
#         # self.sampler = sampler
#         self.list_of_samples = list_of_samples

#     def get_batch(self, start_idxs):
#         selected = []
#         for c, size in enumerate(self.class_samples_per_batch):
#             selected.extend(self.class_idxs[c][start_idxs[c]:start_idxs[c] + size])
#         np.random.shuffle(selected)
#         return selected

#     def __iter__(self):
#         np.random.shuffle(self.list_of_samples)
#         # sampler_iter = iter(self.sampler)
#         while True:
#             try:
#                 idx = 0
#                 batch = []
#                 while True:
#                     data = self.list_of_samples[idx]
#                     idx += 1
#                     if len(batch) > self.batch_size:
#                         break
#                 yield batch      
#                 # batch = [next(sampler_iter) for _ in range(self.batch_size)]
                
#             except StopIteration:
#                 break

#         # [cidx.shuffle() for cidx in self.class_idxs]
#         # start_idxs = np.zeros(self.n_classes, dtype=int)
#         # for bidx in range(self.n_batches):
#         #     yield self.get_batch(start_idxs)
#         #     start_idxs += self.class_samples_per_batch

#     # def __len__(self):
#     #     return len(self.vid_list_file) // self.batch_size # drop last


# class SequenceAlignSampler(BatchSampler):
#     def __iter__(self):
#         # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
#         if self.drop_last:
#             sampler_iter = iter(self.sampler)
#             while True:
#                 try:
#                     batch_input, batch_target = [], []
#                     for _ in range(self.batch_size):
#                         input, target = next(sampler_iter)
#                         batch_input .append(input)
#                         batch_target.append(target)
#                     length_of_sequences = list(map(len, batch_target))

#                     batch_input_tensor = torch.zeros(len(batch_input), torch.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
#                     batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
#                     mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
#                     for i in range(len(batch_input)):
#                         batch_input_tensor[i, :, :torch.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
#                         batch_target_tensor[i, :torch.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
#                         mask[i, :, :torch.shape(batch_target[i])[0]] = torch.ones(self.num_classes, torch.shape(batch_target[i])[0])


#                     batch = [next(sampler_iter) for _ in range(self.batch_size)]
#                     yield batch
#                 except StopIteration:
#                     break
#         else:
#             batch = [0] * self.batch_size
#             idx_in_batch = 0
#             for idx in self.sampler:
#                 batch[idx_in_batch] = idx
#                 idx_in_batch += 1
#                 if idx_in_batch == self.batch_size:
#                     yield batch
#                     idx_in_batch = 0
#                     batch = [0] * self.batch_size
#             if idx_in_batch > 0:
#                 yield batch[:idx_in_batch]


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask

