from concurrent.futures import process
import os
import numpy as np
import warnings
import h5py

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class CollisionDataLoader(Dataset):
    def __init__(self, root, arg, num_config, config_batch_size, split = 'train'):
        self.root = root
        # self.world_num = arg.world_num
        # self.camera_num = arg.cam_num
        # self.filter = arg.filter
        # self.filter_distance = arg.filter_distance
        # self.process_data = arg.process_data
        self.world_num = 70
        self.camera_num = 8
        self.filter = True
        self.filter_distance = 1.45
        self.process_data = False
        
        self.split = split
        self.pc_tracker = [ConfigTracker(num_config, config_batch_size, split) for _ in range(self.world_num * self.camera_num)]

        # self.collision_label = os.path.join(self.cam_path, 'pointcloud_collision_label.h5')
        # self.pc = os.path.join(self.cam_path, 'pointcloud.h5')

    def load_pc(self, path):
        with h5py.File(path, 'r') as hf:
            pc = torch.tensor(hf["pointcloud"][:])

        if self.filter:
            pc = pc[pc[:, 2] < self.filter_distance]
        
        if self.process_data:
            pc = pc_normalize(pc)
        
        return pc
    
    def load_config_label(self, path):
        config = torch.load(os.path.join(path, 'robot_config.pt'))
        label = torch.load(os.path.join(path, 'collision_label.pt'))
        
        return config, label

    def __len__(self):
        return self.world_num * self.camera_num
    
    def __getitem__(self, idx):
        
        if self.split == 'train':
            world_idx = int(idx // self.camera_num)
            cam_idx = int(idx % self.camera_num)
        elif self.split == 'validation':
            world_idx = idx - 1
            cam_idx = 8
        elif self.split == 'test':
            world_idx = idx - 1
            cam_idx = 9
            
        world_path = os.path.join(self.root, 'world_' + str(world_idx))
        cam_path = os.path.join(world_path, 'cam_' + str(cam_idx))
        config, label = self.load_config_label(world_path)
        pc = self.load_pc(os.path.join(cam_path,'pc.h5'))

        start, end = self.pc_tracker[idx].step()
        
        return config[start:end], label[start:end], pc
    
    
class ConfigTracker():
    def __init__(self, num_config, batch_size, split, validation_reset = 5000, test_reset = 7500):
        self.batch_size = batch_size
        if split == 'train':
            self.config_ptr = 0
            self.reset = 0
            self.num_config = num_config
        elif split == 'validation':
            self.config_ptr = validation_reset
            self.reset = validation_reset
            self.num_config = self.reset + num_config
        elif split == 'test':
            self.config_ptr = test_reset
            self.reset = test_reset
            self.num_config = self.reset + num_config
        
    def step(self):
        if self.config_ptr + self.batch_size < self.num_config:
            start = self.config_ptr
            self.config_ptr += self.batch_size
            end = self.config_ptr
            return start, end
        else:
            start = self.config_ptr
            end = self.num_config
            self.config_ptr = self.reset
            return start, end
        

if __name__ == '__main__':
    import torch
    import time
    root = '/home/chengjing/Desktop/data_generation_2'    
    
    data = CollisionDataLoader(root, None, 10000, 128, 'train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True, num_workers=10)
    i = 0
    start = time.time()
    for config, label, pc in DataLoader:
        print(config.shape, label.shape, pc.shape)
    end = time.time()
    print("script running time: ", end - start)
