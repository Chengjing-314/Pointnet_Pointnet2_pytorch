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
    def __init__(self, root, arg):
        self.root = root
        self.world_num = arg.world_num
        self.camera_num = arg.cam_num
        self.filter = arg.filter
        self.filter_distance = arg.filter_distance
        self.cam_path = os.path.join(self.root, 'cam_'+ str(self.camera_num))
        self.process_data = arg.process_data

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
        world_idx = idx // self.camera_num
        cam_idx = idx % self.camera_num
        config, label = self.load_config_label(os.path.join(self.cam_path, "world_"+ str(world_idx)))
        pc = self.load_pc(os.path.join(self.cam_path, str(world_idx), str(cam_idx), 'pc.h5'))
        return config, label, pc

if __name__ == '__main__':
    import torch

    data = CollisionDataLoader()
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True, num_workers=8)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
