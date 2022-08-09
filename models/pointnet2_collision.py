import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg,PointNetSetAbstraction,PointNetFeaturePropagation
from model import * 
from kernel import *


class PointNetEncoder(nn.Module):
    def __init__(self):
        super(PointNetEncoder, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, 128, 1)

    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        l0_points = xyz
        l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        # Feature Propagation layers
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        # FC layers
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        x = self.drop1(feat)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        return x


class CollisionDecoder(nn.Module):
    def __init__(self, robot_arm_fk_len = 15):
        super(CollisionDecoder, self).__init__()
        self.fc1 = nn.Linear(128 + robot_arm_fk_len, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)
        return x




class get_model(nn.Module):
    def __init__(self, robot_arm_fk_len = 15):
        super(get_model, self).__init__()
        self.encoder = PointNetEncoder()
        self.decoder = CollisionDecoder(robot_arm_fk_len)
        self.enviroment_feature = None
        self.panda_feature = PandaConfig()


    def forward(self, xyz, robot_configs):
        robot_configs = self.panda_feature(robot_configs)
        x = self.encoder(xyz)
        x = self.enviroment_feature.repeat(robot_configs.size(dim = 0))
        x = torch.cat((x, robot_configs), dim = 1)
        x = self.decoder(x)

        return x



class PandaConfig():
    def __init__(self):
        self.panda = PandaFK()
    
    def __call__(self, cfg):
        fk = self.panda.fkine(cfg)
        return torch.flatten(fk, start_dim=1)

