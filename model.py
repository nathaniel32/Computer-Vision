import torch
import torch.nn as nn
import torch.nn.functional as F

class TNet(nn.Module):
    """Transformation Network"""
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = torch.max(x, 2)[0]
        
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity matrix
        identity = torch.eye(self.k, device=x.device).flatten().unsqueeze(0).repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, self.k, self.k)
        
        return x


class PointNetSegmentation(nn.Module):
    """PointNet for Part Segmentation with RGB"""
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()
        
        self.num_classes = num_classes
        
        # Input transform (XYZ only - 3 channels)
        self.input_transform = TNet(k=3)
        
        # First MLP - process XYZ
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # RGB processing branch - parallel to XYZ
        self.rgb_conv1 = nn.Conv1d(3, 64, 1)
        self.rgb_conv2 = nn.Conv1d(64, 64, 1)
        self.rgb_bn1 = nn.BatchNorm1d(64)
        self.rgb_bn2 = nn.BatchNorm1d(64)
        
        # Feature transform (128 channels - combined XYZ + RGB features)
        self.feature_transform = TNet(k=128)
        
        # Second MLP - process concatenated XYZ + RGB features
        # Input: 64 (XYZ) + 64 (RGB) = 128
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Segmentation MLP
        self.conv6 = nn.Conv1d(1024 + 128, 512, 1)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, num_classes, 1)
        
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # x shape: (B, 6, N) - XYZ + RGB
        num_points = x.size(2)
        
        # Separate XYZ and RGB
        xyz = x[:, :3, :]  # (B, 3, N)
        rgb = x[:, 3:, :]  # (B, 3, N)
        
        # Input transform (only on XYZ)
        input_trans = self.input_transform(xyz)
        xyz_transformed = torch.bmm(xyz.transpose(2, 1), input_trans).transpose(2, 1)
        
        # First MLP - XYZ branch
        xyz_feat = F.relu(self.bn1(self.conv1(xyz_transformed)))
        xyz_feat = F.relu(self.bn2(self.conv2(xyz_feat)))
        
        # RGB branch (parallel processing)
        rgb_feat = F.relu(self.rgb_bn1(self.rgb_conv1(rgb)))
        rgb_feat = F.relu(self.rgb_bn2(self.rgb_conv2(rgb_feat)))
        
        # Concatenate XYZ and RGB features
        combined_feat = torch.cat([xyz_feat, rgb_feat], dim=1)  # (B, 128, N)
        
        # Feature transform (on combined features)
        feature_trans = self.feature_transform(combined_feat)
        combined_feat = torch.bmm(combined_feat.transpose(2, 1), feature_trans).transpose(2, 1)
        
        # Second MLP
        x = F.relu(self.bn3(self.conv3(combined_feat)))
        local_features = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(local_features)))
        
        # Global feature
        global_feature = torch.max(x, 2, keepdim=True)[0]
        global_feature_expanded = global_feature.repeat(1, 1, num_points)
        
        # Concatenate local and global features
        x = torch.cat([local_features, global_feature_expanded], dim=1)
        
        # Segmentation MLP
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout(x)
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.conv9(x)
        
        # Transpose to (batch, points, classes)
        x = x.transpose(2, 1).contiguous()
        
        return x

# demo
""" class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSegmentation, self).__init__()
        # Input: 6 channels (3 XYZ + 3 RGB)
        self.conv1 = nn.Conv1d(6, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.conv5 = nn.Conv1d(128, num_classes, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # (B, 6, N) - XYZ + RGB
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.conv5(x)
        x = x.transpose(2, 1).contiguous()  # (B, num_classes, N) -> (B, N, num_classes)
        return x """