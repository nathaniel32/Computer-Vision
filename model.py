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
    """PointNet for Part Segmentation"""
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()
        
        self.num_classes = num_classes
        
        # Input transform
        self.input_transform = TNet(k=3)
        
        # First MLP
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Feature transform
        self.feature_transform = TNet(k=64)
        
        # Second MLP
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(64)
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
        num_points = x.size(2)
        
        # Input transform
        input_trans = self.input_transform(x)
        x = torch.bmm(x.transpose(2, 1), input_trans).transpose(2, 1)
        
        # First MLP
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Feature transform
        feature_trans = self.feature_transform(x)
        x = torch.bmm(x.transpose(2, 1), feature_trans).transpose(2, 1)
        
        # Second MLP
        x = F.relu(self.bn3(self.conv3(x)))
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
        #x = F.log_softmax(x, dim=-1)
        
        return x

# demo
""" class PointNetSegmentation(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetSegmentation, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 128, 1)
        self.conv5 = nn.Conv1d(128, num_classes, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # (B, 3, N)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout(x)
        x = self.conv5(x)
        x = x.transpose(2, 1).contiguous()  # (B, 3, N) -> (B, N, 3)
        #x = F.log_softmax(x, dim=-1)
        return x """