import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

""" 
class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(Up, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes: int, n_channels: int = 3, bilinear: bool = False):
        super(UNet, self).__init__()
        
        if n_classes < 1:
            raise ValueError("n_classes must be at least 1")
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        # Encoder (contracting path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (expansive path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        
        return logits

SegmentationModel = UNet """


class DoubleConv(nn.Module):
    """Double convolution block used in U-Net decoder"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(Up, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class ResNet50Encoder(nn.Module):
    """ResNet-50 encoder for U-Net"""
    def __init__(self, n_channels: int = 3, pretrained: bool = True):
        super(ResNet50Encoder, self).__init__()
        
        # Load pre-trained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Modify first conv layer if input channels != 3
        if n_channels != 3:
            resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # Initialize new conv1 layer
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        
        # Extract feature extraction layers
        self.conv1 = resnet.conv1      # 3 -> 64 channels
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1    # 64 -> 256 channels
        self.layer2 = resnet.layer2    # 256 -> 512 channels  
        self.layer3 = resnet.layer3    # 512 -> 1024 channels
        self.layer4 = resnet.layer4    # 1024 -> 2048 channels
        
    def forward(self, x):
        # Store features at different resolutions for skip connections
        features = []
        
        # Initial conv block: H/2 x W/2
        x = self.conv1(x)           # 64 channels, H/2 x W/2
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)          # feat0: 64 channels
        
        # MaxPool: H/4 x W/4  
        x = self.maxpool(x)         # 64 channels, H/4 x W/4
        
        # ResNet blocks
        x = self.layer1(x)          # 256 channels, H/4 x W/4
        features.append(x)          # feat1: 256 channels
        
        x = self.layer2(x)          # 512 channels, H/8 x W/8
        features.append(x)          # feat2: 512 channels
        
        x = self.layer3(x)          # 1024 channels, H/16 x W/16
        features.append(x)          # feat3: 1024 channels
        
        x = self.layer4(x)          # 2048 channels, H/32 x W/32
        features.append(x)          # feat4: 2048 channels
        
        return features

class UNetResNet50(nn.Module):
    def __init__(self, n_classes: int, n_channels: int = 3, pretrained: bool = True, bilinear: bool = True):
        super(UNetResNet50, self).__init__()
        
        if n_classes < 1:
            raise ValueError("n_classes must be at least 1")
        
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear
        
        # ResNet-50 Encoder
        self.encoder = ResNet50Encoder(n_channels, pretrained)
        
        # Feature channels from ResNet-50: [64, 256, 512, 1024, 2048]
        # We'll use these for skip connections
        
        # Center/Bridge block
        self.center = DoubleConv(2048, 1024)
        
        # Decoder blocks with proper channel matching
        # Up from 1024 + skip(1024) = 2048 -> 512
        self.up1 = Up(1024 + 1024, 512, bilinear)
        
        # Up from 512 + skip(512) = 1024 -> 256  
        self.up2 = Up(512 + 512, 256, bilinear)
        
        # Up from 256 + skip(256) = 512 -> 128
        self.up3 = Up(256 + 256, 128, bilinear)
        
        # Up from 128 + skip(64) = 192 -> 64
        self.up4 = Up(128 + 64, 64, bilinear)
        
        # Final upsampling to original size
        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = DoubleConv(64, 64)
        
        # Output layer
        self.outc = OutConv(64, n_classes)
        
        # Initialize decoder weights
        self._initialize_decoder_weights()
        
    def _initialize_decoder_weights(self):
        """Initialize decoder weights (encoder already pre-trained)"""
        decoder_modules = [
            self.center, self.up1, self.up2, self.up3, self.up4, 
            self.final_up, self.final_conv, self.outc
        ]
        
        for module in decoder_modules:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get encoder features: [feat0, feat1, feat2, feat3, feat4]
        # Shapes: [64, 256, 512, 1024, 2048] channels
        # Resolutions: [H/2, H/4, H/8, H/16, H/32]
        features = self.encoder(x)
        
        # Center/bottleneck
        center = self.center(features[4])  # 2048 -> 1024
        
        # Decoder path with skip connections
        # features[3]: 1024 channels at H/16
        up1 = self.up1(center, features[3])     # 1024+1024 -> 512
        
        # features[2]: 512 channels at H/8  
        up2 = self.up2(up1, features[2])        # 512+512 -> 256
        
        # features[1]: 256 channels at H/4
        up3 = self.up3(up2, features[1])        # 256+256 -> 128
        
        # features[0]: 64 channels at H/2
        up4 = self.up4(up3, features[0])        # 128+64 -> 64
        
        # Final upsampling to original resolution H x W
        final = self.final_up(up4)               # H/2 -> H
        final = self.final_conv(final)
        
        # Output
        logits = self.outc(final)
        
        return logits

SegmentationModel = UNetResNet50