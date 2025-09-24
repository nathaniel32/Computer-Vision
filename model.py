import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch.nn.functional as F

class ConditionalFCN(nn.Module):
    def __init__(self, n_classes, emb_dim, dropout_rate):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        
        # Enhanced class embedding with dropout
        self.class_embedding = nn.Sequential(
            nn.Embedding(n_classes, emb_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(emb_dim)
        )
        nn.init.xavier_uniform_(self.class_embedding[0].weight)

        # Backbone with batch normalization
        vgg = vgg16(pretrained=True)
        features = list(vgg.features.children())
        self.backbone = nn.Sequential(*features[:-1])
        
        # Add batch normalization after backbone
        self.backbone_bn = nn.BatchNorm2d(512)
        
        # Freeze early layers for stability
        for i, layer in enumerate(self.backbone):
            if i < 10:
                for param in layer.parameters():
                    param.requires_grad = False

        # Enhanced FiLM layers with residual connections
        self.film_projection = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512 * 2)  # gamma and beta
        )
        
        # More sophisticated decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 1, 1)
        )

        # Progressive upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(8, 1, 4, 2, 1),
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, class_id, binarize=False):
        batch_size = x.size(0)
        
        # Extract and normalize features
        features = self.backbone(x)  # (B, 512, H, W)
        features = self.backbone_bn(features)
        
        # Get class embedding
        emb = self.class_embedding(class_id)  # (B, emb_dim)
        
        # Apply FiLM conditioning
        film_params = self.film_projection(emb)  # (B, 512*2)
        gamma, beta = film_params.chunk(2, dim=1)  # Each: (B, 512)
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, 512, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, 512, 1, 1)
        
        # Apply FiLM modulation with residual connection
        features_modulated = gamma * features + beta
        features = features + features_modulated  # Residual connection
        
        # Decode
        score = self.decoder(features)
        out = self.upsample(score)
        out = torch.sigmoid(out)

        if binarize:
            out = (out >= 0.5).float()
        
        return out

########################################

class ConditionalUNet(nn.Module):
    def __init__(self, n_classes, emb_dim, dropout_rate, input_channels=3):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        
        # Enhanced class embedding
        self.class_embedding = nn.Sequential(
            nn.Embedding(n_classes, emb_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(emb_dim)
        )
        nn.init.xavier_uniform_(self.class_embedding[0].weight)
        
        # Encoder (Contracting path)
        self.enc1 = DoubleConv(input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # FiLM conditioning for bottleneck
        self.film_projection = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024 * 2)  # gamma and beta
        )
        
        # Decoder (Expanding path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 because of skip connection
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # Output layer
        self.final_conv = nn.Conv2d(64, 1, 1)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, class_id, binarize=False):
        # Encoder path
        enc1 = self.enc1(x)  # 64 channels
        enc2 = self.enc2(self.pool(enc1))  # 128 channels
        enc3 = self.enc3(self.pool(enc2))  # 256 channels
        enc4 = self.enc4(self.pool(enc3))  # 512 channels
        
        # Bottleneck with conditional modulation
        bottleneck = self.bottleneck(self.pool(enc4))  # 1024 channels
        
        # Apply FiLM conditioning to bottleneck
        emb = self.class_embedding(class_id)  # (B, emb_dim)
        film_params = self.film_projection(emb)  # (B, 1024*2)
        gamma, beta = film_params.chunk(2, dim=1)  # Each: (B, 1024)
        
        # Reshape for broadcasting
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, 1024, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)    # (B, 1024, 1, 1)
        
        # Apply FiLM modulation with residual connection
        bottleneck_modulated = gamma * bottleneck + beta
        bottleneck = bottleneck + bottleneck_modulated  # Residual connection
        bottleneck = self.dropout(bottleneck)
        
        # Decoder path with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)
        dec4 = self.dropout(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)  # Skip connection
        dec3 = self.dec3(dec3)
        dec3 = self.dropout(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)  # Skip connection
        dec2 = self.dec2(dec2)
        dec2 = self.dropout(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)  # Skip connection
        dec1 = self.dec1(dec1)
        
        # Final output
        out = self.final_conv(dec1)
        out = torch.sigmoid(out)
        
        if binarize:
            out = (out >= 0.5).float()
        
        return out

########################################

class DoubleConv(nn.Module):
    """Double Convolution block used in U-Net"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Upsample g1 to match x1 size if needed
        if g1.size()[2:] != x1.size()[2:]:
            g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=False)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, n_classes, emb_dim, dropout_rate, input_channels=3):
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        
        # Class embedding
        self.class_embedding = nn.Sequential(
            nn.Embedding(n_classes, emb_dim),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(emb_dim)
        )
        nn.init.xavier_uniform_(self.class_embedding[0].weight)
        
        # Encoder
        self.enc1 = DoubleConv(input_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)
        
        # FiLM conditioning
        self.film_projection = nn.Sequential(
            nn.Linear(emb_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 1024 * 2)
        )
        
        # Attention gates
        self.att4 = AttentionGate(512, 512, 256)
        self.att3 = AttentionGate(256, 256, 128)
        self.att2 = AttentionGate(128, 128, 64)
        self.att1 = AttentionGate(64, 64, 32)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, class_id, binarize=False):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck with conditioning
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Apply FiLM conditioning
        emb = self.class_embedding(class_id)
        film_params = self.film_projection(emb)
        gamma, beta = film_params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        
        bottleneck_modulated = gamma * bottleneck + beta
        bottleneck = bottleneck + bottleneck_modulated
        bottleneck = self.dropout(bottleneck)
        
        # Decoder with attention gates
        dec4 = self.upconv4(bottleneck)
        enc4_att = self.att4(enc4, dec4)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)
        dec4 = self.dropout(dec4)
        
        dec3 = self.upconv3(dec4)
        enc3_att = self.att3(enc3, dec3)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)
        dec3 = self.dropout(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2_att = self.att2(enc2, dec2)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)
        dec2 = self.dropout(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1_att = self.att1(enc1, dec1)
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.dec1(dec1)
        
        out = self.final_conv(dec1)
        out = torch.sigmoid(out)
        
        if binarize:
            out = (out >= 0.5).float()
        
        return out

#ConditionalSegmentationModel = ConditionalFCN
ConditionalSegmentationModel = ConditionalUNet
#ConditionalSegmentationModel = AttentionUNet

if __name__ == "__main__":
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # FCN
    model_fcn = ConditionalFCN(n_classes=10).to(device)

    # Standard U-Net
    model_unet = ConditionalUNet(n_classes=10).to(device)
    
    # Attention U-Net
    model_att_unet = AttentionUNet(n_classes=10).to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    class_ids = torch.tensor([1, 5]).to(device)
    
    with torch.no_grad():
        output_fcn = model_fcn(x, class_ids)
        output_unet = model_unet(x, class_ids)
        output_att_unet = model_att_unet(x, class_ids)
    
    print("="*50)
    print(f"\nInput shape: {x.shape}\n")
    print(f"U-Net output shape: {output_fcn.shape}")
    print(f"U-Net output shape: {output_unet.shape}")
    print(f"Attention U-Net output shape: {output_att_unet.shape}\n")
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"FCN parameters: {count_parameters(model_fcn):,}")
    print(f"U-Net parameters: {count_parameters(model_unet):,}")
    print(f"Attention U-Net parameters: {count_parameters(model_att_unet):,}\n")
    print("="*50)