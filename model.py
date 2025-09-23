import torch
import torch.nn as nn
from torchvision.models import vgg16

class ConditionalSegmentationModel(nn.Module):
    def __init__(self, n_class, emb_dim=128, dropout_rate=0.1):
        super().__init__()
        self.n_class = n_class
        self.emb_dim = emb_dim
        
        # Enhanced class embedding with dropout
        self.class_embedding = nn.Sequential(
            nn.Embedding(n_class, emb_dim),
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