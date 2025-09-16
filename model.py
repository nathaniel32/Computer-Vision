import torch
import torch.nn as nn
from torchinfo import summary

class Natnet(nn.Module):
    def __init__(self, input_shape, hidden_units, output_shape, size):
        super(Natnet, self).__init__()
        self.size = size
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        with torch.no_grad():
            dummy = torch.zeros(1, input_shape, size, size)
            out = self.conv_block_1(dummy)
            out = self.conv_block_2(out)
            n_features = out.numel()
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=n_features, out_features=output_shape),
            nn.Dropout(0.2)
        )
    
    def info(self):
        summary(self, input_size=[1, 3, self.size, self.size])
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x