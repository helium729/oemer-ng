"""
OMR Model architecture.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class AttentionModule(nn.Module):
    """Self-attention module."""
    
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        batch, channels, height, width = x.size()
        
        # Compute attention
        query = self.query(x).view(batch, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch, -1, height * width)
        attention = torch.softmax(torch.bmm(query, key), dim=-1)
        
        value = self.value(x).view(batch, -1, height * width)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch, channels, height, width)
        
        return self.gamma * out + x


class OMRModel(nn.Module):
    """
    OMR Model for optical music recognition.
    
    A convolutional neural network for processing sheet music images.
    """
    
    def __init__(
        self,
        n_channels: int = 3,
        num_classes: int = 128,
        base_channels: int = 64,
        use_attention: bool = True
    ):
        """
        Initialize OMR model.
        
        Args:
            n_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of output classes
            base_channels: Base number of channels in the network
            use_attention: Whether to use attention modules
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        
        # Encoder
        self.enc1 = ConvBlock(n_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Residual blocks
        self.res1 = ResidualBlock(base_channels * 8)
        self.res2 = ResidualBlock(base_channels * 8)
        
        # Attention
        if use_attention:
            self.attention = AttentionModule(base_channels * 8)
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(base_channels * 8, num_classes)
    
    def get_feature_maps(self, x):
        """Extract intermediate feature maps."""
        features = []
        
        x = self.enc1(x)
        features.append(x)
        x = self.pool(x)
        
        x = self.enc2(x)
        features.append(x)
        x = self.pool(x)
        
        x = self.enc3(x)
        features.append(x)
        x = self.pool(x)
        
        x = self.enc4(x)
        features.append(x)
        
        return features
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, num_classes)
        """
        # Encoder
        x = self.enc1(x)
        x = self.pool(x)
        
        x = self.enc2(x)
        x = self.pool(x)
        
        x = self.enc3(x)
        x = self.pool(x)
        
        x = self.enc4(x)
        x = self.pool(x)
        
        # Residual blocks
        x = self.res1(x)
        x = self.res2(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Classifier
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
