import torch
import torch.nn as nn
import torch.nn.functional as F

# Obtain channel attention
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio, pool_types):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Conv1d(gate_channels, gate_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(gate_channels // reduction_ratio, gate_channels, 1, bias=False)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool1d(x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).expand_as(x)
        return x * scale

# A basic CNN module
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
        
# Perform pooling operation along the channels.
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


# Obtain spatial attention
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(in_planes=2, out_planes=1, kernel_size=7)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale
