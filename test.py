import torch
from torchviz import make_dot

from pan import PAN

input_channels = 3
num_classes = 22
model = PAN(input_channels, num_classes)
y = model(torch.randn(1, 3, 512, 512))
make_dot(y.mean(), params=dict(model.named_parameters()))
print(model)
