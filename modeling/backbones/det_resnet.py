from torch import nn
import timm
import torch
from .bottleneck import Bottleneck

class Resnet(nn.Module):
    def __init__(self, in_channels=3, layers=50, out_indices = None):
        super().__init__()
        self.layers = layers
        self.input_image_channel = in_channels

        supported_layers = [18, 34, 50, 101, 152]
        assert (
            layers in supported_layers
        ), "supported layers are {} but input layer is {}".format(
            supported_layers, layers
        )
        out_indices = out_indices if out_indices is not None else (1,2,3)
        print("Resnet Backbone with {} layers created".format(layers))
        self.out_channels = [256, 512, 1024, 2048] if layers >= 50 else [64, 128, 256, 512]
        self.backbone = timm.create_model(f'resnet{layers}', features_only=True, out_indices=out_indices, pretrained=True)
        self.layer4 = nn.Sequential(
            Bottleneck(self.out_channels[-2], self.out_channels[-3], downsample=nn.Sequential(
                nn.Conv2d(self.out_channels[-2], self.out_channels[-3]*4, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(self.out_channels[-3]*4),
            ), stride=2),
            Bottleneck(self.out_channels[-3]*4, self.out_channels[-3]),
            Bottleneck(self.out_channels[-3]*4, self.out_channels[-3]),
        )
    def forward(self, inputs):
        out = []
        for x in self.backbone(inputs):
            # print(x.shape)
            out.append(x)
        out.append(self.layer4(out[-1]))
        # print(out[-1].shape)
        return out
    

if __name__ == '__main__':
    model = Resnet()
    inputs = torch.randn(1, 3, 640, 640)
    out = model(inputs)

