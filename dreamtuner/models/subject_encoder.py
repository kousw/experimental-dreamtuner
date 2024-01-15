from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Callable, Type
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.resnet import resnet50

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward
from diffusers.models.resnet import ResnetBlock2D

from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel

from einops import rearrange, repeat
import math
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision import models

class MapperLocal(nn.Module):
    def __init__(self,
         input_dim: int,
         output_dim: int,
    ):
        super(MapperLocal, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, 1024),
                                                              nn.LayerNorm(1024),
                                                              nn.LeakyReLU(),
                                                              nn.Linear(1024, output_dim)))
    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:])
            hidden_states += (hidden_state.unsqueeze(0),)
        hidden_states = torch.cat(hidden_states, dim=0).mean(dim=0)
        return hidden_states
    
class ResNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        layers: List[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        
        self.in_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels
        

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(models.resnet.BasicBlock, 64, layers[0])
        self.layer2 = self._make_layer(models.resnet.BasicBlock, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(models.resnet.BasicBlock, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(models.resnet.BasicBlock, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * models.resnet.BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, models.resnet.Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, models.resnet.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                models.resnet.conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        print("resnet.conv1", x.shape)
        x = self.bn1(x)
        print("resnet.bn1", x.shape)
        x = self.relu(x)
        x = self.maxpool(x)
        print("resnet.maxpool", x.shape)

        x = self.layer1(x)
        print("resnet.layer1", x.shape)
        x = self.layer2(x)
        print("resnet.layer2", x.shape)
        x = self.layer3(x)
        print("resnet.layer3", x.shape)
        x = self.layer4(x)
        print("resnet.layer4", x.shape)

        x = self.avgpool(x)
        print ("resnet.avgpool", x.shape)
        x = torch.flatten(x, 1)
        print ("resnet.flatten", x.shape)
        x = self.fc(x)
        print ("resnet.fc", x.shape)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)    
    
class MapperResnet(nn.Module):
    def __init__(self,
         input_dim: int,
         output_dim: int,
         dropout: float = 0.2,
         embs_layer_num: int = 5,
    ):
        super(MapperResnet, self).__init__()
        
        self.embs_layer_num = embs_layer_num

        # for i in range(self.embs_layer_num):
        #     setattr(self, f'mapping_{i}', ResnetBlock2D(in_channels=5, out_channels=1, dropout=dropout, groups=1, temb_channels=None))

        #     setattr(self, f'mapping_patch_{i}', ResnetBlock2D(in_channels=5, out_channels=1, dropout=dropout, groups=1, temb_channels=None))
            
        # self.proj =  nn.Sequential(nn.Linear(input_dim, 1024),
        #                                     nn.LayerNorm(1024),
        #                                     nn.LeakyReLU(),
        #                                     nn.Linear(1024, output_dim))
        
        self.resnet_down1 = ResnetBlock2D(in_channels=5, out_channels=20, dropout=dropout, groups=5, down=True, temb_channels=None)   
        self.resnet_down2 = ResnetBlock2D(in_channels=20, out_channels=80, dropout=dropout, groups=5, down=True, temb_channels=None)   
        self.resnet_down3 = ResnetBlock2D(in_channels=80, out_channels=320, dropout=dropout, groups=5, down=True,  temb_channels=None)
        
        self.resnet_up1 = ResnetBlock2D(in_channels=320, out_channels=80, dropout=dropout, groups=5, up=True, temb_channels=None)   
        self.resnet_up2 = ResnetBlock2D(in_channels=80, out_channels=20, dropout=dropout, groups=5, up=True, temb_channels=None)   
        self.resnet_up3 = ResnetBlock2D(in_channels=20, out_channels=5, dropout=dropout, groups=5, up=True,  temb_channels=None)
        
        self.conv = nn.Conv2d(5, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()
        self.proj =  nn.Linear(input_dim, output_dim, bias=False)        
        self.out_norm =  nn.LayerNorm(output_dim)

    
    def forward(self, embs):
                
        batch_size = embs.shape[0]      
        
        x = x1 = self.resnet_down1(embs, temb=None)
        x = x2 = self.resnet_down2(x, temb=None)
        x = x3 = self.resnet_down3(x, temb=None)
        
        x = self.resnet_up1(x, temb=None)
        x = x + x2
        x = self.resnet_up2(x, temb=None)
        x = x + x1
        x = self.resnet_up3(x, temb=None)
        x = x + embs[:, :, 1:]
        # print(hidden_state.shape)
        x = self.conv(x)
        x = self.relu(x)
        x = self.proj(x)
        x = self.out_norm(x)
        
        x = rearrange(x, 'b c h w -> (b c) h w')
        
        return x


# from https://arxiv.org/abs/2312.16272
class MapperSSR(nn.Module):
    def __init__(self,
         input_dim: int,
         output_dim: int,
         dropout: float = 0.2,
         embs_layer_num: int = 5,
    ):
        super(MapperSSR, self).__init__()
        
        self.embs_layer_num = embs_layer_num

        self.in_norm = nn.LayerNorm(input_dim)

        for i in range(self.embs_layer_num):
            setattr(self, f'w_{i}', nn.Linear(input_dim, input_dim))
                 
            
        self.relu = nn.ReLU()
        self.proj =  nn.Linear(input_dim, output_dim, bias=True)        
        self.out_norm =  nn.LayerNorm(output_dim)

    
    def forward(self, embs):
        
        # enmb.shape = (batch_size, layer_num, token_num, feature_dim)
        batch_size = embs.shape[0]      
        
        x = embs
        x = self.in_norm(x)
        
        # chunk with layer_num
        x = x.chunk(self.embs_layer_num, dim=1)
        
        assert len(x) == self.embs_layer_num        
        
        outputs = []
        for i, x_ in enumerate(x):
            output = getattr(self, f'w_{i}')(x_)
            outputs.append(output)
            
        x = torch.cat(outputs, dim=2)

        x = self.relu(x)
        x = self.proj(x)
        x = self.out_norm(x)
        
        x = rearrange(x, 'b c h w -> (b c) h w')
        
        return x

class SubjectEncoder(ModelMixin, ConfigMixin):
    def __init__(
        self,
        input_dim = 1024,
        output_dim = 768,
        subject_encoder_mode = 'ssr'
    ):
        super().__init__()
        
        self.subject_encoder_mode = subject_encoder_mode
        
        self.image_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        if self.subject_encoder_mode == 'ssr':
            self.mapper = MapperSSR(input_dim=input_dim, output_dim=output_dim)
        elif self.subject_encoder_mode == 'resnet':
            self.mapper = MapperResnet(input_dim=input_dim, output_dim=output_dim)
        else:
            raise NotImplementedError()
        
    def encode_image(self, image):
        assert image.shape[-2] == 224 and image.shape[-1] == 224
        image_features = self.image_encoder(image, output_hidden_states=True)

        image_embeddings = [image_features[0], image_features[2][4], image_features[2][8], image_features[2][12],
                        image_features[2][16]]
        
        image_embeddings = torch.stack(image_embeddings, dim=1)
        
        return image_embeddings
        
    def forward(self, image_embeddings):        
        
        hidden_states = self.mapper(image_embeddings)
        
        return hidden_states
