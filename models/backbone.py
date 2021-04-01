# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Backbone modules.
"""
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from torchvision.models.alexnet import model_urls

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                                             unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool, return_interm_layers: bool, dilation: bool):
        norm_layer = FrozenBatchNorm2d
        backbone = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation],
                                                     pretrained=is_main_process(), norm_layer=norm_layer)
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    # 是否需要训练backbone
    train_backbone = args.lr_backbone > 0
    # 是否需要记录backbone的每层的输出。
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 将backbone和位置编码集合在一个model。
    model = Joiner(backbone, position_embedding)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # 预训练模型下载链接，block模块函数，layers是每个layer的重复次数，pretrained是否使用预训练模型
    model = ResNet(block, layers, **kwargs)  # 模型定义
    if pretrained:  # 加载预训练模型
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


class ResNet(nn.Module):
    # renet类
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        # block基本模块类BasicBlock（18,34）或者Bottleneck（50,101,152），layes=[2, 2, 2, 2]是每个layer的重复次数， num_classes类别数
        super(ResNet, self).__init__()
        if norm_layer is None:  # norm_layer未指定，默认为nn.BatchNorm2d
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # 赋给self

        self.inplanes = 64  # 初始化的输出层数
        self.dilation = 1  # 膨胀率1
        if replace_stride_with_dilation is None:  # 替换步长用膨胀率，如果为None，设置默认值为[False, False, False]
            # each element in the tuple indicates if we should replace
            #  tuple中的每个元素都表示是否应该替换
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:  # 检查是否为None或者长度为3
            raise ValueError("replace_stride_with_dilation should be None ""or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups  # 组数为1
        self.base_width = width_per_group  # 每个组为64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)  # 第一个卷积层，（3,64,7,2,3）
        self.bn1 = norm_layer(self.inplanes)  # nn.BatchNorm2d层
        self.relu = nn.ReLU(inplace=True)  # relu层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0])  # 输出层数64，该层重复2次
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])  # 输出层数128，该层重复2次，步长为2，
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])  # 输出层数256，该层重复2次
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])  # 输出层数512，该层重复2次
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层，输出大小为（1,1）
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # fc层（expansion为1或4）

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # 函数BasicBlock类，输出层128，该层重复次数2，步长1，是否使用膨胀参数替代步长
        norm_layer = self._norm_layer  # nn.BatchNorm2d层
        downsample = None  # 下采样层初始化
        previous_dilation = self.dilation  # 先前的膨胀率
        if dilate:  # 用膨胀，更新膨胀率
            self.dilation *= stride  # 膨胀率= 1*步长
            stride = 1  # 步长为1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 步长不为1，或则self.inplances=64 不等于输出层数乘以基本类的扩张率1 ，则给下采样层赋值
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )  # 1x1的卷积层作为下采样层

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion  # 更新self.inplanes 值
        for _ in range(1, blocks):  # 重复次数f2的or循环
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # torch.Size([1, 3, 224, 224])
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # torch.Size([1, 64, 112, 112])
        x = self.maxpool(x)  # torch.Size([1, 64, 56, 56])

        x = self.layer1(x)  # torch.Size([1, 64, 56, 56])
        x = self.layer2(x)  # torch.Size([1, 128, 28, 28])
        x = self.layer3(x)  # torch.Size([1, 128, 14, 14])
        x = self.layer4(x)  # torch.Size([1, 512, 7, 7])

        x = self.avgpool(x)  # torch.Size([1, 512, 1, 1])
        x = torch.flatten(x, 1)  # torch.Size([1, 512])
        x = self.fc(x)  # torch.Size([1, 1000])

        return x

    def forward(self, x):
        return self._forward_impl(x)


class BasicBlock(nn.Module):
    # 基本的block类
    expansion = 1  # 扩张值，用于通道数倍增
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        # 输入通道数，输出通道数，步长，下采样层，组数，基本宽度，膨胀率，归一化层
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:  # 组数不等于1或则基本宽度不等于64，则报错，表明只支持组数为1，且base_width为64
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:  # 膨胀只能为1，
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # 3x3卷积层
        self.bn1 = norm_layer(planes)  # BN层
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)  # # 3x3卷积层
        self.bn2 = norm_layer(planes)  # BN层
        self.downsample = downsample  # 下采样层
        self.stride = stride

    def forward(self, x):
        identity = x  # 恒等值

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:  # 如果下采样层不为None
            identity = self.downsample(x)  # 下采样处理

        out += identity  # 残差链接 与原来x相加
        out = self.relu(out)  #

        return out


class Bottleneck(nn.Module):
    expansion = 4  # 扩张值，用于通道数倍增
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups  # 重新计算输出层
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)  # 1x1卷积层
        self.bn1 = norm_layer(width)  # BN层
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 3x3卷积层
        self.bn2 = norm_layer(width)  # BN层
        self.conv3 = conv1x1(width, planes * self.expansion)  # 1x1卷积层
        self.bn3 = norm_layer(planes * self.expansion)  # BN层
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 下采样层
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    # 3x3卷积带有padding
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    # 1x1卷积
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
