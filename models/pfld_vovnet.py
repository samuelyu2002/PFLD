import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp * expand_ratio,
                inp * expand_ratio,
                3,
                stride,
                1,
                groups=inp * expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),

        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv3x3(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


def conv1x1(in_channels, out_channels, module_name, postfix,
            stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat
        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stride_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if stride_per_block == 2:
            self.add_module('Pooling',
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = 'OSA{stage_num}_1'
        self.add_module(module_name,
            _OSA_module(in_ch,
                        stage_ch,
                        concat_ch,
                        layer_per_block,
                        module_name))
        for i in range(block_per_stage-1):
            module_name = 'OSA{stage_num}_{i+2}'
            self.add_module(module_name,
                _OSA_module(concat_ch,
                            stage_ch,
                            concat_ch,
                            layer_per_block,
                            module_name,
                            identity=True))

class VoVNet(nn.Module):
    def __init__(self, 
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stride_per_block,
                 num_landmarks=68):
        super(VoVNet, self).__init__()

        # Stem module
        stem = conv3x3(3, 32, 'stem', '1', stride=2)

        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [32]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(5): #num_stages
            name = 'stage{}'.format(i+2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block[i],
                                       stride_per_block[i],
                                       i+2))

        self.conv6_1 = InvertedResidual(384, 16, 1, False, 2)  # [16, 14, 14]
        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.bn8 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(4832, num_landmarks*2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        out1 = None
        for name in self.stage_names:
            x = getattr(self, name)(x)

        x = self.conv6_1(x)
        x1 = x.view(x.size(0), -1)
        x = self.conv7(x)
        x2 = x.view(x.size(0), -1)
        x = self.relu(self.conv8(x))
        x3 =  x.view(x.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(nn.Dropout(p=0.1, inplace=True)(multi_scale))

        return landmarks

def _vovnet(arch,
            config_stage_ch,
            config_concat_ch,
            block_per_stage,
            layer_per_block,
            stride_per_block,
            pretrained,
            progress,
            num_landmarks,
            **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch,
                   block_per_stage, layer_per_block,stride_per_block,num_landmarks,
                   **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vovnet_pfld(pretrained=False, progress=True, num_landmarks=68, **kwargs):

    return _vovnet('vovnet_pfld', [32, 64, 96, 128,160], [64, 96, 144, 192, 384],[1,1,1,1,1], [3,4,5,5,5],[1,2,1,2,1], pretrained, progress, num_landmarks,**kwargs)

# class AuxiliaryVoVNet(nn.Module):
#     def __init__(self):
#         super(AuxiliaryVoVNet, self).__init__()
#         self.conv1 = conv_bn(144, 128, 3, 2)
#         self.conv2 = conv_bn(128, 128, 3, 1)
#         self.conv3 = conv_bn(128, 32, 3, 2)
#         self.conv4 = conv_bn(32, 128, 7, 1)
#         self.max_pool1 = nn.MaxPool2d(3)
#         self.fc1 = nn.Linear(128, 32)
#         self.fc2 = nn.Linear(32, 3)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.max_pool1(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)

#         return x