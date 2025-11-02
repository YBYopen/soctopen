
import sys
from utils.layers_spiketransformer import *
from utils.GAU import TA,SCA
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from timm.models.layers import to_2tuple, trunc_normal_, DropPath

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def upconv(in_planes, out_planes, stride=1):
    """transpose2d convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1)


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output
    
class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)
    
class MS_MLP(nn.Module):
    def __init__(
        self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv1d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.fc2_conv = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x
    
class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.head_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.q_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.k_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.v_conv = nn.Sequential(RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim))

        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")

        self.attn_lif = MultiStepLIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
        )

        self.proj_conv = nn.Sequential(
            RepConv(dim, dim, bias=False), nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W

        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)

        return x
    
class MS_Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)

        return x
    
class BasicBlock_MS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,ksize=9):
        super(BasicBlock_MS, self).__init__()
        if norm_layer is None:
            # norm_layer = tdGroupNorm
            norm_layer = GN
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.downsample = downsample
        
        self.spike = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=(1,ksize), padding=(0,ksize//2))
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=(ksize,1),padding=(ksize//2,0))
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = norm_layer(int(planes/16),planes)
        
        self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
        # self.convfn_s = tdLayer(self.convfn)
        
    def forward(self, x):
        T, B, _, _, _ = x.shape
        identity = x

        out = self.spike(x)

        out2 = self.conv3(out.flatten(0,1))
        out2 = self.conv4(out2)
        out2 = self.conv5(out2)
        _, C, H, W = out2.shape
        out2 = self.bn2(out2).reshape(T, B, C, H, W)
        out2 = self.spike2(out2)

        if self.downsample is not None:
            Td,Bd,Cd,Hd,Wd = x.shape
            identity = self.downsample(x.flatten(0,1)).reshape(Td,Bd,-1,H,W)
        out = out2 + identity
            
        return out
 
class BasicBlock_MS_UP(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, upsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,ksize=9):
        super(BasicBlock_MS_UP, self).__init__()
        if norm_layer is None:
            # norm_layer = tdGroupNorm
            norm_layer = GN
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        
        self.upsample = upsample
       
        self.spike = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.conv_s = nn.Conv2d(planes*2, planes, kernel_size=1, stride=1, padding=0)

        self.conv3 = nn.Conv2d(planes, planes, kernel_size=(1,ksize), padding=(0,ksize//2))
        self.conv4 = nn.Conv2d(planes, planes, kernel_size=(ksize,1), padding=(ksize//2,0))
        self.conv5 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = norm_layer(int(planes/16),planes)

        self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        
    def forward(self, x):
        T, B, _, _, _ = x[0].shape
        identity = x[0]
        if self.upsample is not None:
            identity = self.upsample(x[0].flatten(0,1))
            _, C0, H0, W0 = identity.shape
            identity = identity.reshape(T, B, C0, H0, W0)
            # if identity.shape[4] != x[1].shape[4]:
            #     identity = torch.cat((identity[:,:,:,:,:-1],x[1]), 2)
            # else :
            identity = torch.cat((identity,x[1]), 2)
            b, t, c, h, w = identity.shape
            identity = identity.flatten(0,1)
            identity = self.conv_s(identity)
            identity = identity.reshape(b,t, int(c/2), h, w)
        
        out = self.spike(identity)
        
        out2 = self.conv3(out.flatten(0,1))
        out2 = self.conv4(out2)
        out2 = self.conv5(out2)
        _, C, H, W = out2.shape
        out2 = self.bn2(out2).reshape(T, B, C, H, W)
        out2 = self.spike2(out2)

        out = out2 + identity
        
        return out, x[1]
class GAC(nn.Module):
    def __init__(self,T,out_channels):
        super().__init__()
        self.TA = TA(T = T)
        self.SCA = SCA(in_planes= out_channels,kerenel_size=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq, spikes):
        # x_seq B T inplanes H W
        # spikes B T inplanes H W

        TA = self.TA(x_seq)
        SCA = self.SCA(x_seq)
        out = self.sigmoid(TA * SCA)
        y_seq = out * spikes
        return y_seq


class ResNet(nn.Module):
    def __init__(self, block, layers,layersup,T, num_classes=10, using_GAC=True,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None,):
        super(ResNet, self).__init__()
        if norm_layer is None:
            # norm_layer = tdGroupNorm
            norm_layer = GN
        self._norm_layer = norm_layer
        self.GAC = using_GAC
        self.inplanes = 32
        self.inplanes_up = 256
        self.dilation = 1
        dpr = [
            x.item() for x in torch.linspace(0, 0.0, 8)
        ]  # stochastic depth decay rule
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn1 = norm_layer(int(self.inplanes/16),self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.strans_1 = MS_Block(dim = 64,num_heads=8,mlp_ratio=4,qkv_bias=False,qk_scale=None,drop=0.0,attn_drop=0.0,drop_path=dpr[0],norm_layer=nn.LayerNorm,sr_ratio=1)
        self.strans_2 = MS_Block(dim = 128,num_heads=8,mlp_ratio=4,qkv_bias=False,qk_scale=None,drop=0.0,attn_drop=0.0,drop_path=dpr[1],norm_layer=nn.LayerNorm,sr_ratio=1)
        self.strans_3 = MS_Block(dim = 256,num_heads=8,mlp_ratio=4,qkv_bias=False,qk_scale=None,drop=0.0,attn_drop=0.0,drop_path=dpr[2],norm_layer=nn.LayerNorm,sr_ratio=1)

        self.layer_up1 = self._make_layer_up(BasicBlock_MS_UP, 128, layersup[2], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer_up2 = self._make_layer_up(BasicBlock_MS_UP, 64, layersup[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer_up3 = self._make_layer_up(BasicBlock_MS_UP, 32, layersup[0], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        
        self.spike = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.spike2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        self.fnl = tdLayer(nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0))
        
        
        self.down1 = nn.Sequential(conv1x1(32, 64, 2),norm_layer(int(64 * block.expansion/16),64))
        self.down2 = nn.Sequential(conv1x1(64, 128, 2),norm_layer(int(128 * block.expansion/16),128))
        self.down3 = nn.Sequential(conv1x1(128, 256, 2),norm_layer(int(256 * block.expansion/16),256))
        
        self.conv1_cat = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                               bias=False),norm_layer(int(64/16),64))
        self.conv2_cat = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1,
                               bias=False),norm_layer(int(128/16),128))
        self.conv3_cat = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1,
                               bias=False),norm_layer(int(256/16),256))
        
        self.T = T
        if using_GAC==True:
            self.encoding = GAC(T=self.T ,out_channels =32)
        
        


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = tdLayer(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(int(planes * block.expansion/16),planes * block.expansion)
            # )
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(int(planes * block.expansion/16),planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _make_layer_up(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes_up != planes * block.expansion:
            # upsample = tdLayer(
            #     upconv(self.inplanes_up, planes * block.expansion, stride = 2),
            #     norm_layer(planes * block.expansion)
            # )
            upsample = nn.Sequential(upconv(self.inplanes_up, planes * block.expansion, stride = 2), nn.ReLU())

        layers = []
        layers.append(block(self.inplanes_up, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes_up = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes_up, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward_impl(self, x):
        # x B C H W
        '''encoding'''
        
        if self.GAC==True:
            B,T,C,H,W = x.shape
            x = self.conv1(x.flatten(0,1))
            x = self.bn1(x).reshape(B, T, -1, H, W)
            img = x
            x = self.spike(x)
            x1 = self.encoding(img,x)
        else:
            x1 = self.conv1_s(x)
        '''encoding'''
        x2 = self.layer1(x1)    
        B,T,C,H,W = x1.shape
        x2_t = self.strans_1(self.down1(x1.flatten(0,1)).reshape(B,T,-1,int(H/2),int(W/2)))
        
        x2 = torch.cat((x2,x2_t), 2)
        x2 = self.conv1_cat(x2.flatten(0,1)).reshape(B,T,-1,int(H/2),int(W/2))
        
        x3 = self.layer2(x2)          
        B,T,C,H,W = x2.shape
        x3_t = self.strans_2(self.down2(x2.flatten(0,1)).reshape(B,T,-1,int(H/2),int(W/2)))
        x3 = torch.cat((x3,x3_t), 2)
        x3 = self.conv2_cat(x3.flatten(0,1)).reshape(B,T,-1,int(H/2),int(W/2))
        
        x4 = self.layer3(x3)         
        B,T,C,H,W = x3.shape
        x4_t = self.strans_3(self.down3(x3.flatten(0,1)).reshape(B,T,-1,int(H/2),int(W/2)))
        x4 = torch.cat((x4,x4_t), 2)
        x4 = self.conv3_cat(x4.flatten(0,1)).reshape(B,T,-1,int(H/2),int(W/2))
        
        x5 = self.spike2(x4)           
        
        x3_up, _ = self.layer_up1((x5, x3))    
        x2_up, _ = self.layer_up2((x3_up, x2)) 
        x1_up, _ = self.layer_up3((x2_up, x1)) 
        
        x_final = self.fnl(x1_up) 
        x_final = torch.mean(x_final, dim=1)
        return x_final

    def forward(self, x):
        if x.dim() != 5:
            x_i = add_dimention(x, self.T)
        else:
            x_i = x
        x_o = self.forward_impl(x_i)
        return x_o


def _resnet(arch, block, layers,layersup,T, **kwargs):
    model = ResNet(block, layers,layersup,T, **kwargs)
    return model
def soctnet(T,pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock_MS, [3, 3, 2], [3, 3, 2],T,
                   **kwargs)
