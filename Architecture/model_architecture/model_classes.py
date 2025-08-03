import math,os,random,cv2,numpy as np,torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, groups = 1, activation = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias = False,groups = groups)
        self.bn = nn.BatchNorm2d(out_channels,eps = 0.001,momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()
    
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# 2.1 Bottleneck: stack of 2 Conv with shortcut connection (True/Fals)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,shortcut = True):
        super().__init__()
        self.conv1 = Conv(in_channels,out_channels,kernel_size=3,stride = 1,padding=1)
        self.conv2 = Conv(out_channels,out_channels,kernel_size=3,stride = 1,padding=1)
        self.shortcut = shortcut
    
    def forward(self,x):
        x_in = x # for residual connection
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x=x+x_in
        return x

# 2.2 C2f: Conv + Bottleneck*n + Conv

class C2f(nn.Module):
    def __init__(self,in_channels,out_channels,num_bottlenecks,shortcut = True):
        super().__init__()
        self.mid_channels = out_channels//2
        self.num_bottlenecks = num_bottlenecks
        self.conv1 = Conv(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.m = nn.ModuleList([Bottleneck(self.mid_channels,self.mid_channels) for _ in range(num_bottlenecks)])
        self.conv2 = Conv((num_bottlenecks+2)*out_channels//2, out_channels, kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.conv1(x)

        x1,x2 = x[:,:x.shape[1]//2,:,:],x[:,x.shape[1]//2:,:,:]
        outputs=[x1,x2] # x1 is fed to the bottlenecks
        for i in range( self.num_bottlenecks):
            x1 = self.m[i](x1) # [bs,0.5c_out,w,h]
            outputs.insert(0,x1)
        
        outputs=torch.cat(outputs,dim=1) #[bs,0.5c_out(num_bottlenecks+2),w,h]
        out = self.conv2(outputs)
        return out

# sanity check
# c2f = C2f(in_channels=64,out_channels=128,num_bottlenecks=2)
# print(f"{sum(p.numel() for p in c2f.parameters())/1e6} million parameters")

# dummy_input = torch.rand((1,64,244,244))
# dummy_output = c2f(dummy_input)
# print("Output shape: ",dummy_output.shape)


class SPPF(nn.Module):
    def __init__(self,in_channels,out_channels,kernal_size=5):
        # kernal_size = sie of maxpool
        super().__init__()
        hidden_channels = in_channels//2
        self.conv1 = Conv(in_channels,hidden_channels,kernel_size=1,stride=1,padding=0)
        # concatenate    outputs of maxpool and feed to conv2
        self.conv2 = Conv(4*hidden_channels,out_channels,kernel_size=1,stride=1,padding=0)
        # maxpool is applied at 3 different scales
        self.m = nn.MaxPool2d(kernel_size=kernal_size, stride = 1, padding=kernal_size//2,dilation=1,ceil_mode=False)

    def forward(self,x):
        x = self.conv1(x)

        # apply maxpooling at different scales
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        # concatenate
        y=torch.cat([x,y1,y2,y3],dim=1)

        #final conv
        y = self.conv2(y)   
        return y

# sppf = SPPF(in_channels=128,out_channels=512)
# print(f"{sum(p.numel() for p in sppf.parameters())/1e6} million parameters")

# dummy_input = torch.rand((1,128,244,244))
# dummy_output = sppf(dummy_input)
# print("Output shape: ",dummy_output.shape)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self,x):
        return nn.functional.interpolate(x,scale_factor=self.scale_factor,mode=self.mode)
    
class Concat(nn.Module):
        def __init__(self, dim = 1):
            super().__init__()
            self.dim = dim
        
        def forward(self, x):
            return torch.cat(x,dim = self.dim)
        

# DFL

class DFL(nn.Module):
    def __init__(self,ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch,out_channels=1,kernel_size=1,bias=False).requires_grad_(False)

        # initialise conv with [0,...,ch-1]
        x = torch.arange(ch,dtype=torch.float).view(1,ch,1,1)
        self.conv.weight.data[:] = torch.nn.Parameter(x) # DFL only has ch parameters
    def forward(self,x):
        # x must have num_channels = 4*ch: x=[bs, 4*ch, c]
        b, c, a = x.shape                           # c = 4*ch
        x = x.view(b, 4, self.ch, a).transpose(1,2) # [bs, ch, 4,  a]

        # take softmax on channel dimension to get distribution probabilities
        x = x.softmax(1)            # [b, ch, 4, a]
        x = self.conv(x)            # [b, 1, 4, a]
        return x.view(b, 4, a)      # [b, 4, a]

# dummy_input = torch.rand((1,64,128))
# dfl = DFL()
# print(f"{sum(p.numel() for p in dfl.parameters())} Parameters")

# dummy_output = dfl(dummy_input)
# print(dummy_output.shape)

# print(dfl)


class Head(nn.Module):
    def __init__(self, version='n', ch=16, num_classes=80):
        super().__init__()
        self.ch = ch                        # dfl channels
        self.coordinates = self.ch*4        # number of bounding boxes
        self.nc = num_classes               # 80 for COCO
        self.no = self.coordinates+self.nc  # number of outputs for anchor box
        self.stride = torch.zeros(3)        # strides completed during build
        d,w,r = yolo_params(version=version)

        # for bounding boxes
        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w),self.coordinates,kernel_size=3,stride=1,padding=1),
                Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w*r),self.coordinates,kernel_size=3,stride=1,padding=1),
                Conv(self.coordinates,self.coordinates,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(self.coordinates,self.coordinates,kernel_size=1,stride=1)
            ),
        ])

        # for classification
        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w),self.nc,kernel_size=3,stride=1,padding=1),
                Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w),self.nc,kernel_size=3,stride=1,padding=1),
                Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)
            ),
            nn.Sequential(
                Conv(int(512*w*r),self.nc,kernel_size=3,stride=1,padding=1),
                Conv(self.nc,self.nc,kernel_size=3,stride=1,padding=1),
                nn.Conv2d(self.nc,self.nc,kernel_size=1,stride=1)
            ),
        ])
        
        # dfl
        self.dfl = DFL()
    
    def forward(self, x):
        # x = output of Neck = list of 3 tensors with different resolution and different channel dim
        # x[0] = [bs, ch0, w0, h0], x[1] = [bs, ch1, w1, h1], x[2] = [bs, ch2, w2, h2]

        for i in range(len(self.box)):          # detection head i
            box = self.box[i](x[i])             # [bs, num_coordinates, w, h]
            cls = self.cls[i](x[i])             # [bs, num_classes, w ,h]
            x[i] = torch.cat([box,cls], dim=1)  # [bs, num_coordinates+num_classes, w, h]
        
        # in training, no dfl output
        if self.training:
            return x                            # [3, bs, num_coordinates+num_classes, w, h]
        
        # in inference time, dfl produces refined bounding box coordinates 
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))

        # concatenated prediction from all dtection layers
        x = torch.cat([i.view(x[0].shape, self.no, -1) for i in x], dim = 2) # [bs, 4*self.ch + self.nc, sum(h[i]w[i])]

        # split out predictions for box and cls
        #       box = [bs, 4*self.ch, sum_i(h[i]w[i])]
        #       cls = [bs, self.nc, sum_i(h[i]w[i])]
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a,b = self.dfl(box).chunk(2,1) # a=b=[bs, 2*self.ch, sum_i(h[i]w[i])]
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a+b)/2,b-a), dim=1)
        
        return torch.cat(tensors=(box*strides, cls.sigmoid()), dim=1)
    
    @staticmethod
    def make_anchors(x, strides, offset=0.5):
        # x = list of features maps: x = [x(0),....,x[N-1], in our case N = num_detection_heads = 3]
        #                            each giing shape [bs, ch, w, h]
        #     each feature map x[i] gives output[i] = w*h anchor coordinates + w*h strides values

        # strides = list of strides values indicariong how much 
        #           the spatial resolution of the feature map is reduced compared to the original image

        assert x is not None
        anchor_tensor, stride_tensor = [],[]
        dtype, device = x[0].type,x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset       # x coordinates of anchor centers
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset       # y coordinates of anchor centers
            sx, sy = torch.meshgrid(sy, sx)                                     # all anchor centers
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


def yolo_params(version):
    if version == 'n':
        return 1/3,1/4,2.0
    elif version == 's':
        return 1/3,1/2,2.0
    elif version == 'm':
        return 2/3,3/4,1.5
    elif version == 'l':
        return 1.0,1.0,1.0
    elif version == 'x':
        return 1.0,1.25,1.0
        