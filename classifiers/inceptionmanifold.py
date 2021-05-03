import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np

__all__ = ['InceptionTime']

def mixup(x, shuffle, lam, i, j):
    if shuffle is not None and lam is not None and i == j:
        x = lam * x + (1 - lam) * x[shuffle]
    return x

class InceptionTime(nn.Module):

    def __init__(self,num_classes, input_dim=1,num_layers=6, hidden_dims=128,use_bias=False, use_residual= True, device=torch.device("cpu")):
        super(InceptionTime, self).__init__()
        self.modelname = f"InceptionTime_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"hidden-dims={hidden_dims}_num-layers={num_layers}"
        #self.inlinear = nn.Linear(input_dim, hidden_dims)
        self.num_layers = num_layers
        self.use_residual = use_residual
        #self.inception_modules_list = [InceptionModule(kernel_size=40, num_filters=hidden_dims,
                                                       #use_bias=use_bias, device=device) for _ in range(num_layers)]
        self.inception_modules_list = [InceptionModule(input_dim = input_dim, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device)]
        #for i in range(num_layers-1):
        for i in range(num_layers):
          
          self.inception_modules_list.append(InceptionModule(input_dim = hidden_dims, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device))
        #self.inception_modules = nn.Sequential(
            #*self.inception_modules_list
        #)
        self.shortcut_layer_list = [ShortcutLayer(input_dim,hidden_dims,stride = 1, bias = False)]
        for i in range(num_layers//3):
          self.shortcut_layer_list.append(ShortcutLayer(hidden_dims,hidden_dims,stride = 1, bias = False))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.outlinear = nn.Linear(hidden_dims,num_classes)

        self.to(device)

    def forward(self,x,mixup_alpha = 0.1):
        
        # step 1 selecting a random layer, selected the input layer
        if isinstance(x, list):
            x, shuffle, lam = x
        else:
            shuffle = None
            lam = None


        # Decide which layer to mixup
        j = np.random.randint(4)
        # N x T x D -> N x D x T
        x = x.transpose(1,2)
        #print('x_init ',x.shape)

        input_res = x
        # expand dimensions
        #x = self.inlinear(x.transpose(1, 2)).transpose(1, 2)
        for d in range(self.num_layers):
            x = self.inception_modules_list[d](x)

            x = mixup(x, shuffle, lam, 0, j)
            #print('x0 ',x.shape)

            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layer_list[d//3](input_res, x)
                input_res = x

            x = mixup(x, shuffle, lam, 1, j) 
            #print('x1 ', x.shape)

        x = self.avgpool(x).squeeze(2)
        x = mixup(x, shuffle, lam, 2, j)
        #print('x2 ', x.shape)
        x = self.outlinear(x)
        x = mixup(x, shuffle, lam, 3, j)
        print('x3', x.shape)
        logprobabilities = F.log_softmax(x, dim=-1)
        return logprobabilities
        #return x

class InceptionModule(nn.Module):
    def __init__(self,input_dim=32, kernel_size=40, num_filters= 32, residual=False, use_bias=False, device=torch.device("cpu")):
        super(InceptionModule, self).__init__()

        self.residual = residual

        self.bottleneck = nn.Conv1d(input_dim, num_filters , kernel_size = 1, stride=1, padding= 0,bias=use_bias)
        
        # the for loop gives 40, 20, and 10 convolutions
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        self.convolutions = [nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size+1, stride=1, bias= False, padding=kernel_size//2).to(device) for kernel_size in kernel_size_s] #padding is 1 instead of kernel_size//2
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(input_dim, num_filters,kernel_size=1, stride = 1,padding=0, bias=use_bias) 
        )

        self.bn_relu = nn.Sequential(
            nn.BatchNorm1d(num_filters*4),
            nn.ReLU()
        )


        self.to(device)


    def forward(self, input_tensor):
        # collapse feature dimension

        input_inception = self.bottleneck(input_tensor)
        features = [conv(input_inception) for conv in self.convolutions]
        features.append(self.pool_conv(input_tensor.contiguous()))
        features = torch.cat(features, dim=1) 
        features = self.bn_relu(features)
        
        return features

class ShortcutLayer(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bias):
        super(ShortcutLayer, self).__init__()
        self.sc = nn.Sequential(nn.Conv1d(in_channels=in_planes,
                                          out_channels=out_planes,
                                          kernel_size=1,
                                          stride=stride,
                                          bias=bias),
                                nn.BatchNorm1d(num_features=out_planes))
        self.relu = nn.ReLU()

    def forward(self, input_tensor, out_tensor):
        x = out_tensor + self.sc(input_tensor)
        x = self.relu(x)

        return x
