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

class InceptionTime_Hidden(nn.Module):

    def __init__(self,num_classes, input_dim=13,num_layers=6, hidden_dims=128,use_bias=False, use_residual= True,mixup_hidden = False, device=torch.device("cpu")):
        super(InceptionTime_Hidden, self).__init__()
        self.modelname = f"InceptionTime_input-dim={input_dim}_num-classes={num_classes}_" \
                         f"hidden-dims={hidden_dims}_num-layers={num_layers}"
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.mixup_hidden = mixup_hidden
        self.num_classes = num_classes

        self.inception_modules_list = nn.ModuleList([InceptionModule(input_dim = input_dim,kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device)])

        for i in range(num_layers-1):
          self.inception_modules_list.append(InceptionModule(input_dim = hidden_dims, kernel_size=40, num_filters=hidden_dims//4,
                                                       use_bias=use_bias, device=device))
        
        
        self.shortcut_layer_list = nn.ModuleList([ShortcutLayer(input_dim,hidden_dims,stride = 1, bias = False)])
        for i in range(num_layers//3):
          self.shortcut_layer_list.append(ShortcutLayer(hidden_dims,hidden_dims,stride = 1, bias = False))
        
       
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.outlinear = nn.Linear(hidden_dims,num_classes)

        self.to(device)

    #def forward(self,x,mixup_alpha = 0.1):
    def forward(self, x, lam=None, target=None):  

        def mixup_process(out, target_reweighted, lam):
            # target_reweighted is one-hot vector
            # target is the target class.

            # shuffle indices of mini-batch
            indices = np.random.permutation(out.size(0))

            out = out*lam.expand_as(out) + out[indices]*(1-lam.expand_as(out))
            target_shuffled_onehot = target_reweighted[indices]
            target_reweighted = target_reweighted * lam.expand_as(target_reweighted) + target_shuffled_onehot * (1 - lam.expand_as(target_reweighted))
            return out, target_reweighted

        def to_one_hot(inp, num_classes):
            y_onehot = torch.FloatTensor(inp.size(0), num_classes)
            y_onehot.zero_()
            y_onehot.scatter_(1, inp.unsqueeze(1).cpu(), 1)
            return y_onehot.to("cpu")

        if self.mixup_hidden:
            layer_mix = np.random.randint(0,self.num_layers)
        else:
            layer_mix = 0

        #out = x


        
        #if isinstance(x, list):
            #x, shuffle, lam = x
        #else:
            #shuffle = None
            #lam = None


        # Decide which layer to mixup
        #j = np.random.randint(self.num_layers)
        
        
        # N x T x D -> N x D x T
        x = x.transpose(1,2)
        input_res = x
        
        for d in range(self.num_layers):
            x = self.inception_modules_list[d](x)

            if lam is not None:
              target_reweighted = to_one_hot(target, self.num_classes)

            if lam is not None and self.mixup_hidden and layer_mix == 0:
              x, target_reweighted = mixup_process(x, target_reweighted, lam)
              #x = mixup(x, shuffle, lam, d, j)

            if self.use_residual and d % 3 == 2:
                x = self.shortcut_layer_list[d//3](input_res, x)
                input_res = x
        x = self.avgpool(x).squeeze(2)
        x = self.outlinear(x)
        logprobabilities = F.log_softmax(x, dim=-1)

        if lam is None:
            return logprobabilities
        else:
            return logprobabilities, target_reweighted
        #return logprobabilities, target_reweighted

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
