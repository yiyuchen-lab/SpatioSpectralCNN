import torch
import torch.nn as nn
import torch.nn.functional as F

# class Usqueeze(nn.Module):
#     def forward(self, input):
#         return torch.unsqueeze(input,1)
#
# class Squeeze(nn.Module):
#     def forward(self, input):
#         return torch.squeeze(input)

# class SpatioSpectralCNN(nn.Sequential):

    # def __init__(self, in_channel, in_filter, in_class, drop_rate=0.5):
    #     super(SpatioSpectralCNN, self).__init__()
    #     self.add_module('unsqeeze',Usqueeze())
    #     self.add_module('conv_1', nn.Conv3d(1, 50, kernel_size=(3,3,3), bias=True))
    #     self.add_module('Relu_1', nn.ReLU())
    #     self.add_module('dropout1', nn.Dropout(p=drop_rate))
    #
    #     self.add_module('conv_2',  nn.Conv3d(50, 100, kernel_size=(in_channel-2,in_channel-2,in_filter-2),bias=True))
    #     self.add_module('Relu_2', nn.ReLU())
    #     self.add_module('dropout2', nn.Dropout(p=drop_rate))
    #
    #     self.add_module('Squeeze', Squeeze())
    #     self.add_module('dense', nn.Linear(100, in_class))
    #     self.add_module("softmax", nn.LogSoftmax(dim=1))


class SpatioSpectralCNN3D(nn.Module):

    def __init__(self, in_channel, in_filter, in_class, conv1_filter=50, conv2_filter=100, drop_rate=0.5):
        super(SpatioSpectralCNN3D, self).__init__()

        self.conv_1   = nn.Conv3d(1, conv1_filter, kernel_size=(3,3,3), bias=True)
        self.conv_2   = nn.Conv3d(conv1_filter, conv2_filter, kernel_size=(in_channel-2,in_channel-2,in_filter-2),bias=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.dense    = nn.Linear(conv2_filter, in_class)

    def forward(self, X):

        X = torch.unsqueeze(X,1)

        X = F.relu(self.conv_1(X))
        X = self.dropout(X)

        X = F.relu(self.conv_2(X))
        X = self.dropout(X)

        X = self.dense(torch.squeeze(X))

        return X


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, input):
        x = torch.cat(input, dim=1)

        return x

class filterband_Block(nn.Sequential):
    def __init__(self, n_cspComp,conv1_filter=10, conv2_filter=14, conv3_filter=18, drop_rate=0.5, zero_padding=4):
        super().__init__()

        dense_in_dim = (n_cspComp+(2*zero_padding)-6)*\
                       (n_cspComp+(2*zero_padding)-6)*\
                       conv3_filter

        self.add_module('conv1', nn.Conv2d(1, conv1_filter, kernel_size=(3, 3), padding=zero_padding))
        self.add_module('relu',nn.ReLU())
        self.add_module('dropout',nn.Dropout(p=drop_rate))

        self.add_module('conv2',nn.Conv2d(conv1_filter,conv2_filter,kernel_size=(3,3)))
        self.add_module('relu',nn.ReLU())
        self.add_module('dropout',nn.Dropout(p=drop_rate))

        self.add_module('conv3',nn.Conv2d(conv2_filter,conv3_filter,kernel_size=(3,3)))
        self.add_module('relu', nn.ReLU())
        self.add_module('dropout',nn.Dropout(p=drop_rate))

        self.add_module('flatten',nn.Flatten())
        self.add_module('dense',nn.Linear(dense_in_dim,256))



class SpatioSpectralCNN2D(nn.Module):

    def __init__(self, n_cspComp, n_filterbands, n_class, conv1_filter=10, conv2_filter=14, conv3_filter=18, drop_rate=0.5):
        super().__init__()

        filterband_blocks = []
        for i in range(n_filterbands):
            filterbandBlock = filterband_Block(n_cspComp,conv1_filter, conv2_filter, conv3_filter, drop_rate)
            filterband_blocks.append(filterbandBlock)

        self.n_filterbands              = n_filterbands
        self.filterband_blocks          = nn.ModuleList(filterband_blocks)
        self.concatenation_fusion_layer = nn.Linear(256*n_filterbands,n_class)
        self.cat                        = Concatenate() # make concatente module visible in torchsummary

    def forward(self,X):

        X = torch.unsqueeze(X, 1)
        # X.shape: (n_batch, n_model_filter, n_csp_comp, n_csp_comp, n_filterband)

        output_per_block = []
        for band_i, block in enumerate(self.filterband_blocks):
            output_per_block.append(block(X[:,:,:,:,band_i]))

        X_cat = self.cat(output_per_block)
        X_out = self.concatenation_fusion_layer(X_cat)

        return X_out

# from functions.SpatioSpectralCNN.torchsummary import summary_depth
# model = SpatioSpectralCNN2D(20, 15, 2)
# sm = summary_depth(model, (20, 20, 15), batch_size=64, device='cpu')