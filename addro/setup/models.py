'''Setup: model design and functions for getting specific models.'''

# External modules.
from math import floor as mathfloor
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

# Internal modules.
from sharpdro.resnet import WideResNet


###############################################################################


class CNN_Prototyping(nn.Module):
    '''
    Simple CNN for prototyping. Original is at the link below.
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    We modified it to be able to handle images of any size.
    '''

    def size_change(self, size_in, padding, dilation, ker_size, stride):
        '''
        Applies for both Conv2d and MaxPool2d. See docs below for reference.
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        '''
        size_out = mathfloor((size_in + 2*padding - dilation*(ker_size-1) - 1)/stride + 1)
        return size_out
    
    def __init__(self, num_classes, img_height, img_width):
        super().__init__()
        # Some key parameters of the model.
        conv1_ch_in = 3
        conv1_ch_out = 6
        conv1_ker_size = 5
        pool_ker_size = 2
        pool_stride = 2
        conv2_ch_in = 6
        conv2_ch_out = 16
        conv2_ker_size = 5
        # Module defns before fully connected layers.
        self.conv1 = nn.Conv2d(conv1_ch_in, conv1_ch_out, conv1_ker_size)
        self.pool = nn.MaxPool2d(pool_ker_size, pool_stride)
        self.conv2 = nn.Conv2d(conv2_ch_in, conv2_ch_out, conv2_ker_size)
        # Compute the proper sizes for first fully connected layer.
        ## First apply conv1.
        modified_height = self.size_change(size_in=img_height, padding=0, dilation=1,
                                           ker_size=conv1_ker_size, stride=1)
        modified_width = self.size_change(size_in=img_width, padding=0, dilation=1,
                                           ker_size=conv1_ker_size, stride=1)
        ## Then apply pool.
        modified_height = self.size_change(size_in=modified_height, padding=0, dilation=1,
                                           ker_size=pool_ker_size, stride=pool_stride)
        modified_width = self.size_change(size_in=modified_width, padding=0, dilation=1,
                                           ker_size=pool_ker_size, stride=pool_stride)
        ## Then apply conv2.
        modified_height = self.size_change(size_in=modified_height, padding=0, dilation=1,
                                           ker_size=conv2_ker_size, stride=1)
        modified_width = self.size_change(size_in=modified_width, padding=0, dilation=1,
                                           ker_size=conv2_ker_size, stride=1)
        ## Finally apply pool once more.
        modified_height = self.size_change(size_in=modified_height, padding=0, dilation=1,
                                           ker_size=pool_ker_size, stride=pool_stride)
        modified_width = self.size_change(size_in=modified_width, padding=0, dilation=1,
                                           ker_size=pool_ker_size, stride=pool_stride)
        flattened_dim = conv2_ch_out * modified_height * modified_width
        self.fc1 = nn.Linear(flattened_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class IshidaMLP_synthetic(nn.Module):
    '''
    The multi-layer perceptron used for synthetic data tests by
    Ishida et al. (2020).

    NOTE: they call this a "five hidden layer" MLP.
          Follows their public code exactly.
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        width = 500
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, output_dim, bias=True)
        )
        return None

    def forward(self, x):
        return self.linear_relu_stack(x)


class IshidaMLP_benchmark(nn.Module):
    '''
    The multi-layer perceptron used for benchmark data tests
    in Ishida et al. (2020).

    NOTE: they call this a "two hidden layer" MLP.
          No public code, so we just infer based on
          prev example in IshidaMLP_synthetic.
          The BatchNorm1d inclusion is also inferred.
    '''
    def __init__(self, input_dim, output_dim):
        super().__init__()
        width = 1000
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Linear(width, output_dim, bias=True)
        )
        return None

    def forward(self, x):
        return self.linear_relu_stack(x)


def get_model(model_name, model_paras):
    
    mp = model_paras
    _pt = mp["pre_trained"]
    pt_weights = _pt if _pt != "None" else None
    
    if model_name == "IshidaMLP_synthetic":
        return IshidaMLP_synthetic(input_dim=mp["dimension"],
                                   output_dim=mp["num_classes"])
    elif model_name == "IshidaMLP_benchmark":
        return IshidaMLP_benchmark(input_dim=mp["dimension"],
                                   output_dim=mp["num_classes"])
    elif model_name == "WideResNet_Huang":
        model = WideResNet(width=2, classes=mp["num_classes"])
        return model
    elif model_name == "CNN_Prototyping":
        model = CNN_Prototyping(num_classes=mp["num_classes"],
                                img_height=mp["height"],
                                img_width=mp["width"])
        return model
    elif model_name == "ResNet18":
        if pt_weights == None:
            return torchvision.models.resnet18(num_classes=mp["num_classes"])
        else:
            model = torchvision.models.resnet18(weights=pt_weights)
            print("Final layer (pre-trained model):", model.fc)
            model.fc = nn.Linear(512, mp["num_classes"])
            print("Final layer (adjusted to current dataset):", model.fc)
            return model
    elif model_name == "ResNet34":
        if pt_weights == None:
            return torchvision.models.resnet34(num_classes=mp["num_classes"])
        else:
            model = torchvision.models.resnet34(weights=pt_weights)
            print("Final layer (pre-trained model):", model.fc)
            model.fc = nn.Linear(512, mp["num_classes"])
            print("Final layer (adjusted to current dataset):", model.fc)
            return model
    else:
        raise ValueError("Unrecognized model name.")


###############################################################################
