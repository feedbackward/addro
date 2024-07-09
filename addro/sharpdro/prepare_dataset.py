'''Dataset preparation helper functions, based on the code of Huang et al. (2023).'''

# External modules.
import torchvision.transforms as transforms

# Internal modules.


###############################################################################


# Reference for this code:
# https://github.com/zhuohuangai/SharpDRO/blob/main/dataset/prepare_dataset.py


########################
###  Transformation  ###
########################

def get_transform(size, mean, std):
    '''
    This is the function defined used in the SharpDRO code.
    We just removed an extraneous argument from their original definition.
    https://github.com/zhuohuangai/SharpDRO/blob/main/dataset/prepare_dataset.py
    '''
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    test_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)])
    return train_transform, test_transform


###############################################################################
