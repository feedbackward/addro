'''Setup: generate data, set it in a loader, and return the loader.'''

# External modules.
import torch
from torch.utils.data import DataLoader, TensorDataset

# Internal modules.
from setup.cifar10 import CIFAR10_SharpDRO, CIFAR10_Drift
from setup.cifar100 import CIFAR100_SharpDRO, CIFAR100_Drift
from setup.imagenet30 import ImageNet30_SharpDRO, ImageNet30_Drift


###############################################################################


# Functions related to data loaders and generators.

def get_dataloader(dataset_name, dataset_paras, device, verbose=True):
    '''
    This function does the following three things.
    1. Generate data in numpy format.
    2. Convert that data into PyTorch (tensor) Dataset object.
    3. Initialize PyTorch loaders with this data, and return them.
    '''
    
    dp = dataset_paras
    
    # First get the data generators.
    data_tr, data_va, data_te = get_generator(dataset_name=dataset_name,
                                              dataset_paras=dp)
    
    # Generate data, map from ndarray to tensor (sharing memory), make Dataset.
    if dataset_name in ["cifar10-sharpdro", "cifar100-sharpdro", "imagenet30-sharpdro", "cifar10-drift", "cifar100-drift", "imagenet30-drift"]:
        
        X_tr, Y_tr, S_tr = map(torch.from_numpy, data_tr())
        X_va, Y_va, S_va = map(torch.from_numpy, data_va())
        X_te, Y_te, S_te = map(torch.from_numpy, data_te())
        
        if verbose:
            print("dtypes (tr): {}, {}, {}".format(X_tr.dtype, Y_tr.dtype, S_tr.dtype))
            print("dtypes (va): {}, {}, {}".format(X_va.dtype, Y_va.dtype, S_va.dtype))
            print("dtypes (te): {}, {}, {}".format(X_te.dtype, Y_te.dtype, S_te.dtype))
        
        Z_tr = TensorDataset(X_tr.to(device), Y_tr.to(device), S_tr.to(device))
        Z_va = TensorDataset(X_va.to(device), Y_va.to(device), S_va.to(device))
        Z_te = TensorDataset(X_te.to(device), Y_te.to(device), S_te.to(device))
        
    else:
        
        X_tr, Y_tr = map(torch.from_numpy, data_tr())
        X_va, Y_va = map(torch.from_numpy, data_va())
        X_te, Y_te = map(torch.from_numpy, data_te())
        
        if verbose:
            print("dtypes (tr): {}, {}".format(X_tr.dtype, Y_tr.dtype))
            print("dtypes (va): {}, {}".format(X_va.dtype, Y_va.dtype))
            print("dtypes (te): {}, {}".format(X_te.dtype, Y_te.dtype))
        
        Z_tr = TensorDataset(X_tr.to(device), Y_tr.to(device))
        Z_va = TensorDataset(X_va.to(device), Y_va.to(device))
        Z_te = TensorDataset(X_te.to(device), Y_te.to(device))
    
    # Prepare the loaders to be returned.
    dl_tr = DataLoader(Z_tr, batch_size=dp["bs_tr"], shuffle=True)
    eval_dl_tr = DataLoader(Z_tr, batch_size=len(X_tr), shuffle=False)
    eval_dl_va = DataLoader(Z_va, batch_size=len(X_va), shuffle=False)
    eval_dl_te = DataLoader(Z_te, batch_size=len(X_te), shuffle=False)

    return (dl_tr, eval_dl_tr, eval_dl_va, eval_dl_te)


def get_generator(dataset_name, dataset_paras):
    
    dp = dataset_paras
    rg = dp["rg"]
    
    # Prepare the data generators for the dataset specified.
    if dataset_name == "cifar10-sharpdro":
        X_tr, Y_tr, S_tr, X_va, Y_va, S_va, X_te, Y_te, S_te = CIFAR10_SharpDRO(
            rg=rg, tr_frac=dp["tr_frac"], corruption_type=dp["corruption_type"],
            num_severity_levels=dp["num_severity_levels"],
            severity_dist=dp["severity_dist"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr, S_tr)
        data_va = lambda : (X_va, Y_va, S_va)
        data_te = lambda : (X_te, Y_te, S_te)
    elif dataset_name == "cifar10-drift":
        X_tr, Y_tr, S_tr, X_va, Y_va, S_va, X_te, Y_te, S_te = CIFAR10_Drift(
            rg=rg, tr_frac=dp["tr_frac"], corruption_type=dp["corruption_type"],
            num_severity_levels=dp["num_severity_levels"],
            severity_dist=dp["severity_dist"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr, S_tr)
        data_va = lambda : (X_va, Y_va, S_va)
        data_te = lambda : (X_te, Y_te, S_te)
    elif dataset_name == "cifar100-sharpdro":
        X_tr, Y_tr, S_tr, X_va, Y_va, S_va, X_te, Y_te, S_te = CIFAR100_SharpDRO(
            rg=rg, tr_frac=dp["tr_frac"], corruption_type=dp["corruption_type"],
            num_severity_levels=dp["num_severity_levels"],
            severity_dist=dp["severity_dist"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr, S_tr)
        data_va = lambda : (X_va, Y_va, S_va)
        data_te = lambda : (X_te, Y_te, S_te)
    elif dataset_name == "cifar100-drift":
        X_tr, Y_tr, S_tr, X_va, Y_va, S_va, X_te, Y_te, S_te = CIFAR100_Drift(
            rg=rg, tr_frac=dp["tr_frac"], corruption_type=dp["corruption_type"],
            num_severity_levels=dp["num_severity_levels"],
            severity_dist=dp["severity_dist"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr, S_tr)
        data_va = lambda : (X_va, Y_va, S_va)
        data_te = lambda : (X_te, Y_te, S_te)
    elif dataset_name == "imagenet30-sharpdro":
        X_tr, Y_tr, S_tr, X_va, Y_va, S_va, X_te, Y_te, S_te = ImageNet30_SharpDRO(
            rg=rg, tr_frac=dp["tr_frac"], corruption_type=dp["corruption_type"],
            num_severity_levels=dp["num_severity_levels"],
            severity_dist=dp["severity_dist"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr, S_tr)
        data_va = lambda : (X_va, Y_va, S_va)
        data_te = lambda : (X_te, Y_te, S_te)
    elif dataset_name == "imagenet30-drift":
        X_tr, Y_tr, S_tr, X_va, Y_va, S_va, X_te, Y_te, S_te = ImageNet30_Drift(
            rg=rg, tr_frac=dp["tr_frac"], corruption_type=dp["corruption_type"],
            num_severity_levels=dp["num_severity_levels"],
            severity_dist=dp["severity_dist"]
        )() # note the call
        data_tr = lambda : (X_tr, Y_tr, S_tr)
        data_va = lambda : (X_va, Y_va, S_va)
        data_te = lambda : (X_te, Y_te, S_te)
    else:
        raise ValueError("Unrecognized dataset name.")
    
    return (data_tr, data_va, data_te)


###############################################################################
