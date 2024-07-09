'''Runs the SharpDRO-style experiments and records settings using mlflow.'''

# External modules.
import copy
import mlflow
import numpy as np
import os

# Internal modules.
from setup.utils import get_seeds


###############################################################################


# Experiment parameter settings.

num_trials = 3 # follows Huang et al. (2023); see section B.2.
base_seed = 13611284172945
random_seeds = get_seeds(base_seed=base_seed, num=num_trials)

## Settings which are common across all runs.
paras_common = {
    "base_gpu_id": 0,
    "corruption_type": "gaussian_noise", # "gaussian_noise" or "jpeg_compression"
    "epochs": 200, # follows Huang et al. (2023); see section B.2.
    "force_cpu": "no",
    "force_one_gpu": "yes",
    "loss": "CrossEntropy",
    "num_severity_levels": 5, # typically set to 5, but can be less.
    "pre_trained": "None", # accepts either "DEFAULT" or "None"
    "gradnorm": "no", # accepts either "yes" or "no"
    "saving_freq": 0, # save after this many epochs; if 0, no saving.
    "scheduler": "sharpdro", # either "none" or "sharpdro"
    "severity_dist": "poisson" # choose from ["poisson", "uniform", "revpoisson"]
}

## Initial settings for parameters that may change depending on method.
paras_mth_defaults = {
    "adaptive": "no", # takes either "no" or "yes".
    "eta": 1.0,
    "flood_level": 0.0,
    "momentum": 0.9, # follows Huang et al. (2023); see section B.2.
    "optimizer": "SGD",
    "optimizer_base": "none",
    "prob_update_factor": 0.01, # this is what Huang et al. call "step_size" in their code.
    "quantile_level": 0.0,
    "radius": 0.05,
    "sigma": 1.0,
    "softad_level": 0.0,
    "step_size": 0.03, # this is what Huang et al. call "lr" in their code; see also appendix B.2.
    "tilt": 0.0,
    "weight_decay": 0.0005 # follows Huang et al. (2023); see section B.2.
}

## List of datasets to be covered in current experiment.
## Choose any subset of ["cifar10-sharpdro", "cifar100-sharpdro", "imagenet30-sharpdro", "cifar10-drift", "cifar100-drift", "imagenet30-drift"].
dataset_names = ["cifar10-sharpdro", "cifar100-sharpdro"]

## Dataset-specific number of classes.
num_classes_dict = {
    "cifar10-sharpdro": 10,
    "cifar100-sharpdro": 100,
    "imagenet30-sharpdro": 30,
    "cifar10-drift": 10,
    "cifar100-drift": 100,
    "imagenet30-drift": 30
}

## Dataset-dependent tr/va fractions, all following Huang et al. (2023).
tr_frac_dict = {
    "cifar10-sharpdro": 0.799,
    "cifar100-sharpdro": 0.799,
    "imagenet30-sharpdro": 0.7661538461538462,
    "cifar10-drift": 0.799,
    "cifar100-drift": 0.799,
    "imagenet30-drift": 0.7661538461538462
}

## Model specifics (depends on the dataset).
## Choose from either "CNN_Prototyping" or "WideResNet_Huang"
model_dict = {
    "cifar10-sharpdro": "CNN_Prototyping",
    "cifar100-sharpdro": "CNN_Prototyping",
    "imagenet30-sharpdro": "CNN_Prototyping",
    "cifar10-drift": "CNN_Prototyping",
    "cifar100-drift": "CNN_Prototyping",
    "imagenet30-drift": "CNN_Prototyping"
}

## Skip singles? (depends on the model, which depends on dataset)
skip_singles_dict = {
    "cifar10-sharpdro": "yes",
    "cifar100-sharpdro": "yes",
    "imagenet30-sharpdro": "yes",
    "cifar10-drift": "yes",
    "cifar100-drift": "yes",
    "imagenet30-drift": "yes"
}

## Batch size specifics (depends on dataset).
## Follows SharpDRO repo:
## https://github.com/zhuohuangai/SharpDRO/blob/main/main.py
bs_tr_dict = {
    "cifar10-sharpdro": 100,
    "cifar100-sharpdro": 100,
    "imagenet30-sharpdro": 100,
    "cifar10-drift": 100,
    "cifar100-drift": 100,
    "imagenet30-drift": 100
}

## Input dimensions (depends on dataset; not all models use this).
dimension_dict = {
    "cifar10-sharpdro": 3*32*32,
    "cifar100-sharpdro": 3*32*32,
    "imagenet30-sharpdro": 3*64*64,
    "cifar10-drift": 3*32*32,
    "cifar100-drift": 3*32*32,
    "imagenet30-drift": 3*64*64
}

## Height (depends on dataset; not all models use this).
height_dict = {
    "cifar10-sharpdro": 32,
    "cifar100-sharpdro": 32,
    "imagenet30-sharpdro": 64,
    "cifar10-drift": 32,
    "cifar100-drift": 32,
    "imagenet30-drift": 64
}

## Width (depends on dataset; not all models use this).
width_dict = {
    "cifar10-sharpdro": 32,
    "cifar100-sharpdro": 32,
    "imagenet30-sharpdro": 64,
    "cifar10-drift": 32,
    "cifar100-drift": 32,
    "imagenet30-drift": 64
}

## Methods to be evaluated.
methods = ["AutoFloodedCVaR", "SoftFloodedCVaR", "SoftFloodedDRO", "AutoSoftFloodedCVaR", "FloodedCVaR", "FloodedDRO", "SharpDRO", "ERM"]

### Vanilla ERM.
mth_ERM = {}
mth_ERM_list = [mth_ERM]

### SharpDRO.
mth_SharpDRO = {
    "optimizer": "SharpDRO",
    "optimizer_base": "SGD",
}
radius_values = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
mth_SharpDRO_list = []
for i in range(len(radius_values)):
    to_add = copy.deepcopy(mth_SharpDRO)
    to_add["radius"] = radius_values[i]
    mth_SharpDRO_list += [to_add]

### Storage area for parameters that are common to multiple MJH methods.
dro_radius = 10.0
quantile_level = 0.5
flood_levels = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]
softad_levels = [0.0125, 0.025, 0.05, 0.1, 0.2, 0.4]

### FloodedDRO wrapper (fix the radius, try various flood levels).
mth_FloodedDRO = {}
mth_FloodedDRO["radius"] = dro_radius # fix the radius
mth_FloodedDRO_list = []
for i in range(len(flood_levels)):
    to_add = copy.deepcopy(mth_FloodedDRO)
    to_add["flood_level"] = flood_levels[i]
    mth_FloodedDRO_list += [to_add]

### FloodedCVaR wrapper (fix the quantile level, try various flood levels).
mth_FloodedCVaR = {}
mth_FloodedCVaR["quantile_level"] = quantile_level # fix the quantile level
mth_FloodedCVaR_list = []
for i in range(len(flood_levels)):
    to_add = copy.deepcopy(mth_FloodedCVaR)
    to_add["flood_level"] = flood_levels[i]
    mth_FloodedCVaR_list += [to_add]

### SoftFloodedDRO wrapper (fix the radius, try various flood levels).
mth_SoftFloodedDRO = {}
mth_SoftFloodedDRO["radius"] = dro_radius # fix the radius
mth_SoftFloodedDRO_list = []
for i in range(len(softad_levels)):
    to_add = copy.deepcopy(mth_SoftFloodedDRO)
    to_add["softad_level"] = softad_levels[i]
    mth_SoftFloodedDRO_list += [to_add]

### FloodedCVaR wrapper (fix the quantile level, try various flood levels).
mth_SoftFloodedCVaR = {}
mth_SoftFloodedCVaR["quantile_level"] = quantile_level # fix the quantile level
mth_SoftFloodedCVaR_list = []
for i in range(len(softad_levels)):
    to_add = copy.deepcopy(mth_SoftFloodedCVaR)
    to_add["softad_level"] = softad_levels[i]
    mth_SoftFloodedCVaR_list += [to_add]

### Sharpness-aware minimization (SAM).
mth_SAM = {
    "optimizer": "SAM",
    "optimizer_base": "SGD",
}
radius_values = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
mth_SAM_list = []
for i in range(len(radius_values)):
    to_add = copy.deepcopy(mth_SAM)
    to_add["radius"] = radius_values[i]
    mth_SAM_list += [to_add]

mth_paras_lists = {
    "ERM": mth_ERM_list,
    "FloodedCVaR": mth_FloodedCVaR_list,
    "FloodedDRO": mth_FloodedDRO_list,
    "AutoFloodedCVaR": mth_FloodedCVaR_list, # same as non-auto case
    "AutoFloodedDRO": mth_FloodedDRO_list, # same as non-auto case
    "SoftFloodedCVaR": mth_SoftFloodedCVaR_list,
    "SoftFloodedDRO": mth_SoftFloodedDRO_list,
    "AutoSoftFloodedCVaR": mth_SoftFloodedCVaR_list, # same as non-auto (soft) case
    "AutoSoftFloodedDRO": mth_SoftFloodedDRO_list, # same as non-auto (soft) case
    "SharpDRO": mth_SharpDRO_list,
    "SAM": mth_SAM_list
}


## MLflow clerical matters.
project_uri = os.getcwd()


# Driver function.

def main():
    
    # Loop over datasets. One mlflow experiment per dataset.
    for dataset_name in dataset_names:
        
        exp_name = "exp:{}".format(dataset_name)
        exp_id = mlflow.create_experiment(exp_name)
        paras_common["dataset"] = dataset_name
        paras_common["num_classes"] = num_classes_dict[dataset_name]
        paras_common["tr_frac"] = tr_frac_dict[dataset_name]
        paras_common["model"] = model_dict[dataset_name]
        paras_common["bs_tr"] = bs_tr_dict[dataset_name]
        paras_common["dimension"] = dimension_dict[dataset_name]
        paras_common["height"] = height_dict[dataset_name]
        paras_common["width"] = width_dict[dataset_name]
        paras_common["skip_singles"] = skip_singles_dict[dataset_name]
        
        # Parent run for consistency with other exps (just one parent).
        with mlflow.start_run(
                run_name="parent:run",
                experiment_id=exp_id
        ):
            for method in methods:
                
                paras_common["method"] = method
                mth_paras_list = mth_paras_lists[method]
                num_settings = len(mth_paras_list)
                    
                for j in range(num_settings):
                    
                    # Complete paras dict (set to defaults).
                    paras = dict(**paras_common, **paras_mth_defaults)
                    
                    # Reflect method-specific paras.
                    paras.update(mth_paras_list[j])
                    
                    # One child run for each setting (times num_trials).
                    for t in range(num_trials):
                        
                        # Make naming easy for post-processing.
                        rn = "child:{}-{}-t{}".format(method, j, t)
                        
                        # Be sure to give a fresh seed for each trial.
                        paras["random_seed"] = random_seeds[t]
                        
                        # Do the run.
                        mlflow.projects.run(uri=project_uri,
                                            entry_point="main_experiment",
                                            parameters=paras,
                                            experiment_id=exp_id,
                                            run_name=rn,
                                            env_manager="local")
    return None


if __name__ == "__main__":
    main()


###############################################################################
