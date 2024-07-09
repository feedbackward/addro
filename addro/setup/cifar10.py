'''Setup: CIFAR-10 dataset.'''

# External modules.
import numpy as np
import os
from PIL import Image
import torchvision

# Internal modules.
from sharpdro.prepare_dataset import get_transform
from setup.directories import data_path
from setup.utils import makedir_safe, do_normalization


###############################################################################


def get_cifar10(return_data=True):
    makedir_safe(data_path)
    data_raw_tr = torchvision.datasets.CIFAR10(root=data_path,
                                               train=True,
                                               download=True,
                                               transform=None)
    data_raw_te = torchvision.datasets.CIFAR10(root=data_path,
                                               train=False,
                                               download=True,
                                               transform=None)
    if return_data:
        return data_raw_tr, data_raw_te
    else:
        return "Got CIFAR-10."


class CIFAR10_SharpDRO:
    '''
    Prepare data from CIFAR-10 dataset following the SharpDRO paper.
    https://github.com/zhuohuangai/SharpDRO
    '''
    
    # Label dictionary.
    label_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    
    def __init__(self, rg, tr_frac=0.799, num_severity_levels=5, corruption_type="gaussian_noise",
                 severity_dist="poisson", label_dict=label_dict, data_path=data_path):
        '''
        - rg : a Numpy random generator.
        '''
        
        print("--Preparing benchmark data (SharpDRO-style CIFAR-10)--")
        
        # Get number of classes based on label dictionary.
        num_classes = len(label_dict)

        # Hard-coded dataset sizes.
        self.n = 50000
        self.n_te = 10000
        
        # Hang on to the generator.
        self.rg = rg

        # Hang on to the fraction to be used for training.
        self.tr_frac = tr_frac

        # Set the number of points to be used for training.
        self.n_tr = int(self.n*self.tr_frac)

        # Set the location for accessing data.
        self.storage_path = os.path.join(data_path, "cifar10-sharpdro")

        # Transform settings following the SharpDRO authors.
        # https://github.com/zhuohuangai/SharpDRO/blob/main/dataset/prepare_dataset.py
        transform_settings = {
            "size": (32, 32),
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2471, 0.2435, 0.2616)
        }
        train_transform, test_transform = get_transform(size=transform_settings["size"],
                                                        mean=transform_settings["mean"],
                                                        std=transform_settings["std"])

        # Get random severity levels via specified distribution.
        if severity_dist == "poisson":
            severities = np.clip(rg.poisson(lam=1.0, size=self.n), a_min=None,
                                 a_max=num_severity_levels).astype(np.uint8)
            severities_te = np.clip(rg.poisson(lam=1.0, size=self.n_te), a_min=None,
                                    a_max=num_severity_levels).astype(np.uint8)
        elif severity_dist == "uniform":
            severities = rg.integers(low=0, high=num_severity_levels+1, size=self.n).astype(np.uint8)
            severities_te = rg.integers(low=0, high=num_severity_levels+1, size=self.n_te).astype(np.uint8)
        elif severity_dist == "revpoisson":
            severities = np.clip(rg.poisson(lam=1.0, size=self.n), a_min=None, a_max=num_severity_levels)
            severities = severities-num_severity_levels
            severities = np.negative(severities).astype(np.uint8)
            severities_te = np.clip(rg.poisson(lam=1.0, size=self.n_te), a_min=None, a_max=num_severity_levels)
            severities_te = severities_te-num_severity_levels
            severities_te = np.negative(severities_te).astype(np.uint8)
        else:
            raise ValueError("The severity distribution specified is not valid.")

        
        # Loop over severity levels and construct the data to be used for training and testing.
        X_list = []
        X_te_list = []
        Y_list = []
        Y_te_list = []
        S_list = []
        S_te_list = []
        for severity_level in range(0, num_severity_levels+1):
        
            # Get the indices which will be assigned the current severity level.
            idx = severities == severity_level
            idx_te = severities_te == severity_level
        
            # Obtain the relevant data for the current severity level.
            if severity_level == 0:
                # Get the clean data and extract sub-arrays.
                data_raw_tr, data_raw_te = get_cifar10(return_data=True)
                X_tmp_raw = data_raw_tr.data[idx]
                X_te_tmp_raw = data_raw_te.data[idx_te]
                Y_tmp = np.array(data_raw_tr.targets)[idx].astype(np.uint8)
                Y_te_tmp = np.array(data_raw_te.targets)[idx_te].astype(np.uint8)
                # Run the SharpDRO-style transforms.
                X_tmp = []
                X_te_tmp = []
                for i in range(len(X_tmp_raw)):
                    X_tmp += [
                        np.expand_dims(np.array(train_transform(Image.fromarray(X_tmp_raw[i], mode="RGB"))), axis=0)
                    ]
                X_tmp = np.copy(np.concatenate(X_tmp).astype(np.float32))
                for i in range(len(X_te_tmp_raw)):
                    X_te_tmp += [
                        np.expand_dims(np.array(test_transform(Image.fromarray(X_te_tmp_raw[i], mode="RGB"))), axis=0)
                    ]
                X_te_tmp = np.copy(np.concatenate(X_te_tmp).astype(np.float32))
                # Add to lists, will concatenate later.
                X_list += [np.copy(X_tmp)]
                X_te_list += [np.copy(X_te_tmp)]
                Y_list += [np.copy(Y_tmp)]
                Y_te_list += [np.copy(Y_te_tmp)]
            else:
                # Get corrupted data and extract sub-arrays.
                X_tmp_raw = np.load(os.path.join(self.storage_path, "train", str(severity_level),
                                                 "{}.npy".format(corruption_type)))[idx]
                X_te_tmp_raw = np.load(os.path.join(self.storage_path, "test", str(severity_level),
                                                    "{}.npy".format(corruption_type)))[idx_te]
                Y_tmp = np.load(os.path.join(self.storage_path, "train", str(severity_level), "labels.npy"))[idx]
                Y_tmp = Y_tmp.astype(np.uint8)
                Y_te_tmp = np.load(os.path.join(self.storage_path, "test", str(severity_level), "labels.npy"))[idx_te]
                Y_te_tmp = Y_te_tmp.astype(np.uint8)
                # Run the SharpDRO-style transforms.
                X_tmp = []
                X_te_tmp = []
                for i in range(len(X_tmp_raw)):
                    X_tmp += [
                        np.expand_dims(np.array(train_transform(Image.fromarray(X_tmp_raw[i], mode="RGB"))), axis=0)
                    ]
                X_tmp = np.copy(np.concatenate(X_tmp).astype(np.float32))
                for i in range(len(X_te_tmp_raw)):
                    X_te_tmp += [
                        np.expand_dims(np.array(test_transform(Image.fromarray(X_te_tmp_raw[i], mode="RGB"))), axis=0)
                    ]
                X_te_tmp = np.copy(np.concatenate(X_te_tmp).astype(np.float32))
                # Add to lists, will concatenate later.
                X_list += [np.copy(X_tmp)]
                X_te_list += [np.copy(X_te_tmp)]
                Y_list += [np.copy(Y_tmp)]
                Y_te_list += [np.copy(Y_te_tmp)]

            # Append the severity level "labels".
            S_list += [np.full(shape=(len(X_tmp),), fill_value=severity_level, dtype=np.uint8)]
            S_te_list += [np.full(shape=(len(X_te_tmp),), fill_value=severity_level, dtype=np.uint8)]
        
        # Now that the loop has finished, concatenate to get the full corrupted datasets.
        self.X = np.copy(np.concatenate(X_list))
        self.Y = np.copy(np.concatenate(Y_list))
        self.S = np.copy(np.concatenate(S_list))
        self.X_te = np.copy(np.concatenate(X_te_list))
        self.Y_te = np.copy(np.concatenate(Y_te_list))
        self.S_te = np.copy(np.concatenate(S_te_list))
        #print(self.X[0], self.Y[0], self.S[0])
        #print(self.X.shape, self.X.dtype, self.Y.shape, self.Y.dtype)
        #print(self.X_te.shape, self.X_te.dtype, self.Y_te.shape, self.Y_te.dtype)
        del X_tmp_raw, X_te_tmp_raw
        del X_list, Y_list, S_list, X_te_list, Y_te_list, S_te_list

        # Check that sizes are as expected.
        if len(self.X) != self.n or len(self.X_te) != self.n_te:
            raise ValueError("Data sizes are not as expected.")
        else:
            print("Data sizes as expected; X is {}, X_te is {}.".format(len(self.X), len(self.X_te)))
            print("Data sizes as expected; Y is {}, Y_te is {}.".format(len(self.Y), len(self.Y_te)))
        
        return None
        
    
    def __call__(self):
        '''
        Each call gives us a chance to shuffle up the tr/va data.
        '''
        
        # Do random shuffling.
        idx_shuffled = np.arange(self.n)
        #print("before", idx_shuffled)
        self.rg.shuffle(idx_shuffled)
        #print("after", idx_shuffled)
        self.X = self.X[idx_shuffled]
        self.Y = self.Y[idx_shuffled]
        self.S = self.S[idx_shuffled]

        # Do tr/va split.
        X_tr = np.copy(self.X[0:self.n_tr])
        Y_tr = np.copy(self.Y[0:self.n_tr])
        S_tr = np.copy(self.S[0:self.n_tr])
        X_va = np.copy(self.X[self.n_tr:])
        Y_va = np.copy(self.Y[self.n_tr:])
        S_va = np.copy(self.S[self.n_tr:])
        
        return (X_tr, Y_tr, S_tr, X_va, Y_va, S_va, self.X_te, self.Y_te, self.S_te)



class CIFAR10_Drift:
    '''
    We prepare data from the CIFAR-10 dataset essentially
    following the SharpDRO paper, except that now we use
    *clean* original data for training, and use corrupted
    data only for test evaluation.
    '''
    
    # Label dictionary.
    label_dict = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }
    
    def __init__(self, rg, tr_frac=0.799, num_severity_levels=5, corruption_type="gaussian_noise",
                 severity_dist="poisson", label_dict=label_dict, data_path=data_path):
        '''
        - rg : a Numpy random generator.
        '''
        
        print("--Preparing benchmark data (Drift-style CIFAR-10)--")
        
        # Get number of classes based on label dictionary.
        num_classes = len(label_dict)

        # Hard-coded dataset sizes.
        self.n = 50000
        self.n_te = 10000
        
        # Hang on to the generator.
        self.rg = rg

        # Hang on to the fraction to be used for training.
        self.tr_frac = tr_frac

        # Set the number of points to be used for training.
        self.n_tr = int(self.n*self.tr_frac)

        # Set the location for accessing data.
        self.storage_path = os.path.join(data_path, "cifar10-sharpdro")

        # Transform settings following the SharpDRO authors.
        # https://github.com/zhuohuangai/SharpDRO/blob/main/dataset/prepare_dataset.py
        transform_settings = {
            "size": (32, 32),
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2471, 0.2435, 0.2616)
        }
        train_transform, test_transform = get_transform(size=transform_settings["size"],
                                                        mean=transform_settings["mean"],
                                                        std=transform_settings["std"])

        # Get random severity levels via specified distribution.
        if severity_dist == "poisson":
            severities_te = np.clip(rg.poisson(lam=1.0, size=self.n_te), a_min=None,
                                    a_max=num_severity_levels).astype(np.uint8)
        elif severity_dist == "uniform":
            severities_te = rg.integers(low=0, high=num_severity_levels+1, size=self.n_te).astype(np.uint8)
        elif severity_dist == "revpoisson":
            severities_te = np.clip(rg.poisson(lam=1.0, size=self.n_te), a_min=None, a_max=num_severity_levels)
            severities_te = severities_te-num_severity_levels
            severities_te = np.negative(severities_te).astype(np.uint8)
        else:
            raise ValueError("The severity distribution specified is not valid.")

        # Get the clean data.
        data_raw_tr, data_raw_te = get_cifar10(return_data=True)
        
        # Training data is independent of corruption severity levels, so we prepare it first.
        X_tmp_raw = data_raw_tr.data
        Y_tmp = np.array(data_raw_tr.targets).astype(np.uint8)
        # Run the SharpDRO-style transforms.
        X_tmp = []
        for i in range(len(X_tmp_raw)):
            X_tmp += [
                np.expand_dims(np.array(train_transform(Image.fromarray(X_tmp_raw[i], mode="RGB"))), axis=0)
            ]
        # Prepare training data.
        self.X = np.copy(np.concatenate(X_tmp).astype(np.float32))
        self.Y = np.copy(Y_tmp)
        self.S = np.full(shape=(len(self.X),), fill_value=0, dtype=np.uint8)
        
        # Loop over severity levels and construct the data to be used for training and testing.
        X_te_list = []
        Y_te_list = []
        S_te_list = []
        for severity_level in range(0, num_severity_levels+1):
        
            # Get the indices which will be assigned the current severity level.
            idx_te = severities_te == severity_level
        
            # Obtain the relevant data for the current severity level.
            if severity_level == 0:
                # Extract sub-arrays from clean data.
                X_te_tmp_raw = data_raw_te.data[idx_te]
                Y_te_tmp = np.array(data_raw_te.targets)[idx_te].astype(np.uint8)
                # Run the SharpDRO-style transforms.
                X_te_tmp = []
                for i in range(len(X_te_tmp_raw)):
                    X_te_tmp += [
                        np.expand_dims(np.array(test_transform(Image.fromarray(X_te_tmp_raw[i], mode="RGB"))), axis=0)
                    ]
                X_te_tmp = np.copy(np.concatenate(X_te_tmp).astype(np.float32))
                # Add to lists, will concatenate later.
                X_te_list += [np.copy(X_te_tmp)]
                Y_te_list += [np.copy(Y_te_tmp)]
            else:
                # Get corrupted data and extract sub-arrays.
                X_te_tmp_raw = np.load(os.path.join(self.storage_path, "test", str(severity_level),
                                                    "{}.npy".format(corruption_type)))[idx_te]
                Y_te_tmp = np.load(os.path.join(self.storage_path, "test", str(severity_level), "labels.npy"))[idx_te]
                Y_te_tmp = Y_te_tmp.astype(np.uint8)
                # Run the SharpDRO-style transforms.
                X_te_tmp = []
                for i in range(len(X_te_tmp_raw)):
                    X_te_tmp += [
                        np.expand_dims(np.array(test_transform(Image.fromarray(X_te_tmp_raw[i], mode="RGB"))), axis=0)
                    ]
                X_te_tmp = np.copy(np.concatenate(X_te_tmp).astype(np.float32))
                # Add to lists, will concatenate later.
                X_te_list += [np.copy(X_te_tmp)]
                Y_te_list += [np.copy(Y_te_tmp)]

            # Append the severity level "labels".
            S_te_list += [np.full(shape=(len(X_te_tmp),), fill_value=severity_level, dtype=np.uint8)]
        
        # Now that the loop has finished, concatenate to get the full corrupted datasets.
        self.X_te = np.copy(np.concatenate(X_te_list))
        self.Y_te = np.copy(np.concatenate(Y_te_list))
        self.S_te = np.copy(np.concatenate(S_te_list))
        #print(self.X[0], self.Y[0], self.S[0])
        #print(self.X.shape, self.X.dtype, self.Y.shape, self.Y.dtype)
        #print(self.X_te.shape, self.X_te.dtype, self.Y_te.shape, self.Y_te.dtype)
        del X_tmp_raw, X_te_tmp_raw
        del X_te_list, Y_te_list, S_te_list

        # Check that sizes are as expected.
        if len(self.X) != self.n or len(self.X_te) != self.n_te:
            raise ValueError("Data sizes are not as expected.")
        else:
            print("Data sizes as expected; X is {}, X_te is {}.".format(len(self.X), len(self.X_te)))
            print("Data sizes as expected; Y is {}, Y_te is {}.".format(len(self.Y), len(self.Y_te)))
        
        return None
        
    
    def __call__(self):
        '''
        Each call gives us a chance to shuffle up the tr/va data.
        '''
        
        # Do random shuffling.
        idx_shuffled = np.arange(self.n)
        #print("before", idx_shuffled)
        self.rg.shuffle(idx_shuffled)
        #print("after", idx_shuffled)
        self.X = self.X[idx_shuffled]
        self.Y = self.Y[idx_shuffled]
        self.S = self.S[idx_shuffled]
        
        # Do tr/va split.
        X_tr = np.copy(self.X[0:self.n_tr])
        Y_tr = np.copy(self.Y[0:self.n_tr])
        S_tr = np.copy(self.S[0:self.n_tr])
        X_va = np.copy(self.X[self.n_tr:])
        Y_va = np.copy(self.Y[self.n_tr:])
        S_va = np.copy(self.S[self.n_tr:])
        
        return (X_tr, Y_tr, S_tr, X_va, Y_va, S_va, self.X_te, self.Y_te, self.S_te)
        

###############################################################################
