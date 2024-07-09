'''Setup: ImageNet-30 dataset.'''

# External modules.
import numpy as np
import os
from PIL import Image

# Internal modules.
from sharpdro.prepare_dataset import get_transform
from sharpdro.make_imagenet_c import default_loader, find_classes, make_dataset
from setup.directories import data_path


###############################################################################


class ImageNet30_SharpDRO:
    '''
    Prepare data from ImageNet-30 dataset following the SharpDRO paper.
    https://github.com/zhuohuangai/SharpDRO
    '''
    
    # Label dictionary.
    label_dict = {
        0: "acorn",
        1: "airliner",
        2: "ambulance",
        3: "american_alligator",
        4: "banjo",
        5: "barn",
        6: "bikini",
        7: "digital_clock",
        8: "dragonfly",
        9: "dumbbell",
        10: "forklift",
        11: "goblet",
        12: "grand_piano",
        13: "hotdog",
        14: "hourglass",
        15: "manhole_cover",
        16: "mosque",
        17: "nail",
        18: "parking_meter",
        19: "pillow",
        20: "revolver",
        21: "rotary_dial_telephone",
        22: "schooner",
        23: "snowmobile",
        24: "soccer_ball",
        25: "stingray",
        26: "strawberry",
        27: "tank",
        28: "toaster",
        29: "volcano"
    }
    
    def __init__(self, rg, tr_frac=0.7661538461538462, num_severity_levels=5, corruption_type="gaussian_noise",
                 severity_dist="poisson", label_dict=label_dict, data_path=data_path):
        '''
        - rg : a Numpy random generator.
        '''
        
        print("--Preparing benchmark data (SharpDRO-style ImageNet-30)--")
        
        # Get number of classes based on label dictionary.
        num_classes = len(label_dict)

        # Hard-coded dataset sizes.
        self.n = 39000
        self.n_te = 3000
        
        # Hang on to the generator.
        self.rg = rg

        # Hang on to the fraction to be used for training.
        self.tr_frac = tr_frac

        # Set the number of points to be used for training.
        self.n_tr = int(self.n*self.tr_frac)

        # Set the location for accessing data.
        self.storage_path = os.path.join(data_path, "imagenet30-sharpdro")

        # Transform settings following the SharpDRO authors.
        # https://github.com/zhuohuangai/SharpDRO/blob/main/dataset/prepare_dataset.py
        transform_settings = {
            "size": (64, 64),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
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
            
            # Obtain the relevant data paths for the current severity level.
            print("== Severity level {} ==".format(severity_level))
            if severity_level == 0:
                # Get the clean data.
                path_tr = os.path.join(self.storage_path, "one_class_train") # dir for clean data
                path_te = os.path.join(self.storage_path, "one_class_test") # dir for clean data
            else:
                # Get the corrupted data.
                path_tr = os.path.join(self.storage_path, "train",
                                       corruption_type, str(severity_level)) # dir for corrupted data
                path_te = os.path.join(self.storage_path, "test",
                                       corruption_type, str(severity_level)) # dir for corrupted data
            # Do a directory check to be safe.
            if os.path.exists(path_tr) == False or os.path.exists(path_te) == False:
                raise ValueError("Directory expected to hold images is not found.")
            
            # Get the data and apply transforms all in one fell swoop.
            classes_tr, class_to_idx_tr = find_classes(path=path_tr)
            classes_te, class_to_idx_te = find_classes(path=path_te)
            idx_to_class_tr = {v: k for k, v in class_to_idx_tr.items()}
            idx_to_class_te = {v: k for k, v in class_to_idx_te.items()}
            path_label_list_tr = make_dataset(path=path_tr, class_to_idx=class_to_idx_tr)
            path_label_list_te = make_dataset(path=path_te, class_to_idx=class_to_idx_te)
            X_tmp = []
            Y_tmp = []
            X_te_tmp = []
            Y_te_tmp = []
            print("Reading in the training dataset...")
            for path, label in path_label_list_tr:
                X_tmp += [np.expand_dims(
                    np.array(train_transform(default_loader(path=path))).astype(np.float32),
                    axis=0)]
                Y_tmp += [int(label)]
            print("Reading in the test dataset...")
            for path, label in path_label_list_te:
                X_te_tmp += [np.expand_dims(
                    np.array(test_transform(default_loader(path=path))).astype(np.float32),
                    axis=0
                )]
                Y_te_tmp += [int(label)]
            # Concatenate transformed images, and extract all relevant sub-arrays.
            X_tmp = np.copy(np.concatenate(X_tmp)[idx])
            Y_tmp = np.copy(np.array(Y_tmp).astype(np.uint8)[idx])
            X_te_tmp = np.copy(np.concatenate(X_te_tmp)[idx_te])
            Y_te_tmp = np.copy(np.array(Y_te_tmp).astype(np.uint8)[idx_te])
            #print("X_tmp info:", X_tmp.shape, X_tmp.dtype)
            #print("Y_tmp shape:", Y_tmp.shape, Y_tmp.dtype)
            #print("X_te_tmp info:", X_te_tmp.shape, X_te_tmp.dtype)
            #print("Y_te_tmp shape:", Y_te_tmp.shape, Y_te_tmp.dtype)
            
            # Add to lists, will be concatenated later.
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



class ImageNet30_Drift:
    '''
    We prepare data from the ImageNet-30 dataset essentially
    following the SharpDRO paper, except that now we use
    *clean* original data for training, and use corrupted
    data only for test evaluation.
    '''
    
    # Label dictionary.
    label_dict = {
        0: "acorn",
        1: "airliner",
        2: "ambulance",
        3: "american_alligator",
        4: "banjo",
        5: "barn",
        6: "bikini",
        7: "digital_clock",
        8: "dragonfly",
        9: "dumbbell",
        10: "forklift",
        11: "goblet",
        12: "grand_piano",
        13: "hotdog",
        14: "hourglass",
        15: "manhole_cover",
        16: "mosque",
        17: "nail",
        18: "parking_meter",
        19: "pillow",
        20: "revolver",
        21: "rotary_dial_telephone",
        22: "schooner",
        23: "snowmobile",
        24: "soccer_ball",
        25: "stingray",
        26: "strawberry",
        27: "tank",
        28: "toaster",
        29: "volcano"
    }
    
    def __init__(self, rg, tr_frac=0.7661538461538462, num_severity_levels=5, corruption_type="gaussian_noise",
                 severity_dist="poisson", label_dict=label_dict, data_path=data_path):
        '''
        - rg : a Numpy random generator.
        '''
        
        print("--Preparing benchmark data (Drift-style ImageNet-30)--")
        
        # Get number of classes based on label dictionary.
        num_classes = len(label_dict)

        # Hard-coded dataset sizes.
        self.n = 39000
        self.n_te = 3000
        
        # Hang on to the generator.
        self.rg = rg

        # Hang on to the fraction to be used for training.
        self.tr_frac = tr_frac

        # Set the number of points to be used for training.
        self.n_tr = int(self.n*self.tr_frac)

        # Set the location for accessing data.
        self.storage_path = os.path.join(data_path, "imagenet30-sharpdro")

        # Transform settings following the SharpDRO authors.
        # https://github.com/zhuohuangai/SharpDRO/blob/main/dataset/prepare_dataset.py
        transform_settings = {
            "size": (64, 64),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225)
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

        # Get the clean training data.
        path_tr = os.path.join(self.storage_path, "one_class_train") # dir for clean data
        # Do a directory check to be safe.
        if os.path.exists(path_tr) == False:
            raise ValueError("Directory expected to hold training images is not found.")
        # Get the data and apply transforms all in one fell swoop.
        classes_tr, class_to_idx_tr = find_classes(path=path_tr)
        idx_to_class_tr = {v: k for k, v in class_to_idx_tr.items()}
        path_label_list_tr = make_dataset(path=path_tr, class_to_idx=class_to_idx_tr)
        X_tmp = []
        Y_tmp = []
        print("Reading in the training dataset...")
        for path, label in path_label_list_tr:
            X_tmp += [np.expand_dims(
                np.array(train_transform(default_loader(path=path))).astype(np.float32),
                axis=0)]
            Y_tmp += [int(label)]
        # Concatenate transformed images, and extract all relevant sub-arrays.
        X_tmp = np.copy(np.concatenate(X_tmp))
        Y_tmp = np.copy(np.array(Y_tmp).astype(np.uint8))
        #print("X_tmp info:", X_tmp.shape, X_tmp.dtype)
        #print("Y_tmp shape:", Y_tmp.shape, Y_tmp.dtype)
        # Prepare training data.
        self.X = np.copy(X_tmp)
        self.Y = np.copy(Y_tmp)
        self.S = np.full(shape=(len(self.X),), fill_value=0, dtype=np.uint8)
        
        # Loop over severity levels and construct the data to be used for training and testing.
        X_te_list = []
        Y_te_list = []
        S_te_list = []
        for severity_level in range(0, num_severity_levels+1):
            
            # Get the indices which will be assigned the current severity level.
            idx_te = severities_te == severity_level
            
            # Obtain the relevant data paths for the current severity level.
            print("== Severity level {} ==".format(severity_level))
            if severity_level == 0:
                # Get the clean test data.
                path_te = os.path.join(self.storage_path, "one_class_test") # dir for clean data
            else:
                # Get the corrupted test data.
                path_te = os.path.join(self.storage_path, "test",
                                       corruption_type, str(severity_level)) # dir for corrupted data
            # Do a directory check to be safe.
            if os.path.exists(path_te) == False:
                raise ValueError("Directory expected to hold test images is not found.")
            
            # Get the data and apply transforms all in one fell swoop.
            classes_te, class_to_idx_te = find_classes(path=path_te)
            idx_to_class_te = {v: k for k, v in class_to_idx_te.items()}
            path_label_list_te = make_dataset(path=path_te, class_to_idx=class_to_idx_te)
            X_te_tmp = []
            Y_te_tmp = []
            print("Reading in the test dataset...")
            for path, label in path_label_list_te:
                X_te_tmp += [np.expand_dims(
                    np.array(test_transform(default_loader(path=path))).astype(np.float32),
                    axis=0
                )]
                Y_te_tmp += [int(label)]
            # Concatenate transformed images, and extract all relevant sub-arrays.
            X_te_tmp = np.copy(np.concatenate(X_te_tmp)[idx_te])
            Y_te_tmp = np.copy(np.array(Y_te_tmp).astype(np.uint8)[idx_te])
            #print("X_te_tmp info:", X_te_tmp.shape, X_te_tmp.dtype)
            #print("Y_te_tmp shape:", Y_te_tmp.shape, Y_te_tmp.dtype)
            
            # Add to lists, will be concatenated later.
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
