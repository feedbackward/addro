# Repository for concentrated OCE learning algorithm

In this repository, we provide software and demonstrations related to the following paper:

- Making robust generalizers less rigid via soft ascent-descent. Matthew J. Holland and Toma Hamada. *Preprint*.

We provide code which can be used to faithfully reproduce all the experiment results given in the above paper, and can also be applied easily to more general machine learning tasks outside the examples considered here.


## Overview of what the paper is about

Essentially, we consider a strategy which combines transforming losses in an optimized certainty equivalent (OCE) style with a learning criterion that penalizes poor concentration of the transformed loss distribution. We establish formal links between our proposed algorithm and sharpness-penalizing methods, and compare and contrast our approach with the [SharpDRO method of Huang et al. (2023)](https://github.com/zhuohuangai/SharpDRO).


## Runthrough of software setup

To begin, we describe how we installed the various software used. All our heavy-duty computations are done on servers with Ubuntu (Server) 22.04.4 LTS, but this is not a strict requirement.


### SharpDRO-specific setup

We will be using some code from the [repository](https://github.com/zhuohuangai/SharpDRO) provided by the authors of the [SharpDRO paper](https://arxiv.org/abs/2303.13087) (Huang et al., [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Huang_Robust_Generalization_Against_Photon-Limited_Corruptions_via_Worst-Case_Sharpness_Minimization_CVPR_2023_paper.html)). Their code makes use of [Wand](https://docs.wand-py.org/en/latest/index.html), a Python binding of [ImageMagick](https://imagemagick.org/index.php). We installed the latter using the following command (see also [the install docs](https://docs.wand-py.org/en/latest/guide/install.html)), and we will get to the former a bit later on.

```
$ sudo apt-get install libmagickwand-dev
```


### CUDA toolkit preparations

Assuming one has an NVIDIA GPU installed and wants to utilize it via [CUDA](https://developer.nvidia.com/cuda-toolkit), some preparation is involved. This can of course vary significantly depending on the hardware and OS that you have, but for reference, here are the steps we use (on Ubuntu).

First, do a standard update-upgrade to ensure everything is sufficiently up-to-date.

```
$ sudo apt update
$ sudo apt upgrade
```

Next, for CUDA-related preparation, just do the "pre-installation actions" recommended in the [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), including installing [gcc](https://gcc.gnu.org/) as required.

The final step is installation of NVIDIA drivers. This can always be a rather tricky process, but in all our tests, we had no issues with the following simple sequence of commands.

```
$ ubuntu-drivers devices
$ ubuntu-drivers list
$ ubuntu-drivers install
```

It is possible to specify particular packages, but we just went with the recommendations for our hardware as detected by this utility. For changes to take effect, a reboot is required.

This wraps up the CUDA-related *preparation*. Note that we have not actually installed CUDA yet; this will be done alongside the installation of PyTorch later on.


### Virtual environment setup using conda

To manage virtual environments, we [install miniforge](https://github.com/conda-forge/miniforge) (i.e., a conda-forge aligned miniconda). Once this has been installed, we run the usual conda update to be safe.

```
$ conda update -n base conda
```

Next we create a new virtual environment with pytorch installed in a CUDA-ready fashion.

```
$ conda create -n pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

There are several other software libraries we will take advantage of here. First we get [JupyterLab](https://jupyter.org/install), [matplotlib](https://matplotlib.org/stable/users/installing/index.html), and [scikit-image](https://scikit-image.org/docs/stable/user_guide/install.html) using conda.

```
$ conda activate addro
(addro) $ conda install jupyterlab
(addro) $ conda install matplotlib
(addro) $ conda install scikit-image
```

We then use pip to install [mlflow](https://mlflow.org/), [Wand](https://docs.wand-py.org/en/latest/guide/install.html), and [OpenCV](https://opencv.org/get-started/).

```
(addro) $ pip install mlflow
(addro) $ pip install Wand
(addro) $ pip install opencv-python
```

This wraps up our software preparation. We will need to make use of this software when preparing data.


## Data setup

Here we describe how to prepare the datasets that we have used in our experimental evaluation.


### Preparation of SharpDRO-style CIFAR-10/100 datasets

For the experiments done in the style of Huang et al. (2023), the idea is to modify the original "clean" data following the well-cited work of Hendrycks and Dietterich (2019; [arXiv](https://arxiv.org/abs/1903.12261), [GitHub](https://github.com/hendrycks/robustness)) such that both the training data and test data includes "corrupted" or "perturbed" examples. The basic notion of "five levels of corruption severity" comes directly from the Hendrycks and Dietterich work. It should also be noted that originally the CIFAR-10-C ([online repository](https://zenodo.org/records/2535967)) and CIFAR-100-C ([online repository](https://zenodo.org/records/3555552)) corrupted datasets as designed by Hendrycks and Dietterich were intended for *testing*, and thus the online repositories just linked only include modified versions of the 10,000 images designated as "test images" in the [original CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) released by Krizhevsky, Nair, and Hinton. Modification of the 50,000 "training images" in a similar fashion must thus be done manually, and the SharpDRO repository includes code to prepare modified versions of the training and test data manually.

With this all established, getting things organized requires just two simple steps, which we describe below.

- Download the clean CIFAR-10/100 data via the [torchvision datasets](https://pytorch.org/vision/main/datasets.html) implementation.
- Modify the data in the style of CIFAR-10/100-C designed by Hendrycks and Dietterich (2019).

Within our repository here, both steps are handled within a simple script called `run_data_getter.py`. To obtain CIFAR-10 and CIFAR-100 datasets in the style of the SharpDRO paper, just run the following command.

```
(addro) $ python run_data_getter.py --cifar10-sharpdro --cifar100-sharpdro
```

Our default settings have `addro/data` set as the main data directory. Clean data via torchvision is saved by default in sub-directories called `cifar-10-batches-py` and `cifar-100-python`, and the SharpDRO-style modified data is stored in `cifar10-sharpdro` and `cifar100-sharpdro`.

Using the procedure just described, we have six "alternative versions" of the original 60,000 images in CIFAR-10/100 (1 clean + 5 corrupted = 6). For training in the style of the SharpDRO experiments, each original image is used *only once*, with varying degree of corruption severity. The fraction of training images subjected to each level of corruption is a flexible point of experiment design, but the SharpDRO authors are interested in so-called "photo-limited corruptions" that occur in a random, non-uniform fashion. More specifically, this is modeled using a Poisson distribution with shape parameter of 1.0 ("shape" refers to `lam` in the [NumPy implementation](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.poisson.html)). According to this distribution, the probability that the degree of corruption will be no greater than 5 is approximately 99.94%, and so what we do is simply take an independent and identically distributed (IID) sample from the aforementioned Poisson distribution, truncating values greater than 5 (i.e., values of 6, 7, 8, and greater are changed to 5). Of the 50,000 original training images, this Poisson-based random sample of severity levels determines which images are left clean, and which images receive corruption (and how severe it is).

Of the 50,000 images available for training (after corruption), we follow the SharpDRO authors in using 39,950 for actual training and 10,050 for validation. We note that this split is made *after* obtaining the corrupted dataset and shuffling the order of images.


### Preparation of SharpDRO-style ImageNet-30 dataset

Our explanation here is mostly analogous to that for CIFAR-10/100 given above, save for a few technical details. To begin, let us clarify what "clean" data is being used here. In Huang et al. (2023), they say they use a mixture of clean and corrupted variants of the images in the "ImageNet30" dataset. The reference they provide is to [the general ImageNet paper](https://link.springer.com/article/10.1007/s11263-015-0816-y), but some digging through their code suggests that "ImageNet30" refers to the 30-class simplified version of ImageNet that was constructed in the work of [Hendrycks et al. (2019)](https://arxiv.org/abs/1906.12340); see their Appendix B ("ImageNet OOD Dataset") for reference, and [their GitHub repository](https://github.com/hendrycks/ss-ood) for more details and access to the actual data. Henceforth, we refer to this original dataset as the (clean) *ImageNet-30* dataset.

Moving forward, we assume the user has carried out the following steps manually.

- Download the ImageNet-30 dataset from [their GitHub repository](https://github.com/hendrycks/ss-ood). For convenience, here are links to the [training data](https://drive.google.com/file/d/1B5c39Fc3haOPzlehzmpTLz6xLtGyKEy4/view?usp=sharing) and [testing data](https://drive.google.com/file/d/13xzVuQMEhSnBRZr-YaaO08coLU2dxAUq/view?usp=sharing).

- Having downloaded the data, composed of two directories named `one_class_train` and `one_class_test`, place these two directories in a new directory `addro/data/imagenet30-sharpdro` (please make this directory).

- Finally, note that in `one_class_test/airliner`, there is a hidden file called `._1.JPEG` that doesn't play well with the scripts for preparing data provided by Huang et al. at their GitHub repository. As such, please delete this hidden file.

Having come this far, the *clean* data is in hand, and our next task is to prepare the corrupted data. This is done using the following command.

```
(addro) $ python run_data_getter.py --imagenet30-sharpdro
```

By default, the corrupted verions of the training and testing data are stored in `addro/data/imagenet30-sharpdro/train` and `addro/data/imagenet30-sharpdro/test` respectively, split up by severity levels, just as in CIFAR-10/100.

Everything else regarding how we sub-sample indices from each severity level is identical to that described above for CIFAR-10/100. The only minor difference is the size of the training-validation split; of the 39,000 original training images, 29,880 are used for training, and 9,120 are used for validation. This is in line with the code provided by the SharpDRO authors.


## Guide to using the repository


### Running main experiments

All our main experiments are controlled by two key files, namely `main_experiment.py` and `run_main_experiment.py`. The former is the main driver script, and the latter is used to specify experimental setup details which are then passed to the driver script, via mlflow.

As such, assuming all the software and data has been prepared as described, running experiments is a one-line operation:

```
(addro) python run_main_experiment.py
```

Of course, users are free to specify experimental settings different from our defaults. Here are some key pointers.

- Q: *How do I change the datasets used?* A: By modifying the list called `dataset_names` within `run_main_experiment.py`.
- Q: *How do I distinguish between experiments with and without distribution drift?* A: In `dataset_names` just mentioned, the format for elements of the list is assumed to be `[dataset]-[exptype]`, where the "dataset" is `cifar10`, `cifar100`, or `imagenet30`, and the "exptype" is either `sharpdro` (no drift) or `drift`.
- Q: *How do I specify which model (CNN or WRN) to use?* A: Use the dictionary called `model_dict` in `run_main_experiment.py`. Our simple CNN is denoted by `CNN_Prototyping`, and the WRN is denoted by `WideResNet_Huang`.
- Q: *How do I specify which methods to test?* A: Methods are specified using the list called `methods` within `run_main_experiments.py`.
- Q: *Which method names correspond to the methods tested in the paper?* A: COCE and COCE-A are respectively referred to as `SoftFloodedCVaR` and `AutoSoftFloodedCVaR`. ERM is `ERM`, SharpDRO is `SharpDRO`, and SAM is `SAM`.
- Q: *How do I change the corruption type or severity distribution used?* A: Both are set within the dictionary `paras_common` in `run_main_experiment.py`.


### Obtaining figures

Having run the experiments using `run_main_experiment.py` as described, experimental results will be stored in a directory called `mlruns` by default. All the figures showing experimental results in our paper are generated from within the Jupyter notebook called `eval_main_experiment.ipynb`. Here are pointers for how to use this notebook.

- Specify the severity distribution and model used in `severity_dist` and `model_used`.
- Specify the experiment name in `exp_name` using the form `exp:[dataset_name]`, where this `dataset_name` should correspond to any one of the elements in the list `dataset_names` within `run_main_experiment.py` when the experiments were run.
- Run all the cells in the notebook, and check the `img` directory, which is where the generated figures will be stored by default.

Other illustrative figures that appear in the paper are generated within `figure_functions.ipynb` and `figure_state_distribution.ipynb`.
