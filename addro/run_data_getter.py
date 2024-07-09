'''Script for preparation of all relevant datasets.'''

# External modules.
from argparse import ArgumentParser

# Internal modules.
from setup.cifar10 import get_cifar10
from sharpdro.make_cifar_c import make_cifar_c
from sharpdro.make_imagenet_c import make_imagenet30_c


###############################################################################


def get_parser():
    parser = ArgumentParser(
        description="Preparation of all relevant datasets.",
        add_help=True
    )
    parser.add_argument("--cifar10", default=False, action="store_true")
    parser.add_argument("--cifar10-sharpdro", default=False, action="store_true")
    parser.add_argument("--cifar100-sharpdro", default=False, action="store_true")
    parser.add_argument("--imagenet30-sharpdro", default=False, action="store_true")
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def main(args):
    
    if args.cifar10:
        print("Getting CIFAR-10.")
        result_of_get = get_cifar10()
        print(result_of_get)
        
    if args.cifar10_sharpdro:
        print("Making SharpDRO-style CIFAR-10.")
        result_of_make = make_cifar_c(cifar_10=True)
        print(result_of_make)
        
    if args.cifar100_sharpdro:
        print("Making SharpDRO-style CIFAR-100.")
        result_of_make = make_cifar_c(cifar_100=True)
        print(result_of_make)

    if args.imagenet30_sharpdro:
        print("Making SharpDRO-style ImageNet-30.")
        result_of_make = make_imagenet30_c()
        print(result_of_make)
    
    return None


if __name__ == "__main__":
    args = get_args()
    main(args)


###############################################################################
