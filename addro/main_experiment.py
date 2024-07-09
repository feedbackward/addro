'''Code for SharpDRO-style experiments (and related tests).'''

# External modules.
from argparse import ArgumentParser
import mlflow
import numpy as np
import os
import torch

# Internal modules.
from setup.data import get_dataloader
from setup.directories import models_path
from setup.eval import run_evals, run_gradnorm_evals
from setup.losses import get_loss
from setup.models import get_model
from setup.optimizers import get_optimizer
from setup.training import Trainer
from setup.utils import makedir_safe
from sharpdro.scheduler import SS_Scheduler


###############################################################################


def get_parser():
    parser = ArgumentParser(
        prog="main_experiment",
        description="Main SharpDRO-style experiments.",
        add_help=True
    )
    parser.add_argument("--adaptive",
                        help="Set to 'yes' for adaptive SAM.",
                        type=str)
    parser.add_argument("--base-gpu-id",
                        default=0,
                        help="Specify which GPU should be the base GPU.",
                        type=int)
    parser.add_argument("--bs-tr",
                        help="Batch size for training data loader.",
                        type=int)
    parser.add_argument("--corruption-type",
                        help="Name of the corruption type to use, if applicable.",
                        type=str)
    parser.add_argument("--dataset",
                        help="Dataset name.",
                        type=str)
    parser.add_argument("--dimension",
                        help="Dimension of inputs.",
                        type=int)
    parser.add_argument("--epochs",
                        help="Number of epochs in training loop.",
                        type=int)
    parser.add_argument("--eta",
                        help="Weight parameter on theta in SoftAD.",
                        type=float)
    parser.add_argument("--flood-level",
                        help="Flood level parameter for Ishida method.",
                        type=float)
    parser.add_argument("--force-cpu",
                        help="Either yes or no; parse within main().",
                        type=str)
    parser.add_argument("--force-one-gpu",
                        help="Either yes or no; parse within main().",
                        type=str)
    parser.add_argument("--gradnorm",
                        help="Measure gradients norms? Either yes or no.",
                        type=str)
    parser.add_argument("--height",
                        help="Image height (number of pixels).",
                        type=int)
    parser.add_argument("--loss",
                        help="Name of the base loss function.",
                        type=str)
    parser.add_argument("--method",
                        help="Abstract method name.",
                        type=str)
    parser.add_argument("--model",
                        help="Model name.",
                        type=str)
    parser.add_argument("--momentum",
                        help="Momentum parameter for optimizers.",
                        type=float)
    parser.add_argument("--num-classes",
                        help="Number of classes (for classification tasks).",
                        type=int)
    parser.add_argument("--num-severity-levels",
                        help="Number of severity levels to consider.",
                        type=int)
    parser.add_argument("--optimizer",
                        help="Optimizer name.",
                        type=str)
    parser.add_argument("--optimizer-base",
                        help="Base optimizer name (for SAM and SharpDRO).",
                        type=str)
    parser.add_argument("--pre-trained",
                        help="Specify pre-trained weights.",
                        type=str)
    parser.add_argument("--prob-update-factor",
                        help="Factor for updating probabilities in dist-aware SharpDRO.",
                        type=float)
    parser.add_argument("--quantile-level",
                        help="Quantile level parameter for CVaR.",
                        type=float)
    parser.add_argument("--radius",
                        help="Radius parameter (SAM, DRO, SharpDRO).",
                        type=float)
    parser.add_argument("--random-seed",
                        help="Integer-valued random seed.",
                        type=int)
    parser.add_argument("--saving-freq",
                        help="Frequency at which to save models.",
                        type=int)
    parser.add_argument("--scheduler",
                        help="String specifying how to do step-size scheduling.",
                        type=str)
    parser.add_argument("--severity-dist",
                        help="String specifying which severity distribution to use.",
                        type=str)
    parser.add_argument("--sigma",
                        help="Scaling parameter for SoftAD.",
                        type=float)
    parser.add_argument("--skip-singles",
                        help="Specify if we need to skip single batches.",
                        type=str)
    parser.add_argument("--softad-level",
                        help="Threshold (shift) parameter for SoftAD.",
                        type=float)
    parser.add_argument("--step-size",
                        help="Step size parameter for optimizers.",
                        type=float)
    parser.add_argument("--tilt",
                        help="Tilt parameter for Tilted ERM.",
                        type=float)
    parser.add_argument("--tr-frac",
                        help="Fraction of data not used for validation.",
                        type=float)
    parser.add_argument("--weight-decay",
                        help="Weight decay parameter for optimizers.",
                        type=float)
    parser.add_argument("--width",
                        help="Image width (number of pixels).",
                        type=int)
    return parser


def get_args():
    parser = get_parser()
    return parser.parse_args()


def main(args):

    # Organize clerical arguments.
    force_cpu = True if args.force_cpu == "yes" else False
    force_one_gpu = True if args.force_one_gpu == "yes" else False
    base_gpu_id = args.base_gpu_id
    skip_singles = True if args.skip_singles == "yes" else False
    gradnorm = True if args.gradnorm == "yes" else False
    saving_freq = int(args.saving_freq)
    save_models = True if saving_freq > 0 else False
    saving_counter = 0 # only used if save_models is True
    if save_models:
        makedir_safe(models_path)
    
    # Device setup.
    if force_cpu and not force_one_gpu:
        device = torch.device(type="cpu")
    elif force_one_gpu and not force_cpu:
        device = torch.device(type="cuda", index=base_gpu_id)
    else:
        raise ValueError("Please specify either CPU or single GPU setting.")
    
    # Seed the random generator (numpy and torch).
    rg = np.random.default_rng(args.random_seed)
    rg_torch = torch.manual_seed(seed=args.random_seed)
    
    # Get the data (placed on desired device).
    dataset_paras = {
        "rg": rg,
        "bs_tr": args.bs_tr,
        "corruption_type": args.corruption_type,
        "dimension": args.dimension,
        "num_severity_levels": args.num_severity_levels,
        "severity_dist": args.severity_dist,
        "tr_frac": args.tr_frac
    }
    dl_tr, eval_dl_tr, eval_dl_va, eval_dl_te = get_dataloader(
        dataset_name=args.dataset,
        dataset_paras=dataset_paras,
        device=device
    )
    
    # Initialize the model (placed on desired device).
    print("== Model: ==")
    model_paras = {
        "rg": rg,
        "dimension": args.dimension,
        "height": args.height,
        "width": args.width,
        "num_classes": args.num_classes,
        "pre_trained": args.pre_trained
    }
    model = get_model(model_name=args.model, model_paras=model_paras)
    model = model.to(device)
    print(model)
    print("============")

    # Get the loss function ready.
    print("== Loss function: ==")
    loss_paras = {"method": args.method,
                  "flood_level": args.flood_level,
                  "quantile_level": args.quantile_level,
                  "radius": args.radius,
                  "tilt": args.tilt,
                  "softad_level": args.softad_level,
                  "sigma": args.sigma,
                  "eta": args.eta}
    loss_fn = get_loss(loss_name=args.loss,
                       loss_paras=loss_paras,
                       device=device)
    print(loss_fn)
    print("====================")
    
    # Set up the optimizer.
    print("== Optimizer: ==")
    if args.method in ["CVaR", "DRO"]:
        extra_paras = True
    else:
        extra_paras = False
    opt_paras = {"momentum": args.momentum,
                 "step_size": args.step_size,
                 "weight_decay": args.weight_decay,
                 "extra_paras": extra_paras,
                 "adaptive": args.adaptive,
                 "radius": args.radius,
                 "optimizer_base": args.optimizer_base}
    optimizer = get_optimizer(opt_name=args.optimizer,
                              opt_paras=opt_paras,
                              model=model,
                              loss_fn=loss_fn)
    print(optimizer)
    print("================")

    # Set up the step size scheduler.
    my_scheduler = SS_Scheduler(schedule_type=args.scheduler, optimizer=optimizer,
                                step_size=args.step_size, total_epochs=args.epochs)
    
    # Set up the trainer object.
    trainer = Trainer(method=args.method,
                      model=model,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      skip_singles=skip_singles,
                      num_severity_levels=args.num_severity_levels,
                      prob_update_factor=args.prob_update_factor,
                      device=device)
    
    # Execute the training loop.
    for epoch in range(-1, args.epochs):

        print("Epoch: {}".format(epoch))
        
        # Do training step, except at initial epoch.
        if epoch >= 0:
            trainer.do_training(dl_tr=dl_tr)
        
        # Check label counts and record them for later use (no training here).
        if epoch == -1:
            for data in eval_dl_tr:
                # Unpack the data properly.
                if len(data) == 2:
                    X, Y = data
                elif len(data) == 3:
                    X, Y, S = data
                else:
                    raise ValueError("Unexpected length of {} for data tuple.".format(len(data)))
                # Look at label counts.
                unique_labels, label_counts = torch.unique(
                    Y, return_counts=True
                )
                unique_labels = unique_labels.numpy(force=True)
                label_counts = label_counts.numpy(force=True)
                label_count_dict = {}
                for i, label in enumerate(unique_labels):
                    label_count_dict[str(label)] = label_counts[i]
            print("Label counts:")
            print(label_count_dict)
            mlflow.log_params(label_count_dict)

        # Save model if desired.
        if save_models:
            to_save = epoch==-1 or (epoch>0 and (epoch+1)%saving_freq == 0)
        else:
            to_save = False
        if to_save:
            fname_model = os.path.join(
                models_path, "{}_{}_{}.pth".format(args.dataset,
                                                   args.method,
                                                   saving_counter)
            )
            torch.save(model.state_dict(), fname_model)
            saving_counter += 1
        
        # Evaluation step.
        model.eval()
        with torch.no_grad():
            metrics = run_evals(
                model=model,
                data_loaders=(eval_dl_tr, eval_dl_va, eval_dl_te),
                loss_name=args.loss,
                loss_paras=loss_paras
            )
        if gradnorm:
            gradnorm_metrics = run_gradnorm_evals(
                model=model,
                optimizer=optimizer,
                data_loaders=(eval_dl_tr, eval_dl_va, eval_dl_te),
                loss_name=args.loss,
                loss_paras=loss_paras
            )
            metrics.update(gradnorm_metrics)
        
        # Log the metrics of interest.
        mlflow.log_metrics(step=epoch+1, metrics=metrics)

        # Finally, update step size if desired.
        my_scheduler(epoch=epoch)

    # Finished.
    return None


if __name__ == "__main__":
    args = get_args()
    main(args)


###############################################################################
