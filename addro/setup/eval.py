'''Setup: design of the basic evaluation routines used.'''

# External modules.
import numpy as np
import torch

# Internal modules.
from setup.losses import get_named_loss
from setup.utils import compute_grad_norm


###############################################################################


def eval_model_norm(model):
    '''
    Compute the l2 norm of the (concatenated) model parameters.
    '''
    norms = []
    for param in model.parameters():
        if param is None:
            continue
        else:
            norms += [torch.linalg.vector_norm(x=param, ord=2)]
    if len(norms) > 0:
        return torch.linalg.vector_norm(x=torch.stack(norms), ord=2).item()
    else:
        raise ValueError("No parameters for which to compute norm for.")


def eval_loss_by_sev(model, data_loader, loss_fn):
    '''
    Performance evaluation on data specified by a data loader,
    in terms of average loss for each severity level.
    
    Key points:
    - Assumes the loss_fn passed does *not* reduce the losses
      by default. These need to be averaged per-severity.
    - We are assuming that the data being loaded includes data
      from all possible severity levels.
    '''

    batch_count = len(data_loader)
    
    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    
    else:

        ave_losses = []
        
        for data in data_loader:
            # Unpack the data correctly.
            if len(data) == 3:
                X, Y, S = data
            else:
                s_err = "Per-severity eval requires triplet (X, Y, S)."
                raise ValueError(s_err)
            # Compute outputs, then compute per-severity ave loss.
            Y_hat = model(X)
            losses = loss_fn(Y_hat, Y)
            severities_seen = torch.unique(S, sorted=True)
            for severity in severities_seen:
                s_idx = S == severity
                ave_losses += [torch.mean(losses[s_idx]).item()]
        
        return ave_losses


def eval_acc_by_sev(model, data_loader):
    '''
    Performance evaluation on data specified by a data loader,
    in terms of accuracy for each severity level.
    
    Key points:
    - Assume the data loaders are *full batch*, i.e., no need to
      use any sum-reducing losses (default mean is fine).
    - Accuracy calculations assume model(X) yields vector
      outputs with length equal to the number of classes.
    - We are assuming that the data being loaded includes data
      from all possible severity levels.
    '''

    batch_count = len(data_loader)
    
    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    
    else:

        accuracies = []
        
        for data in data_loader:
            # Unpack the data correctly.
            if len(data) == 3:
                X, Y, S = data
            else:
                s_err = "Per-severity eval requires triplet (X, Y, S)."
                raise ValueError(s_err)
            # Compute outputs, then compute per-severity accuracies.
            Y_hat = model(X)
            severities_seen = torch.unique(S, sorted=True)
            for severity in severities_seen:
                s_idx = S == severity
                zeroones = torch.where(
                    torch.argmax(Y_hat[s_idx], 1) == Y[s_idx], 1.0, 0.0
                )
                accuracies += [torch.mean(zeroones).item()]
        
        return accuracies


def eval_acc_by_class(model, data_loader):
    '''
    Performance evaluation on data specified by a data loader,
    in terms of accuracy for each class.
    
    Key points:
    - Assume the data loaders are *full batch*, i.e., no need to
      use any sum-reducing losses (default mean is fine).
    - Accuracy calculations assume model(X) yields vector
      outputs with length equal to the number of classes.
    - We have paid special attention to cover the possibility that
      a certain class is not included in a given data set. In this
      case, we simply record "NaN" using np.nan.
    '''

    batch_count = len(data_loader)
    
    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    
    else:

        accuracies = []
        
        for data in data_loader:
            # Unpack the data correctly.
            if len(data) == 2:
                X, Y = data
            elif len(data) == 3:
                X, Y, S = data
            else:
                s_err = "Data tuple length {} is wrong.".format(len(data))
                raise ValueError(s_err)
            # Compute outputs, then compute per-class accuracies.
            Y_hat = model(X)
            num_classes = Y_hat.size(dim=1)
            classes_seen = torch.unique(Y, sorted=True)
            num_seen = len(classes_seen)
            if num_classes == num_seen:
                for c in classes_seen:
                    c_idx = Y == c
                    zeroones = torch.where(
                        torch.argmax(Y_hat[c_idx], 1) == Y[c_idx], 1.0, 0.0
                    )
                    accuracies += [torch.mean(zeroones).item()]
            elif num_classes > num_seen:
                seen_counter = 0
                for i in range(num_classes):
                    if seen_counter == num_seen:
                        # If we've seen all labels, can only be NaN.
                        accuracies += [np.nan]
                    elif i == classes_seen[seen_counter].item():
                        c_idx = Y == classes_seen[seen_counter]
                        zeroones = torch.where(
                            torch.argmax(Y_hat[c_idx], 1) == Y[c_idx], 1.0, 0.0
                        )
                        accuracies += [torch.mean(zeroones).item()]
                        seen_counter += 1
                    else:
                        # If ith label is not "i", record NaN.
                        accuracies += [np.nan]
            else:
                raise ValueError("num_classes is less than num_seen.")
        
        return accuracies


def eval_balanced_error(model, data_loader, single_prob=True):
    '''
    Balanced classification error.
    '''
    
    batch_count = len(data_loader)

    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    else:
        for X, Y in data_loader:
            if single_prob:
                P_hat = model(X) # assumes a single prob of class 1
                Y_hat = torch.where(P_hat >= 0.5, 1, 0)
                idx_0 = Y == 0
                idx_1 = Y == 1
                err_0 = torch.where(
                    Y_hat[idx_0] != Y[idx_0], 1.0, 0.0
                ).mean().item()
                err_1 = torch.where(
                    Y_hat[idx_1] != Y[idx_1], 1.0, 0.0
                ).mean().item()
                balanced_error = (err_0+err_1)/2.0
            else:
                # assumes scores/probs for each class
                raise NotImplementedError()
        
        return balanced_error


def eval_loss_acc(model, data_loader, loss_fn):
    '''
    Performance evaluation on data specified by a data loader,
    in terms of a loss function specified by the user, and
    typical classification accuracy.

    Key points:
    - Assume the data loaders are *full batch*, i.e., no need to
      use any sum-reducing losses (default mean is fine).
    - Accuracy calculations assume model(X) yields vector
      outputs with length equal to the number of classes.
    '''
    
    batch_count = len(data_loader)

    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    
    else:
        for data in data_loader:
            # Unpack the data correctly.
            if len(data) == 2:
                X, Y = data
            elif len(data) == 3:
                X, Y, S = data
            else:
                raise ValueError("Unexpected length of {} for data tuple.".format(len(data)))
            # Compute loss and accuracy.
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y).item()
            zeroones = torch.where(
                torch.argmax(Y_hat, 1) == Y, 1.0, 0.0
            )
            accuracy = torch.mean(zeroones).item()
    
        return (loss, accuracy)


def eval_gradnorm(model, optimizer, data_loader, loss_fn):
    '''
    Evaluation of the l2 norm of the ave loss gradient, on data
    specified by a data loader, with user-specified loss. To ensure
    we get gradients for the right parameters, we take optimizer as
    an argument as well. This also helps with zeroing things out,
    before and after, just to be safe.
    
    Key points:
    - Assume the data loaders are *full batch*, i.e., no need to
      use any sum-reducing losses (default mean is fine).
    - Norm is computed *after* averaging gradient vectors.
    '''
    
    batch_count = len(data_loader)

    if batch_count != 1:
        raise ValueError("Batch count for eval is not 1. Full batch please!")
    
    else:
        for data in data_loader:
            # Unpack the data correctly.
            if len(data) == 2:
                X, Y = data
            elif len(data) == 3:
                X, Y, S = data
            else:
                raise ValueError("Unexpected length of {} for data tuple.".format(len(data)))
            # Compute losses.
            Y_hat = model(X)
            loss = loss_fn(Y_hat, Y)
            # Clear old gradients just in case.
            optimizer.zero_grad()
            # Compute current gradient.
            loss.backward()
            # Compute gradient norm (same as in SAM implementation).
            dev = optimizer.param_groups[0]["params"][0].device
            norm = compute_grad_norm(
                param_groups=optimizer.param_groups, dev=dev
            ).item()
            # Once again clear out old gradients once done.
            optimizer.zero_grad()
        return norm


def run_gradnorm_evals(model, optimizer, data_loaders, loss_name, loss_paras):
    '''
    Run our gradient norm computations. Structure is essentially
    identical to "run_evals", but this function is used outside of
    the no_grad context manager (since we need grads).
    '''

    eval_dl_tr, eval_dl_va, eval_dl_te = data_loaders
    
    loss_fn = get_named_loss(loss_name=loss_name,
                             loss_paras=loss_paras,
                             reduction="mean")

    gradnorm_tr = eval_gradnorm(model=model,
                                optimizer=optimizer,
                                data_loader=eval_dl_tr,
                                loss_fn=loss_fn)
    gradnorm_va = eval_gradnorm(model=model,
                                optimizer=optimizer,
                                data_loader=eval_dl_va,
                                loss_fn=loss_fn)
    gradnorm_te = eval_gradnorm(model=model,
                                optimizer=optimizer,
                                data_loader=eval_dl_te,
                                loss_fn=loss_fn)

    metrics = {"gradnorm_tr": gradnorm_tr,
               "gradnorm_va": gradnorm_va,
               "gradnorm_te": gradnorm_te}
    
    return metrics


def run_evals(model, data_loaders, loss_name, loss_paras):
    
    eval_dl_tr, eval_dl_va, eval_dl_te = data_loaders
    
    loss_fn = get_named_loss(loss_name=loss_name,
                             loss_paras=loss_paras,
                             reduction="mean")
    loss_presum_fn = get_named_loss(loss_name=loss_name,
                                    loss_paras=loss_paras,
                                    reduction="none")

    # Base loss and overall accuracy.
    loss_tr, acc_tr = eval_loss_acc(model=model,
                                    data_loader=eval_dl_tr,
                                    loss_fn=loss_fn)
    loss_va, acc_va = eval_loss_acc(model=model,
                                    data_loader=eval_dl_va,
                                    loss_fn=loss_fn)
    loss_te, acc_te = eval_loss_acc(model=model,
                                    data_loader=eval_dl_te,
                                    loss_fn=loss_fn)

    # Model norm.
    model_norm = eval_model_norm(model=model)
    
    # Per-severity losses.
    loss_by_sev_tr = eval_loss_by_sev(model=model,
                                      data_loader=eval_dl_tr,
                                      loss_fn=loss_presum_fn)
    loss_by_sev_va = eval_loss_by_sev(model=model,
                                      data_loader=eval_dl_va,
                                      loss_fn=loss_presum_fn)
    loss_by_sev_te = eval_loss_by_sev(model=model,
                                      data_loader=eval_dl_te,
                                      loss_fn=loss_presum_fn)
    
    # Per-severity accuracies.
    acc_by_sev_tr = eval_acc_by_sev(model=model, data_loader=eval_dl_tr)
    acc_by_sev_va = eval_acc_by_sev(model=model, data_loader=eval_dl_va)
    acc_by_sev_te = eval_acc_by_sev(model=model, data_loader=eval_dl_te)

    # Per-class accuracies.
    acc_by_class_tr = eval_acc_by_class(model=model, data_loader=eval_dl_tr)
    acc_by_class_va = eval_acc_by_class(model=model, data_loader=eval_dl_va)
    acc_by_class_te = eval_acc_by_class(model=model, data_loader=eval_dl_te)

    # Per-severity test accuracies.
    acc_by_sev_te = eval_acc_by_sev(model=model, data_loader=eval_dl_te)

    # Prepare the metric dictionary to be output.
    metrics = {"loss_tr": loss_tr,
               "loss_va": loss_va,
               "loss_te": loss_te,
               "acc_tr": acc_tr,
               "acc_va": acc_va,
               "acc_te": acc_te,
               "model_norm": model_norm}
    for c in range(len(acc_by_class_tr)):
        metrics["abc_{}_tr".format(c)] = acc_by_class_tr[c]
        metrics["abc_{}_va".format(c)] = acc_by_class_va[c]
        metrics["abc_{}_te".format(c)] = acc_by_class_te[c]
    for s in range(len(loss_by_sev_tr)):
        metrics["lbs_{}_tr".format(s)] = loss_by_sev_tr[s]
    for s in range(len(loss_by_sev_va)):
        metrics["lbs_{}_va".format(s)] = loss_by_sev_va[s]
    for s in range(len(loss_by_sev_te)):
        metrics["lbs_{}_te".format(s)] = loss_by_sev_te[s]
    for s in range(len(acc_by_sev_te)):
        metrics["abs_{}_te".format(s)] = acc_by_sev_te[s]
    
    return metrics


###############################################################################
