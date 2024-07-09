'''Setup: initialize optimizer objects with desired settings.'''

# External modules.
import torch
from torch.optim import SGD, Adam, Optimizer

# Internal modules.
from setup.utils import compute_grad_norm


###############################################################################


# Optimizer-related parsers.

base_optimizers = ["SGD", "Adam"]

def get_base_optimizer(opt_name, opt_paras, model=None, params=None):
    '''
    Usage: pass either a model OR a parameter iterable, not both.
    [note: params can be a list of parameter groups (dicts)]
    '''
    if model is None:
        params = params
    elif params is None:
        params = model.parameters()
    else:
        raise ValueError("Pass model or params, not both.")
    
    op = opt_paras
    if opt_name == "SGD":
        optimizer = SGD(
            params,
            weight_decay=op["weight_decay"],
            lr=op["step_size"],
            momentum=op["momentum"]
        )
    elif opt_name == "Adam":
        optimizer = Adam(
            params,
            weight_decay=op["weight_decay"],
            lr=op["step_size"]
        )
    else:
        raise ValueError("Unrecognized optimizer name.")
    
    return optimizer


def get_optimizer(opt_name, opt_paras, model, loss_fn):
    op = opt_paras

    if op["extra_paras"]:
        # In this case, we use extra paras attached to loss_fn.
        if opt_name in base_optimizers:
            param_groups = [{"params": model.parameters()},
                            {"params": loss_fn.parameters()}]
            # [note: for now, use same opt paras for both param groups,
            #  but it is just a matter of modifying the dicts above if
            #  we want to have different opt paras for each group.]
            # REF: https://github.com/daniellevy/fast-dro/blob/main/train.py
            optimizer = get_base_optimizer(opt_name=opt_name,
                                           opt_paras=opt_paras,
                                           params=param_groups)
        else:
            raise NotImplementedError("Only works for base optimizers.")

    else:
        if opt_name in base_optimizers:
            optimizer = get_base_optimizer(opt_name=opt_name,
                                           opt_paras=opt_paras,
                                           model=model)
        elif opt_name == "SAM":
            opt_base_name = op.pop("optimizer_base")
            radius = op.pop("radius")
            adaptive = op.pop("adaptive")
            optimizer = SAM_Generalized(
                params=model.parameters(),
                opt_base_name=opt_base_name,
                opt_base_paras=op,
                radius=radius,
                adaptive=adaptive,
                sharp_dro_style=False # do not use SharpDRO-style update.
            )
        elif opt_name == "SharpDRO":
            opt_base_name = op.pop("optimizer_base")
            radius = op.pop("radius")
            adaptive = op.pop("adaptive")
            optimizer = SAM_Generalized(
                params=model.parameters(),
                opt_base_name=opt_base_name,
                opt_base_paras=op,
                radius=radius,
                adaptive=adaptive,
                sharp_dro_style=True # use SharpDRO-style update.
            )
        else:
            raise ValueError("Unrecognized optimizer name.")
    
    return optimizer


def get_constructor(opt_name):
    if opt_name == "SGD":
        return SGD
    elif opt_name == "Adam":
        return Adam
    else:
        raise ValueError("Unrecognized optimizer name.")


# Specialized optimizer implementations.

class SAM_Generalized(Optimizer):
    '''
    We use this class to implement the methods from the following
    three methods:
    - SAM (Foret et al., 2021)
    - Adaptive SAM (Kwon et al., 2021)
    - SharpDRO (Huang et al., 2023)
    Key references are as follows.
    - https://github.com/davda54/sam/blob/main/sam.py
    - https://github.com/zhuohuangai/SharpDRO/blob/main/sam.py
    The first link is the representative PyTorch implementation of
    SAM (both original and adaptive variant), and is cited in the original
    SAM paper. The second link is the analogous implementation for SharpDRO.

    - params: model parameters to be optimized.
    - opt_base_name: name for base optimizer.
    - radius: gradient norm size is fixed to this.
    - adaptive: "yes" to enable ASAM of Kwon et al. (2021).
    - opt_base_paras: options to be sent to the base optimizer constructor.
    - sharp_dro_style: True or False, for using the SharpDRO style of implementation.
    '''
    
    def __init__(self, params, opt_base_name,
                 opt_base_paras, radius, adaptive,
                 sharp_dro_style=False):
        
        # Convert to boolean for convenience.
        self.adaptive = True if adaptive == "yes" else False
        
        # The usual name for optimizer option dict is "defaults".
        defaults = {
            "radius": radius,
            "adaptive": self.adaptive,
            **opt_base_paras
        }

        # Boolean controlling whether we use the SharpDRO-style or not.
        self.sharp_dro_style = sharp_dro_style
        
        # Initialize as vanilla Optimizer object with param_groups attribute.
        # Note that param_groups is always a list of dictionaries, and
        # Tensor-like Parameter objects to be optimized are stored (as lists)
        # under the key "params" within each param_group dict; the other
        # options are stored in each dict with their keys as-is.
        super().__init__(params, defaults)
        
        # Construct the base optimizer. It is okay to pass a list of typical
        # Parameter objects OR a list of "parameter groups" (dicts), and since
        # the preceding call of super() ensures we have such groups, it is
        # natural to pass this along.
        self.optimizer_base = get_base_optimizer(opt_name=opt_base_name,
                                                 opt_paras=opt_base_paras,
                                                 params=self.param_groups)
        
        # Update in case any new parameter groups have been added by the
        # base optimizer constructor.
        self.param_groups = self.optimizer_base.param_groups
        
        # Update options to include default values used by the base constructor
        # that were not specified using base_paras.
        self.defaults.update(self.optimizer_base.defaults)
        
        return None
    
    
    def first_step(self):
        '''
        Store the old parameter, and compute the perturbed
        variant which will be used in second step update.
        '''
        grad_norm = self.grad_norm()
        
        for param_group in self.param_groups:

            scale = param_group["radius"] / (grad_norm + 1e-12)

            for param in param_group["params"]:

                if param.grad is None:
                    # If has no gradient, there is nothing to do.
                    continue
                else:
                    # Use the handy state defaultdict to store old param.
                    self.state[param]["old_param"] = param.data.clone()
                    if self.sharp_dro_style:
                        self.state[param]["old_grad"] = param.grad.clone()

                    # Update the parameter by perturbation.
                    if param_group["adaptive"]:
                        # Parameter-wise adaptation if desired.
                        adapter = torch.pow(param, 2)
                        param.add_(param.grad*adapter, alpha=scale)
                    else:
                        param.add_(param.grad, alpha=scale)
                        #param.add_(param.grad*scale.to(param)) # alternative
        return None
    
    
    def second_step(self):
        '''
        After having run the first step and re-evaluating the
        objective function to get a fresh set of gradients at
        the perturbed point, we just need to use this to update.
        '''
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.grad is None:
                    continue
                else:
                    # Go back to initial point for updating.
                    param.data = self.state[param]["old_param"]
                    if self.sharp_dro_style:
                        # Doing this ensures grad is now of *sum* of ave loss and sharpness term.
                        # Huang et al. (2023) comment on this in section B.1 ("Practical Implementation").
                        param.grad.add_(self.state[param]["old_grad"])
        
        # Base optimizer update, noting grads are at the perturbed point.
        self.optimizer_base.step()
        
        return None

    
    def step(self, closure=None):
        raise NotImplementedError
    
    
    def grad_norm(self):

        # Put everything on same device in case of model parallelism.
        dev = self.param_groups[0]["params"][0].device

        # Get l2 norm for concatenated parameters.
        norm = compute_grad_norm(param_groups=self.param_groups, dev=dev)
        return norm
    
    
    def load_state_dict(self, state_dict):
        '''
        Exactly as defined in davda54's SAM repository.
        '''
        super().load_state_dict(state_dict)
        self.optimizer_base.param_groups = self.param_groups


###############################################################################
