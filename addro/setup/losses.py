'''Setup: initialize and pass the desired loss function.'''

# External modules.
import logging
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss, BCELoss, L1Loss, Parameter
from torch.nn.functional import relu

# Internal modules.
from setup.sunhuber import rho_torch


###############################################################################


# Here we define some customized loss classes.


class Loss_SoftFloodedOCElike(Module):
    '''
    Loss class for some (dual-form) OCE-like learning criteria, combined
    with a (potentially softened) Flooding-like threshold.
    '''
    def __init__(self, device, crit_name: str, crit_paras: dict,
                 loss_name: str, loss_paras: dict,
                 theta_init=0.0, tol=1e-4, max_iter=1000,
                 soft_variant=False, solve_internally=False):
        super().__init__()
        self.crit_name = crit_name
        self.crit_paras = crit_paras
        self.tol = tol
        self.max_iter = max_iter
        self.soft_variant = soft_variant
        self.solve_internally = solve_internally
        if self.soft_variant:
            # soft ascent-descent variant
            self.dispersion = lambda x, thres : rho_torch(x-thres)
            self.thres = self.crit_paras["softad_level"]
        else:
            # flooding variant
            self.dispersion = lambda x, thres : (x-thres).abs()
            self.thres = self.crit_paras["flood_level"]
        if self.crit_name == "CVaR":
            qlevel = self.crit_paras["quantile_level"]
            if qlevel == 0.0:
                self.theta = theta_init # irrelevant in this case
            elif qlevel > 0.0:
                if self.solve_internally:
                    self.theta = None
                else:
                    self.theta = Parameter(data=Tensor([theta_init])).to(device)
            else:
                raise ValueError("F-CVaR only works for non-negative qlevel.")
        elif self.crit_name == "DRO":
            radius = self.crit_paras["radius"]
            if radius == 0.0:
                self.theta = theta_init # irrelevant in this case
            elif radius > 0.0:
                if self.solve_internally:
                    self.theta = None
                else:
                    self.theta = Parameter(data=Tensor([theta_init])).to(device)
            else:
                raise ValueError("F-DRO only works for non-negative radius.")
        else:
            raise ValueError("Please pass a recognized criterion name.")
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    

    def bisection(self, theta_min, theta_max, f, tol=1e-6, max_iter=500):
        '''
        Helper function based on bisection function of Levy et al.
        REF: https://github.com/daniellevy/fast-dro/blob/main/robust_losses.py
        '''
        lower = f(theta_min)
        upper = f(theta_max)
    
        # until the root is between theta_min and theta_max, double the length of the 
        # interval starting at either endpoint.
        while lower > 0 or upper < 0:
            length = theta_max - theta_min
            if lower > 0:
                theta_max = theta_min
                theta_min = theta_min - 2 * length
            if upper < 0:
                theta_min = theta_max
                theta_max = theta_max + 2 * length
    
            lower = f(theta_min)
            upper = f(theta_max)
    
        for i in range(max_iter):
            theta = 0.5 * (theta_min + theta_max)
    
            v = f(theta)
    
            if torch.abs(v) <= tol:
                return theta
    
            if v > 0:
                theta_max = theta
            elif v < 0:
                theta_min = theta
    
        # if the minimum is not reached in max_iter, returns the current value
        logging.warning("Maximum number of iterations exceeded in bisection")
        return 0.5 * (theta_min + theta_max)
    
    
    def forward(self, input: Tensor, target: Tensor):
        
        # Compute individual losses.
        loss = self.loss_fn(input=input, target=target)
        
        # Process losses based on specified OCE-like criterion.
        if self.crit_name == "CVaR":
            qlevel = self.crit_paras["quantile_level"]
            if qlevel == 0.0:
                return self.flood_wrapper(x=loss.mean(), fl=self.flood_level)
            elif qlevel > 0.0:
                if self.solve_internally:
                    theta_star = torch.quantile(loss, q=qlevel, interpolation="lower")
                    return self.dispersion(x=theta_star+relu(loss-theta_star).mean()/(1.0-qlevel),
                                           thres=self.thres) + self.thres
                else:
                    return self.dispersion(x=self.theta+relu(loss-self.theta).mean()/(1.0-qlevel),
                                           thres=self.thres) + self.thres
            else:
                raise ValueError("F-CVaR only works for non-negative qlevel.")
        elif self.crit_name == "DRO":
            radius = self.crit_paras["radius"]
            if radius == 0.0:
                return self.dispersion(x=loss.mean(), thres=self.thres) + self.thres
            elif radius > 0.0:
                if self.solve_internally:
                    def p(theta):
                        pp = torch.relu(loss - theta)
                        return pp / pp.sum()
                    def bisection_target(theta):
                        pp = p(theta)
                        w = loss.shape[0] * pp - torch.ones_like(pp)
                        return 0.5 * torch.mean(w ** 2) - radius
                    theta_min = -(1.0 / (np.sqrt(2 * radius + 1) - 1)) * loss.max()
                    theta_max = loss.max()
                    #print("Getting theta_star...")
                    theta_star = self.bisection(theta_min, theta_max, bisection_target,
                                                tol=self.tol, max_iter=self.max_iter)
                    #print("Got theta_star!")
                    if self.soft_variant:
                        sqd = (1+2*radius)*(relu(loss-theta_star)**2) # no mean; modified losses
                        return self.dispersion(x=theta_star+torch.sqrt(sqd), thres=self.thres).mean() + self.thres
                    else:
                        sqd = (1+2*radius)*(relu(loss-theta_star)**2).mean() # take mean first
                        return self.dispersion(x=theta_star+torch.sqrt(sqd), thres=self.thres) + self.thres
                else:
                    if self.soft_variant:
                        sqd = (1+2*radius)*(relu(loss-self.theta)**2) # no mean; modified losses
                        return self.dispersion(x=self.theta+torch.sqrt(sqd), thres=self.thres).mean() + self.thres
                    else:
                        sqd = (1+2*radius)*(relu(loss-self.theta)**2).mean() # take mean first
                        return self.dispersion(x=self.theta+torch.sqrt(sqd), thres=self.thres) + self.thres
            else:
                raise ValueError("F-DRO only works for non-negative radius.")
        else:
            raise ValueError("Please pass a recognized criterion name.")


class Loss_OCElike(Module):
    '''
    Loss class for some (dual-form) OCE-like criteria.
    REF: https://github.com/daniellevy/fast-dro/blob/main/robust_losses.py
    '''
    def __init__(self, crit_name: str, crit_paras: dict,
                 loss_name: str, loss_paras: dict,
                 device, theta_init=0.0):
        super().__init__()
        self.crit_name = crit_name
        self.crit_paras = crit_paras
        if self.crit_name == "CVaR":
            qlevel = self.crit_paras["quantile_level"]
            if qlevel == 0.0:
                self.theta = theta_init # irrelevant in this case
            elif qlevel > 0.0:
                self.theta = Parameter(data=Tensor([theta_init])).to(device)
            else:
                raise ValueError("CVaR only works for non-negative qlevel.")
        elif self.crit_name == "DRO":
            radius = self.crit_paras["radius"]
            if radius == 0.0:
                self.theta = theta_init # irrelevant in this case
            elif radius > 0.0:
                self.theta = Parameter(data=Tensor([theta_init])).to(device)
            else:
                raise ValueError("DRO only works for non-negative radius.")
        else:
            raise ValueError("Please pass a recognized criterion name.")
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None

    def forward(self, input: Tensor, target: Tensor):
        
        # Compute individual losses.
        loss = self.loss_fn(input=input, target=target)
        
        # Process losses based on specified OCE-like criterion.
        if self.crit_name == "CVaR":
            qlevel = self.crit_paras["quantile_level"]
            if qlevel == 0.0:
                return loss.mean()
            elif qlevel > 0.0:
                return self.theta + relu(loss-self.theta).mean()/(1.0-qlevel)
            else:
                raise ValueError("CVaR only works for non-negative qlevel.")
        elif self.crit_name == "DRO":
            radius = self.crit_paras["radius"]
            if radius == 0.0:
                return loss.mean()
            elif radius > 0.0:
                sqd = (1+2*radius)*(relu(loss-self.theta)**2).mean()
                return self.theta + torch.sqrt(sqd)
            else:
                raise ValueError("DRO only works for non-negative radius.")
        else:
            raise ValueError("Please pass a recognized criterion name.")


class Loss_Tilted(Module):
    '''
    Loss class for tilted ERM.
    '''
    def __init__(self, tilt: float, loss_name: str, loss_paras: dict):
        super().__init__()
        self.tilt = tilt
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    
    def forward(self, input: Tensor, target: Tensor):
        loss = self.loss_fn(input=input, target=target)
        if self.tilt > 0.0:
            return torch.log(torch.exp(self.tilt*loss).mean()) / self.tilt
        elif self.tilt == 0.0:
            return loss.mean()
        else:
            raise ValueError("Only defined for non-negative tilt values.")


class Loss_Flood(Module):
    '''
    General purpose loss class for the "flooding" algorithm
    of Ishida et al. (2020).
    '''
    def __init__(self, flood_level: float, loss_name: str, loss_paras: dict):
        super().__init__()
        self.flood_level = flood_level
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="mean")
        return None

    def forward(self, input: Tensor, target: Tensor):
        fl = self.flood_level
        loss = self.loss_fn(input=input, target=target)
        return (loss-fl).abs()+fl


class Loss_SoftAD(Module):
    '''
    Soft ascent-descent (SoftAD), our most basic modified version of
    the flooding algorithm.
    '''
    def __init__(self, theta: float, sigma: float, eta: float,
                 loss_name: str, loss_paras: dict):
        super().__init__()
        self.theta = theta
        self.sigma = sigma
        self.eta = eta
        self.loss_fn = get_named_loss(loss_name=loss_name,
                                      loss_paras=loss_paras,
                                      reduction="none")
        return None
    
    def forward(self, input: Tensor, target: Tensor):
        theta = self.theta
        sigma = self.sigma + 1e-12 # to be safe.
        eta = self.eta
        loss = self.loss_fn(input=input, target=target)
        dispersion = (sigma**2) * rho_torch((loss-theta)/sigma).mean()
        return eta*theta + dispersion


# Here we define various loss function getters.

def get_loss(loss_name, loss_paras, device):
    '''
    Loss function getter for all methods.
    '''
    ln = loss_name
    lp = loss_paras

    softflood_cvar = ["FloodedCVaR", "AutoFloodedCVaR", "AutoSoftFloodedCVaR", "SoftFloodedCVaR"]
    softflood_dro = ["FloodedDRO", "AutoFloodedDRO", "AutoSoftFloodedDRO", "SoftFloodedDRO"]
    softflood_all = softflood_cvar + softflood_dro
    
    if lp["method"] == "ERM":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="mean")
    elif lp["method"] == "Ishida":
        loss_fn = get_flood_loss(loss_name=ln,
                                 loss_paras=lp)
    elif lp["method"] in ["CVaR", "DRO"]:
        lp["crit_name"] = lp["method"]
        loss_fn = get_ocelike_loss(loss_name=ln,
                                   loss_paras=lp,
                                   device=device)
    elif lp["method"] in softflood_all:
        if lp["method"] in softflood_cvar:
            lp["crit_name"] = "CVaR"
        elif lp["method"] in softflood_dro:
            lp["crit_name"] = "DRO"
        else:
            raise ValueError("Provide a proper method name.")
        loss_fn = get_softfloodedocelike_loss(loss_name=ln,
                                              loss_paras=lp,
                                              device=device)
    elif lp["method"] == "SAM":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="mean")
    elif lp["method"] == "SharpDRO":
        loss_fn = get_named_loss(loss_name=ln,
                                 loss_paras=lp,
                                 reduction="none") # no reduction.
    elif lp["method"] == "SoftAD":
        loss_fn = get_softad_loss(loss_name=ln,
                                  loss_paras=lp)
    elif lp["method"] == "Tilted":
        loss_fn = get_tilted_loss(loss_name=ln,
                                  loss_paras=lp)
    else:
        raise ValueError("Unrecognized method name.")
    
    return loss_fn


def get_flood_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_Flood.
    '''
    fl = loss_paras["flood_level"]
    loss_fn = Loss_Flood(flood_level=fl,
                         loss_name=loss_name,
                         loss_paras=loss_paras)
    return loss_fn


def get_softad_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_SoftAD.
    '''
    eta = loss_paras["eta"]
    sigma = loss_paras["sigma"]
    theta = loss_paras["theta"]
    loss_fn = Loss_SoftAD(theta=theta, sigma=sigma, eta=eta,
                          loss_name=loss_name, loss_paras={})
    return loss_fn


def get_ocelike_loss(loss_name, loss_paras, device):
    '''
    A simple wrapper for Loss_OCElike.
    '''
    crit_name = loss_paras["crit_name"]
    crit_paras = {"quantile_level": loss_paras["quantile_level"],
                  "radius": loss_paras["radius"]}
    loss_fn = Loss_OCElike(crit_name=crit_name, crit_paras=crit_paras,
                           loss_name=loss_name, loss_paras={},
                           theta_init=0.0, device=device)
    return loss_fn


def get_softfloodedocelike_loss(loss_name, loss_paras, device):
    '''
    A wrapper for Loss_SoftFloodedOCElike.
    '''
    crit_name = loss_paras["crit_name"]
    crit_paras = {"quantile_level": loss_paras["quantile_level"],
                  "softad_level": loss_paras["softad_level"],
                  "flood_level": loss_paras["flood_level"],
                  "radius": loss_paras["radius"]}
    if loss_paras["method"] in ["FloodedCVaR", "FloodedDRO"]:
        soft_variant = False
        solve_internally = False
    elif loss_paras["method"] in ["SoftFloodedCVaR", "SoftFloodedDRO"]:
        soft_variant = True
        solve_internally = False
    elif loss_paras["method"] in ["AutoSoftFloodedCVaR", "AutoSoftFloodedDRO"]:
        soft_variant = True
        solve_internally = True
    elif loss_paras["method"] in ["AutoFloodedCVaR", "AutoFloodedDRO"]:
        soft_variant = False
        solve_internally = True
    else:
        raise ValueError("Please pass a known criterion name.")
    
    loss_fn = Loss_SoftFloodedOCElike(device=device,
                                      crit_name=crit_name,
                                      crit_paras=crit_paras,
                                      loss_name=loss_name,
                                      loss_paras={},
                                      theta_init=0.0, # simple initial value
                                      tol=1e-2,
                                      max_iter=1000,
                                      soft_variant=soft_variant,
                                      solve_internally=solve_internally)
    
    return loss_fn


def get_tilted_loss(loss_name, loss_paras):
    '''
    A simple wrapper for Loss_Tilted.
    '''
    loss_fn = Loss_Tilted(tilt=loss_paras["tilt"],
                          loss_name=loss_name, loss_paras={})
    return loss_fn


def get_named_loss(loss_name, loss_paras, reduction):
    
    if loss_name == "CrossEntropy":
        loss_fn = CrossEntropyLoss(reduction=reduction)
    elif loss_name == "BCELoss":
        loss_fn = BCELoss(reduction=reduction)
    elif loss_name == "L1Loss":
        loss_fn = L1Loss(reduction=reduction)
    else:
        raise ValueError("Unrecognized loss name.")
    
    return loss_fn


###############################################################################
