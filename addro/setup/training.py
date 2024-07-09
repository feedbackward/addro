'''Setup: training routines specialized for different methods.'''

# External modules.
import torch


###############################################################################


class Trainer:

    def __init__(self, method, model, optimizer, loss_fn, skip_singles, num_severity_levels, prob_update_factor, device):

        self.method = method
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.skip_singles = skip_singles
        
        if self.method == "SharpDRO":
            self.num_severity_levels = num_severity_levels
            self.num_distributions = self.num_severity_levels+1
            self.adv_probs = torch.ones(self.num_distributions).to(device) / self.num_distributions
            self.puf = prob_update_factor
        else:
            self.num_distributions = None
            self.adv_probs = None
            self.puf = None
        
        return None
    
    
    def do_training(self, dl_tr):
        '''
        A single training pass over a data loader.
        '''
        
        self.model.train()

        if self.method == "SAM":
            for data in dl_tr:
                # Unpack the data correctly.
                if len(data) == 2:
                    X, Y = data
                elif len(data) == 3:
                    X, Y, S = data
                else:
                    raise ValueError("Unexpected length of {} for data tuple.".format(len(data)))
                # Get to work on training.
                if len(X) == 1 and self.skip_singles:
                    continue
                else:
                    # First forward-backward pass.
                    Y_hat = self.model(X)
                    loss = self.loss_fn(Y_hat, Y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        self.optimizer.first_step()
                    # Second forward-backward pass.
                    Y_hat = self.model(X)
                    loss = self.loss_fn(Y_hat, Y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    with torch.no_grad():
                        self.optimizer.second_step()
        
        elif self.method == "SharpDRO":
            # Implementation here based on Huang et al. (2023) original code.
            # https://github.com/zhuohuangai/SharpDRO/blob/main/train.py
            
            for data in dl_tr:
                # Unpack the data correctly.
                if len(data) == 3:
                    X, Y, S = data
                else:
                    raise ValueError("Unexpected length of {} for data tuple.".format(len(data)))
                # Get to work on training.
                if len(X) == 1 and skip_singles:
                    continue
                else:
                    # First forward-backward pass.
                    #enable_running_stats(self.model)
                    Y_hat = self.model(X)
                    losses = self.loss_fn(Y_hat, Y) # assumes no "reduction" is done.
                    self.optimizer.zero_grad()
                    losses.mean().backward() # use average loss as objective.
                    with torch.no_grad():
                        self.optimizer.first_step()
                    # Second forward-backward pass.
                    #disable_running_stats(self.model)
                    Y_hat = self.model(X)
                    per_point_sharpness = self.loss_fn(Y_hat, Y) - losses.data # take difference from initial losses.
                    # note: using losses.data (not just losses) is critical so we don't do a double backward pass.
                    distribution_map = torch.where(
                        S == torch.arange(self.num_distributions).unsqueeze(1).to(S.device), 1.0, 0.0
                    )
                    # note: shape of distribution_map is (self.num_distributions, len(S)).
                    #print("Is shape of distribution_map equal to ({}, {})?".format(self.num_distributions, len(S)))
                    #if distribution_map.shape[0] == self.num_distributions and distribution_map.shape[1] == len(S):
                    #    print("\t ... YES.")
                    #else:
                    #    raise ValueError("\t... NO. Something is wrong.")
                    distribution_counts = distribution_map.sum(axis=1)
                    # note: counts the number of data from each distribution.
                    distribution_denom = distribution_counts + torch.where(distribution_counts==0, 1.0, 0.0)
                    # note: the term added is to avoid division by zero.
                    ave_sharpness_per_distribution = torch.matmul(distribution_map,
                                                                  per_point_sharpness) / distribution_denom
                    # note: our "ave_sharpness_per_distribution" is "distribution_sharpness" in the original GitHub repo.
                    ave_loss_per_distribution = torch.matmul(distribution_map,
                                                             losses) / distribution_denom
                    # note: our "ave_loss_per_distribution" is "distribution_loss" in the original GitHub repo.
                    self.adv_probs = self.adv_probs * torch.exp(self.puf*ave_loss_per_distribution.data)
                    self.adv_probs = self.adv_probs / self.adv_probs.sum()
                    sharpdro_objective = torch.matmul(ave_sharpness_per_distribution, self.adv_probs)
                    self.optimizer.zero_grad()
                    sharpdro_objective.backward()
                    with torch.no_grad():
                        self.optimizer.second_step()
        else:
            for data in dl_tr:
                # Unpack the data correctly.
                if len(data) == 2:
                    X, Y = data
                elif len(data) == 3:
                    X, Y, S = data
                else:
                    raise ValueError("Unexpected length of {} for data tuple.".format(len(data)))
                # Get to work on training.
                if len(X) == 1 and skip_singles:
                    continue
                else:
                    Y_hat = self.model(X)
                    loss = self.loss_fn(Y_hat, Y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
        return None


###############################################################################
