'''Step size reduction schedulers, following the code of Huang et al. (2023).'''

# External modules.

# Internal modules.


###############################################################################


# References for this code:
# https://github.com/zhuohuangai/SharpDRO/blob/main/train.py
# https://github.com/zhuohuangai/SharpDRO/blob/main/step_lr.py

class SS_Scheduler:
    def __init__(self, schedule_type, optimizer, step_size: float, total_epochs: int):
        self.schedule_type = schedule_type
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.base = step_size
        return None

    def __call__(self, epoch):
        if self.schedule_type == "sharpdro":
            if epoch < self.total_epochs * 3/10:
                new_step_size = self.base
            elif epoch < self.total_epochs * 6/10:
                new_step_size = self.base * 0.2
            elif epoch < self.total_epochs * 8/10:
                new_step_size = self.base * 0.2 ** 2
            else:
                new_step_size = self.base * 0.2 ** 3
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_step_size
        elif self.schedule_type == "none":
            new_step_size = self.base # constant
        else:
            raise ValueError("Please provide a valid schedule type name.")
            

        return None


###############################################################################
