class WarmupScheduler:
    """ Learning rate scheduler with warm-up period """
    def __init__(self, optimizer, warmup_epochs, initial_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        # Set initial learning rate to initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.initial_lr
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # Linear warm-up: reach target_lr at the end of warm-up
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * ((self.current_epoch + 1) / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
        
    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']