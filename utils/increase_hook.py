from mmcv.runner import HOOKS, Hook
import pdb

@HOOKS.register_module()
class Increase_Hook(Hook):
    def __init__(self):
        pass
    
    def before_run(self, runner):
        runner.model.module.iter = runner.iter
        runner.model.module._inner_iter = runner._inner_iter
        runner.model.module.epoch = runner._epoch
        runner.model.module._max_epochs = runner._max_epochs
    
    def before_iter(self, runner):
        runner.model.module.iter = runner.iter
        runner.model.module._inner_iter = runner._inner_iter

    def before_epoch(self, runner):
        runner.model.module.epoch = runner._epoch