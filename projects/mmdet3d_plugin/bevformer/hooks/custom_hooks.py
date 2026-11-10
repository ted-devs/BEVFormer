from mmcv.runner.hooks.hook import HOOKS, Hook
from projects.mmdet3d_plugin.models.utils import run_time


@HOOKS.register_module()
class TransferWeight(Hook):
    
    def __init__(self, every_n_inters=1):
        self.every_n_inters = every_n_inters
        self._param_buffer_pairs = None

    def _init_pairs(self, runner):
        eval_tensors = dict(runner.eval_model.named_parameters())
        eval_tensors.update(dict(runner.eval_model.named_buffers()))
        pairs = []
        for name, param in runner.model.named_parameters():
            if name in eval_tensors:
                pairs.append((eval_tensors[name], param))
        for name, buffer in runner.model.named_buffers():
            if name in eval_tensors:
                pairs.append((eval_tensors[name], buffer))
        self._param_buffer_pairs = pairs

    def after_train_iter(self, runner):
        if self.every_n_inner_iters(runner, self.every_n_inters):
            if self._param_buffer_pairs is None:
                self._init_pairs(runner)
            for eval_tensor, train_tensor in self._param_buffer_pairs:
                eval_tensor.data.copy_(train_tensor.data)
