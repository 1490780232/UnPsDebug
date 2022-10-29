import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.cuda.amp import custom_fwd, custom_bwd
# from utils.distributed import tensor_gather


class OIM(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, targets, lut, momentum):
        ctx.save_for_backward(inputs, targets, lut, momentum)
        outputs_labeled = inputs.mm(lut.t())
        return outputs_labeled
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets, lut,  momentum = ctx.saved_tensors
        # inputs, targets = tensor_gather((inputs, targets))
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(lut)
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, momentum=0.5):
    return OIM.apply(inputs, targets, lut, torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features,  num_pids, oim_momentum, oim_scalar,num_samples=0):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.num_samples = num_samples
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        projected = oim(inputs, label, self.lut, momentum=self.momentum)
        projected *= self.oim_scalar
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        return loss_oim



class OIMUnsupervisedLoss(nn.Module):
    def __init__(self, num_features,  num_pids,  oim_momentum, oim_scalar,num_samples=0):
        super(OIMUnsupervisedLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.num_samples = num_samples
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("labels", torch.zeros(self.num_samples))
    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        # import pdb;pdb.set_trace()
        targets = torch.cat(roi_label)
        targets = self.labels[targets]
        label = targets - 1  # background label = -1
        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected = oim(inputs, label, self.lut, momentum=self.momentum)
        projected *= self.oim_scalar
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        return loss_oim