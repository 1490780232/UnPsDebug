import torch
import torch.nn.functional as F
from torch import autograd, nn
from torch.cuda.amp import custom_fwd, custom_bwd

# from utils.distributed import tensor_gather

class OIM(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq, header, momentum = ctx.saved_tensors

        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, torch.tensor(header), torch.tensor(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
        self.register_buffer("ins_lut", torch.zeros(self.num_unlabeled, 20000))
        self.register_buffer("ins_label", torch.zeros(20000))

        
        self.header_cq = 0

    def forward(self, inputs, roi_label, roi_indexes):
        # merge into one batch, background label = 0
        # print(roi_indexes, "=======================indexes")
        # print(roi_label, "=======================roi_label")
        # print(self.ins_lut.shape, self.ins_label.shape)
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # projected_ins = inputs.mm(self.ins_lut.t())
        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected *= self.oim_scalar
        # projected_ins *= self.oim_scalar
        # temp_sims = projected_ins.detach().clone()
        # associate_loss=0
        # for k in range(len(label)):
        #     if label[k]>5000:
        #         continue
        #     ori_asso_ind = torch.nonzero(self.ins_label == label[k]).squeeze(-1)
        #     # print(ori_asso_ind.shape, label[k])
        #     weights = F.softmax(1-temp_sims[k, ori_asso_ind],dim=0)
        #     temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
        #     sel_ind = torch.sort(temp_sims[k])[1][-1000:]
        #     concated_input = torch.cat((projected_ins[k, ori_asso_ind], projected_ins[k, sel_ind]), dim=0)
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(torch.device('cuda'))
        #     concated_target[0:len(ori_asso_ind)] = weights #1.0 / len(ori_asso_ind)
        #     associate_loss += -1 * (F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss_wincetance = 0.5 * associate_loss / len(label)
        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554) #+ loss_wincetance
        return loss_oim
