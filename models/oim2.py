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
        self.register_buffer("lut_instance", torch.zeros(self.num_pids, self.num_features))

        self.register_buffer("labels", torch.zeros(self.num_samples))
    # def forward(self, inputs, roi_label):
    #     # merge into one batch, background label = 0
    #     # import pdb;pdb.set_trace()
    #     targets = torch.cat(roi_label)
    #     # print(targets, self.labels.shape)
    #     targets = targets - 1
    #     inds = targets >= 0
    #     label = targets[inds]
    #     # inputs = inputs[inds]
    #     label = self.labels[label]
    #     # print(inputs.shape)
    #     inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     print(inputs.shape, label)
        
    #     # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
    #     projected = oim(inputs, label, self.lut, momentum=self.momentum)
    #     projected *= self.oim_scalar
    #     loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
    #     return loss_oim
    
    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        # import pdb;pdb.set_trace()
        targets = torch.cat(roi_label)
        # print(targets)
        targets = self.labels[targets]
        # print(targets)
        # label = targets - 1  # background label = -1
        inds = targets >= 1
        label = targets[inds]-1

        # print(targets, self.labels.shape, label, self.lut.shape)
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        
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
        self.register_buffer("lut_instance", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("labels", torch.zeros(self.num_samples))
        self.register_buffer("reid_lut", torch.zeros(self.num_samples))
        self.register_buffer("reid_labels", torch.zeros(self.num_samples))
        self.criterion_mse = nn.MSELoss()
    # def forward(self, inputs, roi_label):
    #     targets = torch.cat(roi_label)
    #     targets = targets - 1
    #     inds = targets >= 0
    #     label = targets[inds]
    #     re_id_features = self.lut_instance[label]
    #     # inputs = inputs[inds]
    #     label = self.labels[label]
    #     # print(inputs.shape)
    #     inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     # print(inputs.shape, label)
    #     # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)

    #     projected = oim(inputs, label, self.lut, momentum=self.momentum)
    #     projected *= self.oim_scalar
    #     loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
    #     feature_consistency = torch.mean(1-torch.cosine_similarity(re_id_features, inputs))
    #     return loss_oim+feature_consistency
    def criterion_instance(self, inputs, label):
        outputs_instance = inputs.mm(self.lut_instance.t())
        outputs_instance*= self.oim_scalar
        # logits = torch.einsum("nc, kc->nk", [inputs, self.lut_instence.clone().detach()])
        loss_instance_batch =0 
        for k in range(outputs_instance.shape[0]):
            loss_instance = 0
            negtive_label = torch.nonzero(self.reid_labels != label[k]).squeeze(-1)
            positive_label = torch.nonzero(self.reid_labels == label[k]).squeeze(-1)
            for pos in positive_label:
                input_instance = torch.cat((outputs_instance[k][pos].unsqueeze(-1), outputs_instance[k, negtive_label]),dim=0)
                instance_target =  torch.zeros(len(input_instance), dtype = input_instance.dtype).cuda()
                instance_target[0] = 1
                loss_instance +=  -1* (F.log_softmax(input_instance.unsqueeze(0), dim = 1) * instance_target.unsqueeze(0)).sum()
            loss_instance_batch += loss_instance/len(positive_label)
        loss_instance_batch /= outputs_instance.shape[0]
        # pass
        return loss_instance_batch

    def forward(self, inputs, roi_label):
        targets = torch.cat(roi_label)
        targets = targets - 1
        inds = targets >= 0
        label = targets[inds]
        re_id_features = self.lut_instance[label]
        # inputs = inputs[inds]
        label_reid = self.reid_labels[label]
        label = self.labels[label]
        # print(inputs.shape)
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # print(inputs.shape, label)
        # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        # projected = oim(inputs, label, self.lut, momentum=self.momentum)
        # projected *= self.oim_scalar
        # loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        # feature_consistency = torch.mean(1-torch.cosine_similarity(re_id_features, inputs))
        feature_consistency = self.criterion_mse(re_id_features, inputs)
        index_reid = label_reid>=0
        label_reid = label_reid[index_reid]
        inputs_reid = inputs[index_reid.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        instance_loss = self.criterion_instance(inputs_reid, label_reid)
        outputs_reid = inputs_reid.mm(self.reid_lut.t())
        outputs_reid *= self.oim_scalar
        # print(label_reid, label_reid.shape, outputs_reid.shape, projected.shape, label)
        loss_oim_reid = F.cross_entropy(outputs_reid, label_reid, ignore_index=5554)
        # print(inputs.shape, label)
        return feature_consistency+loss_oim_reid +instance_loss # +loss_oim





class OIMUnsupervisedLossOri(nn.Module):
    def __init__(self, num_features,  num_pids,  oim_momentum, oim_scalar,num_samples=0):
        super(OIMUnsupervisedLossOri, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.num_samples = num_samples
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("lut_instance", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("labels", torch.zeros(self.num_samples))
        self.register_buffer("reid_lut", torch.zeros(self.num_samples))
        self.register_buffer("reid_labels", torch.zeros(self.num_samples))
        self.criterion_mse = nn.MSELoss()
    def forward(self, inputs, roi_label):
        targets = torch.cat(roi_label)
        targets = targets - 1
        inds = targets >= 0
        label = targets[inds]
        # re_id_features = self.lut_instance[label]
        # inputs = inputs[inds]
        label = self.labels[label]
        # print(inputs.shape)
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # print(inputs.shape, label)
        # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)

        projected = oim(inputs, label, self.lut, momentum=self.momentum)
        projected *= self.oim_scalar
        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        # feature_consistency = torch.mean(1-torch.cosine_similarity(re_id_features, inputs))
        return loss_oim #+feature_consistency

    def criterion_instance(self, inputs, label):
        outputs_instance = inputs.mm(self.lut_instance.t())
        outputs_instance*= self.oim_scalar
        # logits = torch.einsum("nc, kc->nk", [inputs, self.lut_instence.clone().detach()])
        loss_instance_batch =0 
        for k in range(outputs_instance.shape[0]):
            loss_instance = 0
            negtive_label = torch.nonzero(self.reid_labels != label[k]).squeeze(-1)
            positive_label = torch.nonzero(self.reid_labels == label[k]).squeeze(-1)
            for pos in positive_label:
                input_instance = torch.cat((outputs_instance[k][pos].unsqueeze(-1), outputs_instance[k, negtive_label]),dim=0)
                instance_target =  torch.zeros(len(input_instance), dtype = input_instance.dtype).cuda()
                instance_target[0] = 1
                loss_instance +=  -1* (F.log_softmax(input_instance.unsqueeze(0), dim = 1) * instance_target.unsqueeze(0)).sum()
            loss_instance_batch += loss_instance/len(positive_label)
        loss_instance_batch /= outputs_instance.shape[0]
        # pass
        return loss_instance_batch

    # def forward(self, inputs, roi_label):
    #     targets = torch.cat(roi_label)
    #     targets = targets - 1
    #     inds = targets >= 0
    #     label = targets[inds]
    #     re_id_features = self.lut_instance[label]
    #     # inputs = inputs[inds]
    #     label_reid = self.reid_labels[label]
    #     label = self.labels[label]
    #     # print(inputs.shape)
    #     inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     # print(inputs.shape, label)
    #     # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
    #     # projected = oim(inputs, label, self.lut, momentum=self.momentum)
    #     # projected *= self.oim_scalar
    #     # loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
    #     # feature_consistency = torch.mean(1-torch.cosine_similarity(re_id_features, inputs))
    #     feature_consistency = self.criterion_mse(re_id_features, inputs)
    #     index_reid = label_reid>=0
    #     label_reid = label_reid[index_reid]
    #     inputs_reid = inputs[index_reid.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     instance_loss = self.criterion_instance(inputs_reid, label_reid)
    #     outputs_reid = inputs_reid.mm(self.reid_lut.t())
    #     outputs_reid *= self.oim_scalar
    #     # print(label_reid, label_reid.shape, outputs_reid.shape, projected.shape, label)
    #     loss_oim_reid = F.cross_entropy(outputs_reid, label_reid, ignore_index=5554)
    #     # print(inputs.shape, label)
    #     return feature_consistency+loss_oim_reid +instance_loss # +loss_oim

    # def forward(self, inputs, roi_label):
    #     # merge into one batch, background label = 0
    #     # import pdb;pdb.set_trace()
    #     targets = torch.cat(roi_label)
    #     # print(targets)
    #     targets = self.labels[targets]
    #     # print(targets)
    #     # label = targets - 1  # background label = -1
    #     inds = targets >= 1
    #     label = targets[inds]-1
    #     # print(targets, self.labels.shape, label, self.lut.shape)
    #     inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        
    #     projected = oim(inputs, label, self.lut, momentum=self.momentum)
    #     projected *= self.oim_scalar
    #     loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
    #     return loss_oim

    # def forward(self, inputs, roi_label):
    #     targets = torch.cat(roi_label)
    #     targets = targets - 1
    #     inds = targets >= 0
    #     label = targets[inds]
    #     re_id_features = self.lut_instance[label]
    #     # inputs = inputs[inds]
    #     label_reid = self.reid_labels[label]
    #     label = self.labels[label]
    #     # print(inputs.shape)
    #     inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     # print(inputs.shape, label)
    #     # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
    #     # projected = oim(inputs, label, self.lut, momentum=self.momentum)
    #     # projected *= self.oim_scalar
    #     # loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
    #     # feature_consistency = torch.mean(1-torch.cosine_similarity(re_id_features, inputs))
    #     feature_consistency = self.criterion_mse(re_id_features, inputs)

    #     instance_consistency = self.criterion_instence(re_id_features, inputs, label)
    #     index_reid = label_reid>=0
    #     label_reid = label_reid[index_reid]

    #     inputs_reid = inputs[index_reid.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
    #     outputs_reid = inputs_reid.mm(self.reid_lut.t())
    #     outputs_reid *= self.oim_scalar
    #     # print(label_reid, label_reid.shape, outputs_reid.shape, projected.shape, label)
    #     loss_oim_reid = F.cross_entropy(outputs_reid, label_reid, ignore_index=5554)
    #     # print(inputs.shape, label)
    #     return feature_consistency+loss_oim_reid # +loss_oim
    
    
