import torch
import pdb

class MCF_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, MFilter):

        MFilter = torch.abs(MFilter)
        
        bin = 0.02
        
        weight_bin = torch.sign(weight) * bin

        output = weight_bin * MFilter

        ctx.save_for_backward(weight, MFilter)
        
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, MFilter = ctx.saved_tensors
        
        para_loss = 0.0001
        bin = 0.02
        weight_bin = torch.sign(weight) * bin

        target1 = para_loss * (weight - weight_bin * MFilter)
        
        gradWeight = target1 + (gradOutput * MFilter)
        target2 = (weight - weight_bin * MFilter) * weight_bin
        
        #pdb.set_trace()
        grad_h2_sum = torch.sum(torch.sum(torch.sum(gradOutput * weight, keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True, dim=1)
        
        grad_target2 = torch.sum(torch.sum(torch.sum(target2,keepdim=True,dim=3),keepdim=True, dim=2),keepdim=True, dim=1)

        gradMFilter = grad_h2_sum - para_loss * grad_target2
        #pdb.set_trace()
        return gradWeight, gradMFilter
