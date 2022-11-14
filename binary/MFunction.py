import torch
import pdb

class MCF_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight, MFilter):
        # size的第一个元素代表out_ch，第二个元素是in_ch
        # pdb.set_trace()
        nOutputPlane = weight.size()[0]
        nInputPlane = weight.size()[1]
        nChannel = weight.size()[2]
        kernel_size = weight.size()[3]
        # output_temp = weight.new(weight.size())
        output = weight.view(nOutputPlane * nChannel, nInputPlane * nChannel, kernel_size, kernel_size)
        MFilter = torch.abs(MFilter)
        bin = 0.02
        MFilterMean_temp=torch.sum(MFilter, dim=1) / kernel_size / kernel_size
        MFilterMean = torch.sum(MFilterMean_temp, dim=1)
        weight_bin = torch.sign(output) * bin
        #for i in range(nChannel):
        #    if i == 0:
        output = weight_bin * MFilterMean
        #    else:
        #        output = torch.cat([output, weight_bin * MFilterMean[i]], dim=2)
        #pdb.set_trace()
        #print(output)
        ctx.save_for_backward(weight, MFilter)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        weight, MFilter = ctx.saved_tensors
        #grad_weight = weight
        #sgrad_MFilter = MFilter
        nChannel = MFilter.size()[0]
        nOutputPlane = gradOutput.size()[0]
        nInputPlane = gradOutput.size()[1]
        kernel_size = gradOutput.size()[2]
        nEntry = nChannel * kernel_size * kernel_size # 9
        #gradWeight = grad_weight.view(nOutputPlane, nInputPlane, nChannel, kernel_size, kernel_size)
        #gradMFilter = grad_MFilter.view(nChannel, kernel_size, kernel_size)
        para_loss = 0.0001
        bin = 0.02
        weight_bin = torch.sign(weight) * bin

        target1 = para_loss * (weight - weight_bin * MFilter)
        #gradWeight = target1
        #pdb.set_trace()
        gradWeight = target1 + (gradOutput * MFilter).view(nOutputPlane, nInputPlane, nChannel, kernel_size, kernel_size)
        target2 = (weight - weight_bin * MFilter) * weight_bin
        #pdb.set_trace()
        grad_h2_sum = torch.sum(torch.sum(gradOutput.view(nOutputPlane, nInputPlane, nChannel, kernel_size,kernel_size) * weight,dim=1),dim=0)
        
        grad_target2 = torch.sum(torch.sum(target2, dim=1),dim=0)
        gradMFilter = grad_h2_sum - para_loss * grad_target2
        #pdb.set_trace()
        return gradWeight, gradMFilter
