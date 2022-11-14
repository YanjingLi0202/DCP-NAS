import os, sys
import shutil
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

import torch
import torch.nn as nn
import torch.optim
import logging
import numpy as np
import torchvision.transforms as transforms


def download_url(url, model_dir="~/.torch/proxyless_nas", overwrite=False):
    model_dir = os.path.expanduser(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        os.makedirs(model_dir, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


def load_url(url, model_dir='~/.torch/proxyless_nas', map_location=None):
    cached_file = download_url(url, model_dir)
    map_location = "cpu" if not torch.cuda.is_available() and map_location is None else None
    return torch.load(cached_file, map_location=map_location)


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, torch.unsqueeze(targets, 1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = torch.mean(torch.sum(-targets * log_probs, 1))
    return loss


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax()
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def shuffle_layer(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    # transpose
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def get_split_list(in_dim, child_num):
    in_dim_list = [in_dim // child_num] * child_num
    for _i in range(in_dim % child_num):
        in_dim_list[_i] += 1
    return in_dim_list


def list_sum(x):
    if len(x) == 1:
        return x[0]
    else:
        return x[0] + list_sum(x[1:])


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


def count_conv_flop(layer, x):
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[
        1] * out_h * out_w / layer.groups
    return delta_ops


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(1.0 / batch_size))
    return res

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def netParams(model):
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @property
    def unit_str(self):
        raise NotImplementedError

    @property
    def config(self):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    def get_flops(self, x):
        raise NotImplementedError


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('darts')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


def as_markdown(config):
    """ Return configs as markdown format """
    text = "|name|value|  \n|-|-|  \n"
    for attr, value in sorted(config.items()):
        text += "|{}|{}|  \n".format(attr, value)

    return text


def print_params(config, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(config.items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")


def param_size(model):
    """ Compute parameter size in M """
    n_params = sum(
        np.prod(v.size()) for k, v in model.named_parameters() if not k.startswith('auxiliary_head')) # auxiliary_head aux_head
    return n_params / 1e6


def param_size_(model):
    """ Compute parameter size in M """
    temp = []
    for i in model.parameters():
        temp.append(i)
    # print(temp)
    M_params = sum(
        np.prod(temp[i-1].size()) for i, (k, v) in enumerate(model.named_parameters()) if 'MFilters' in k and not k.startswith('auxiliary_head')) # auxiliary_head aux_head
    N_params = sum(
        np.prod(temp[i-1].size()) for i, (k, v) in enumerate(model.named_parameters()) if 'MFilters' not in k and not k.startswith('auxiliary_head')) # auxiliary_head aux_head
    FC_params = sum(
        np.prod(temp[i].size()) for i, (k, v) in enumerate(model.named_parameters()) if 'class' in k and not k.startswith('auxiliary_head')) # auxiliary_head aux_head
    ST_params = sum(
        np.prod(temp[i].size()) for i, (k, v) in enumerate(model.named_parameters()) if 'stem' in k and not k.startswith('auxiliary_head')) # auxiliary_head aux_head
    all_params = M_params + N_params
    per = M_params / all_params
    print(M_params / 1e6)
    print(N_params / 1e6)
    print(FC_params / 1e6)
    print(ST_params / 1e6)
    print(per)
    return per#n_params / 1e6


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        # transforms.Resize(224),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform




# transf = [
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             CIFAR10Policy()
#         ]