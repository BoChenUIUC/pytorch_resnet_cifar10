import os.path as osp
import re
from collections import OrderedDict
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, ConvTranspose2d
import seaborn as sns
import matplotlib.pyplot as plt
import resnet
import os, math

def load_checkpoint(model, filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {}) for pruning")

def save_checkpoint(model, filename):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

class FisherPruningHook():
    """Use fisher information to pruning the model, must register after
    optimizer hook.

    Args:
        pruning (bool): When True, the model in pruning process,
            when False, the model is in finetune process.
            Default: True
        delta (str): "acts" or "flops", prune the model by
            "acts" or flops. Default: "acts"
        interval (int): The interval of  pruning two channels.
            Default: 10
        deploy_from (str): Path of checkpoint containing the structure
            of pruning model. Defaults to None and only effective
            when pruning is set True.
        save_flops_thr  (list): Checkpoint would be saved when
            the flops reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
        save_acts_thr (list): Checkpoint would be saved when
            the acts reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
    """
    def __init__(
        self,
        pruning=True,
        delta='flops',
        interval=10,
        use_ista=False,
        deploy_from=None,
        resume_from=None,
        start_from=None,
        save_flops_thr=[0.75, 0.5, 0.25],
        save_acts_thr=[0.75, 0.5, 0.25],
    ):

        assert delta in ('acts', 'flops')
        self.pruning = pruning
        self.use_ista = use_ista
        self.delta = delta
        self.interval = interval
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        
        self.channels = 0
        self.delta = delta
        self.deploy_from = deploy_from
        self.resume_from = resume_from
        self.start_from = start_from

        for i in range(len(save_acts_thr) - 1):
            assert save_acts_thr[i] > save_acts_thr[i + 1]
        for i in range(len(save_flops_thr) - 1):
            assert save_flops_thr[i] > save_flops_thr[i + 1]

        self.save_flops_thr = save_flops_thr
        self.save_acts_thr = save_acts_thr
        
        self.total_flops = self.total_acts = 0
        
        self.iter = 0

    def after_build_model(self, model):
        """Remove all pruned channels in finetune stage.

        We add this function to ensure that this happens before DDP's
        optimizer's initialization
        """
            
        if self.start_from is not None:
            load_checkpoint(model, self.start_from)
        
        self.conv_names = OrderedDict() # prunable
        self.ln_names = OrderedDict()
        self.name2module = OrderedDict()

        for n, m in model.named_modules():
            if n: m.name = n
            self.add_pruning_attrs(m, pruning=self.pruning)
            if isinstance(m, nn.Conv2d):
                self.conv_names[m] = n
                self.name2module[n] = m
            elif isinstance(m, nn.BatchNorm2d):
                self.ln_names[m] = n
                self.name2module[n] = m
        
        self.set_group_masks(model)

        if self.pruning:
            # divide the conv to several group and all convs in same
            # group used same input at least once in model's
            # forward process.
            self.init_flops_acts()
            if self.resume_from is not None:
                load_checkpoint(model, self.resume_from)
            # register forward hook
            for module, name in self.conv_names.items():
                module.register_forward_hook(self.save_input_forward_hook)
        else:
            load_checkpoint(model, self.deploy_from)
            self.deploy_pruning(model)

        self.print_model(model, print_flops_acts=False, print_channel=False)

    def after_backward(self, itr, model):
        if not self.pruning:
            return
        if itr % self.interval == 0:
            self.ista()
            self.total_flops, self.total_acts = self.update_flop_act(model)
        self.init_flops_acts()
        
    def plot(self, print_str):
        scale_factors = torch.tensor([]).cuda()
        for module, name in self.conv_names.items():
            bn_module = self.name2module[module.name.replace('conv','bn')]
            scale_factors = torch.cat((scale_factors,torch.abs(bn_module.weight.data.view(-1))))
        # plot figure
        save_dir = f'metrics/logq/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fig, axs = plt.subplots(ncols=2, figsize=(10,4))
        # plots
        sns.histplot(scale_factors.detach().cpu().numpy(), ax=axs[0])
        scale_factors = torch.clamp(scale_factors,min=1e-10)
        sns.histplot(torch.log10(scale_factors).detach().cpu().numpy(), ax=axs[1])
        fig.savefig(save_dir + f'{self.iter:03d}_{print_str}.png')
        plt.close('all')
        self.iter += 1
        self.init_flops_acts()

    def update_flop_act(self, model, work_dir='work_dir/'):
        flops, acts = self.compute_flops_acts()
        if len(self.save_flops_thr):
            flops_thr = self.save_flops_thr[0]
            if flops < flops_thr:
                self.save_flops_thr.pop(0)
                path = osp.join(
                    work_dir, 'flops_{:.0f}_acts_{:.0f}.pth'.format(
                        flops * 100, acts * 100))
                save_checkpoint(model, filename=path)
        if len(self.save_acts_thr):
            acts_thr = self.save_acts_thr[0]
            if acts < acts_thr:
                self.save_acts_thr.pop(0)
                path = osp.join(
                    work_dir, 'acts_{:.0f}_flops_{:.0f}.pth'.format(
                        acts * 100, flops * 100))
                save_checkpoint(model, filename=path)
        return flops, acts

    def print_model(self, model, work_dir='work_dir/', print_flops_acts=True, print_channel=True):
        """Print the related information of the current model.

        Args:
            runner (Runner): Runner in mmcv
            print_flops_acts (bool): Print the remained percentage of
                flops and acts
            print_channel (bool): Print information about
                the number of reserved channels.
        """

        if print_flops_acts:
            flops, acts = self.update_flop_act(model, work_dir)
            print('Flops: {:.2f}%, Acts: {:.2f}%'.format(flops * 100, acts * 100))
        if print_channel:
            for module, name in self.conv_names.items():
                chans_i = int(module.in_mask.sum().cpu().numpy())
                if hasattr(module, 'child'):
                    child = self.name2module[module.child]
                    chans_o = int(child.in_mask.sum().cpu().numpy())
                    print('{}: input_channels: {}/{}, out_channels: {}/{}'.format(name, chans_i, len(module.in_mask), chans_o, len(child.in_mask)))
                else:
                    chans_o = module.out_channels
                    print('{}: input_channels: {}/{}'.format(name, chans_i, len(module.in_mask)))

    def compute_flops_acts(self):
        """Computing the flops and activation remains."""
        flops = 0
        max_flops = 0
        acts = 0
        max_acts = 0
        for module, name in self.conv_names.items():
            max_flop = self.flops[module]
            i_mask = module.in_mask
            if hasattr(module, 'child'):
                child = self.name2module[module.child]
                real_out_channels = child.in_mask.cpu().sum()
            else:
                real_out_channels = module.out_channels
             
            flops += max_flop / (i_mask.numel() * module.out_channels) * (
                i_mask.cpu().sum() * real_out_channels)
            max_flops += max_flop
            max_act = self.acts[module]
            acts += max_act / module.out_channels * real_out_channels
            max_acts += max_act
        return flops / max_flops, acts / max_acts
        
    def factor_norm(self):
        all_scale_factors = torch.tensor([]).cuda()
            
        for module, name in self.conv_names.items():
            bn_module = self.name2module[module.name.replace('conv','bn')]
            all_scale_factors = torch.cat((all_scale_factors,torch.abs(bn_module.weight.data)))
            
        return 1e10*all_scale_factors.norm(p=1)
                
    def ista(self):
        self.ista_err = torch.tensor([0.0]).cuda(0)
        # locations of bins should fit original dist
        num_bins = 4
        bin_start = -6
        # distance between bins min=2
        bin_stride = 2
        # how centralize the bin is
        bin_width = 1e-1
        # locations we want to quantize
        bins = torch.pow(10.,torch.tensor([bin_start+bin_stride*x for x in range(num_bins)])).cuda(0)
        # trade-off of original distribution and new distribution
        # big: easy to get new distribution, but may degrade performance
        # small: maintain good performance but may not affect distribution much
        decay_factor = 1e-3 # lower this to improve perf
        # how small/low rank bins get more advantage
        amp_factors = torch.tensor([2**(num_bins-1-x) for x in range(num_bins)]).cuda()
        self.ista_err_bins = [0 for _ in range(num_bins)]
        self.ista_cnt_bins = [0 for _ in range(num_bins)]
        
        def exp_quantization(x):
            x = torch.clamp(torch.abs(x), min=1e-8) * torch.sign(x)
            dist = torch.abs(torch.log10(torch.abs(x).unsqueeze(-1)/bins))
            _,min_idx = dist.min(dim=-1)
            all_err = torch.log10(bins[min_idx]/torch.abs(x))
            abs_err = torch.abs(all_err)
            # calculate total error
            self.ista_err += abs_err.sum()
            # calculating err for each bin
            for i in range(num_bins):
                if torch.sum(min_idx==i)>0:
                    self.ista_err_bins[i] += abs_err[min_idx==i].sum().cpu().item()
                    self.ista_cnt_bins[i] += torch.numel(abs_err[min_idx==i])
            # modification of weights
            sn = torch.sign(torch.log(bins[min_idx]/torch.abs(x)))
            multiplier = 10**(sn*bin_stride*decay_factor) 
            x[abs_err>bin_width] *= multiplier[abs_err>bin_width]
            return x
            
        def get_bin_distribution(x):
            x = torch.clamp(torch.abs(x), min=1e-6) * torch.sign(x)
            bins = torch.pow(10.,torch.tensor([bin_start+bin_stride*x for x in range(num_bins)])).to(x.device)
            dist = torch.abs(torch.log10(torch.abs(x).unsqueeze(-1)/bins))
            _,min_idx = dist.min(dim=-1)
            all_err = torch.log10(bins[min_idx]/torch.abs(x))
            abs_err = torch.abs(all_err)
            # calculate total error
            self.ista_err += abs_err.sum()
            # calculating err for each bin
            for i in range(num_bins):
                if torch.sum(min_idx==i)>0:
                    self.ista_err_bins[i] += abs_err[min_idx==i].sum().cpu().item()
                    self.ista_cnt_bins[i] += torch.numel(abs_err[min_idx==i])
                    
        def redistribute(x,bin_indices):
            tar_bins = bins[bin_indices]
            # amplifier based on rank of bin
            amp = amp_factors[bin_indices]
            all_err = torch.log10(tar_bins/torch.abs(x))
            abs_err = torch.abs(all_err)
            # more distant larger multiplier
            # pull force relates to distance and target bin (how off-distribution is it?)
            # low rank bin gets higher pull force
            distance = torch.log10(tar_bins/torch.abs(x))
            multiplier = 10**(distance*decay_factor*amp)
            x[abs_err>bin_width] *= multiplier[abs_err>bin_width]
            return x
            
        all_scale_factors = torch.tensor([]).cuda()
            
        for module, name in self.conv_names.items():
            bn_module = self.name2module[module.name.replace('conv','bn')]
            with torch.no_grad():
                get_bin_distribution(bn_module.weight.data)
            all_scale_factors = torch.cat((all_scale_factors,torch.abs(bn_module.weight.data)))
                    
        # total channels
        total_channels = len(all_scale_factors)
        ch_per_bin = total_channels//num_bins
        _,bin_indices = torch.tensor(self.ista_cnt_bins).sort()
        remain = torch.ones(total_channels).long().cuda()
        assigned_binindices = torch.zeros(total_channels).long().cuda()
        
        for bin_idx in bin_indices[:-1]:
            dist = torch.abs(torch.log10(bins[bin_idx]/all_scale_factors)) 
            not_assigned = remain.nonzero()
            # remaining channels importance
            chan_imp = dist[not_assigned] 
            tmp,ch_indices = chan_imp.sort(dim=0)
            selected_in_remain = ch_indices[:ch_per_bin]
            selected = not_assigned[selected_in_remain]
            remain[selected] = 0
            assigned_binindices[selected] = bin_idx
        assigned_binindices[remain.nonzero()] = bin_indices[-1]
        
        ch_start = 0
        for module, name in self.conv_names.items():
            bn_module = self.name2module[module.name.replace('conv','bn')]
            with torch.no_grad():
                ch_len = len(bn_module.weight.data)
                bn_module.weight.data = redistribute(bn_module.weight.data, assigned_binindices[ch_start:ch_start+ch_len])
            ch_start += ch_len
            
    def deploy_pruning(self,model):
        # sort according to factor
        all_scale_factors = torch.tensor([]).cuda()
            
        for module, name in self.conv_names.items():
            bn_module = self.name2module[module.name.replace('conv','bn')]
            all_scale_factors = torch.cat((all_scale_factors,torch.abs(bn_module.weight.data)))
                
        sorted,factor_indices = all_scale_factors.sort()
        total_channels = len(all_scale_factors)
        prune_ratio = 0.5
        removed_channels = int(prune_ratio * total_channels)
        print('remove:',removed_channels)
        all_masks = torch.ones(total_channels).long().cuda()
        all_masks[factor_indices[:removed_channels]] = 0
        
        # assign mask back
        # todo check remove the right channels
        ch_start = 0
        for module, name in self.conv_names.items():
            bn_module = self.name2module[module.name.replace('conv','bn')]
            with torch.no_grad():
                ch_len = len(bn_module.weight.data)
                bn_mask = all_masks[ch_start:ch_start+ch_len]
                bn_module.weight.data *= bn_mask
            ch_start += ch_len

    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def save_input_forward_hook(self, module, inputs, outputs):
        """Save the input and flops and acts for computing fisher and flops or
        acts. Total flops

        Args:
            module (nn.Module): the module of register hook
        """
        module = self.name2module[module.name]
        layer_name = type(module).__name__
        if layer_name in ['Conv2d']:
            n, oc, oh, ow = module.output_size
            ic = module.in_channels
            kh, kw = module.kernel_size
            self.flops[module] += np.prod([n, oc, oh, ow, ic, kh, kw])
            self.acts[module] += np.prod([n, oc, oh, ow])
        else:
            print('Unrecognized in save_input_forward_hook:',layer_name)
            exit(0)

    def make_groups(self):
        """The modules (convolutions and BNs) connected to the same conv need
        to change the channels simultaneously when pruning.

        This function divides all modules into different groups according to
        the connections.
        """

        idx = -1
        groups, groups_ancest = {}, {}
        for module, name in reversed(self.conv_names.items()):
            added = False
            for group in groups:
                module_ancest = set(self.conv2ancest[module])
                group_ancest = set(groups_ancest[group])
                if len(module_ancest.intersection(group_ancest)) > 0:
                    groups[group].append(module)
                    groups_ancest[group] = list(
                        module_ancest.union(group_ancest))
                    added = True
                    break
            if not added:
                idx += 1
                groups[idx] = [module]
                groups_ancest[idx] = self.conv2ancest[module]
        # key is the ids the group, and value contains all conv
        # of this group
        self.groups = {}
        # key is the ids the group, and value contains all nearest
        # ancestor of this group
        self.ancest = {}
        idx = 0
        # filter the group with only one conv
        for group in groups:
            modules = groups[group]
            if len(modules) > 1:
                self.groups[idx] = modules
                self.ancest[idx] = groups_ancest[group]
                idx += 1
        for id in self.groups:
            module0 = self.groups[id][0]
            for module in self.groups[id]:
                module.group_master = module0.name

    def set_group_masks(self, model):
        """the modules(convolutions and BN) connect to same convolutions need
        change the out channels at same time when pruning, divide the modules
        into different groups according to the connection.

        Args:
            model(nn.Module): the model contains all modules
        """
        # split into prunable or not
        # prunable group modules with coupled inchannel, update outchannel based on correlation
        # self.conv2ancest is a dict, key is all conv/deconv/linear/bitparm/layernorm instance in
        # model, value is a list which contains all [nearest] ancestor
        # bitparm and layer norm only need the mask of conv/deconv/linear to decide mask
        self.find_module_ancestors(model)
        self.make_groups()

        # list contains all the convs which are contained in a
        # group (if more the one conv has same ancestor,
        # they will be in same group)
        self.group_modules = []
        for group in self.groups:
            self.group_modules.extend(self.groups[group])
            
        self.conv_names_group = [[item.name for item in v]
                                 for idx, v in self.groups.items()]
        #for g in self.conv_names_group:
        #    print(g)
        #exit(0)

    def find_module_ancestors(self, model):
        """find the nearest module
        Args:
            loss(Tensor): the output of the network
            pattern(Tuple[str]): the pattern name

        Returns:
            dict: the key is the module match the pattern(Conv or Fc),
             and value is the list of it's nearest ancestor 
        """
        import re
        conv2ancest = {}
        for n, m in model.named_modules():
            if type(m).__name__ not in ['Conv2d','BatchNorm2d']:
                continue
            # find ancestors
            if 'conv1' == n:
                ancestors = []
            elif 'conv' in n:
                a,b,c = re.findall(r'\d+',n)
                if c == '1':
                    if b == '0':
                        if a == '1':
                            ancestors = ['conv1']
                        elif a == '2':
                            ancestors = ['layer1.8.conv2']
                        else:
                            ancestors = ['layer2.8.conv2']
                    else:
                        if a == '1':
                            ancestors = ['layer1.0.conv2',f'layer1.{int(b)-1}.conv2']
                        elif a == '2':
                            ancestors = ['layer2.0.conv2',f'layer2.{int(b)-1}.conv2']
                        else:
                            ancestors = ['layer3.0.conv2',f'layer3.{int(b)-1}.conv2']
                else:
                    ancestors = [f'layer{a}.{b}.conv1']

            if type(m).__name__ in ['Conv2d']:
                conv2ancest[m] = []
            for name in ancestors:
                if type(m).__name__ in ['Conv2d']:
                    conv2ancest[m] += [self.name2module[name]]
                    
            # find child
            if 'conv1' == n or 'bn1' == n:
                m.child = 'layer1.0.conv1'
            elif 'conv' in n or 'bn' in n:
                a,b,c = re.findall(r'\d+',n)
                if c == '1':
                    m.child = f'layer{a}.{b}.conv2'
                else:
                    if b == '8':
                        if a == '3':
                            pass
                        else:
                            m.child = f'layer{int(a)+1}.0.conv1'
                    else:
                        m.child = f'layer{a}.{int(b)+1}.conv1'
            
        self.conv2ancest = conv2ancest

    def add_pruning_attrs(self, module, pruning=False):
        """When module is conv, add `finetune` attribute, register `mask` buffer
        and change the origin `forward` function. When module is BN, add `out_mask`
        attribute to module.

        Args:
            conv (nn.Conv2d):  The instance of `torch.nn.Conv2d`
            pruning (bool): Indicating the state of model which
                will make conv's forward behave differently.
        """
        # same group same softmask
        module.finetune = not pruning
        if type(module).__name__ == 'Conv2d':
            module.register_buffer('in_mask', module.weight.new_ones(module.in_channels,))
            def modified_forward(m, x):
                mask = m.in_mask.view(1,-1,1,1)
                x = x * mask.to(x.device)
                output = F.conv2d(x, m.weight, m.bias, m.stride, m.padding, m.dilation, m.groups)
                m.output_size = output.size()
                return output
            module.forward = MethodType(modified_forward, module) 
        if  type(module).__name__ == 'BatchNorm2d':
            # no need to modify layernorm during pruning since it is not computed over channels
            pass