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
        delta='acts',
        interval=10,
        trained_mask=False,
        noise_mask=False,
        one_shot=False,
        deploy_from=None,
        resume_from=None,
        start_from=None,
        save_flops_thr=[0.75, 0.5, 0.25],
        save_acts_thr=[0.75, 0.5, 0.25],
    ):

        assert delta in ('acts', 'flops')
        self.pruning = pruning
        self.trained_mask = trained_mask
        self.noise_mask = noise_mask
        self.delta = delta
        self.interval = interval
        # The key of self.input is conv module, and value of it
        # is list of conv' input_features in forward process
        self.conv_inputs = {}
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}
        self.temp_mag_info = {}
        self.temp_grad_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}
        self.batch_mags = {}
        self.batch_grads = {}

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.accum_mags = {}
        self.accum_grads = {}
        
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
        self.use_mask = True

    def after_build_model(self, model):
        """Remove all pruned channels in finetune stage.

        We add this function to ensure that this happens before DDP's
        optimizer's initialization
        """

        if not self.pruning:
            for n, m in model.named_modules():
                if n: m.name = n
                self.add_pruning_attrs(m, pruning=self.pruning)
            load_checkpoint(model, self.deploy_from)
            deploy_pruning(model)
            
        if self.start_from is not None:
            load_checkpoint(model, self.start_from)

    def before_run(self, model):
        """Initialize the relevant variables(fisher, flops and acts) for
        calculating the importance of the channel, and use the layer-grouping
        algorithm to make the coupled module shared the mask of input
        channel."""

        self.conv_names = OrderedDict() # prunable
        self.ln_names = OrderedDict()
        self.name2module = OrderedDict()

        for n, m in model.named_modules():
            if n: m.name = n
            if self.pruning:
                self.add_pruning_attrs(m, pruning=self.pruning)
            if isinstance(m, nn.Conv2d):
                self.conv_names[m] = n
                self.name2module[n] = m
            elif isinstance(m, nn.BatchNorm2d):
                self.ln_names[m] = n
                self.name2module[n] = m

        if self.pruning:
            # divide the conv to several group and all convs in same
            # group used same input at least once in model's
            # forward process.
            model.eval()
            self.set_group_masks(model)
            model.train()
            for conv, name in self.conv_names.items():
                self.conv_inputs[conv] = []
                # fisher info
                self.temp_fisher_info[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask)) 
                self.accum_fishers[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask))
                # magnitude info
                self.temp_mag_info[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask)) 
                self.accum_mags[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask))
                # gradiant info    
                self.temp_grad_info[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask)) 
                self.accum_grads[conv] = conv.in_mask.data.new_zeros(len(conv.in_mask))
            for group_id in self.groups:
                module = self.groups[group_id][0]
                # fisher info
                self.temp_fisher_info[group_id] = module.in_mask.data.new_zeros(len(module.in_mask))
                self.accum_fishers[group_id] = module.in_mask.data.new_zeros(len(module.in_mask))
                # magnitude info
                self.temp_mag_info[group_id] = module.in_mask.data.new_zeros(len(module.in_mask)) 
                self.accum_mags[group_id] = module.in_mask.data.new_zeros(len(module.in_mask))
                # gradiant info    
                self.temp_grad_info[group_id] = module.in_mask.data.new_zeros(len(module.in_mask)) 
                self.accum_grads[group_id] = module.in_mask.data.new_zeros(len(module.in_mask))
            self.init_flops_acts()
            self.init_temp_fishers()
            if self.resume_from is not None:
                load_checkpoint(model, self.resume_from)
            # register forward hook
            for module, name in self.conv_names.items():
                module.register_forward_hook(self.save_input_forward_hook)

        self.print_model(model, print_flops_acts=False, print_channel=False)

    def after_backward(self, itr, model, loss):
        if not self.pruning:
            return
        # compute fisher
        for module, name in self.conv_names.items():
            self.compute_fisher_backward(module)
        # do pruning every interval
        self.group_fishers()
        self.accumulate_fishers()
        self.init_temp_fishers()
        if itr % self.interval == 0:
            # this makes sure model is converged before each pruning
            self.channel_prune()
            self.init_accum_fishers()
            self.total_flops, self.total_acts = self.update_flop_act(model)
            # plot figure
            if itr == 0:
                # fisher
                plt.figure(1)
                self.fisher_list[self.fisher_list==0] = 1e-50
                self.fisher_list = torch.log10(self.fisher_list).detach().cpu().numpy()
                sns.displot(self.fisher_list, kind='hist', aspect=1.2)
                plt.savefig(f'metrics/dist_fisher_{self.iter}_{int(self.total_flops*100):4d}_{int(self.total_acts*100):4d}_{loss:.4f}.png')
                # magnitude
                #plt.figure(2)
                #self.mag_list[self.mag_list==0] = 1e-50
                #self.mag_list = torch.log10(self.mag_list).detach().cpu().numpy()
                #sns.displot(self.mag_list, kind='hist', aspect=1.2)
                #plt.savefig(f'metrics/dist_mag_{self.iter}_{int(self.total_flops*100):4d}_{int(self.total_acts*100):4d}_{loss:.4f}.png')
                # gradient
                #plt.figure(3)
                #self.grad_list[self.grad_list==0] = 1e-50
                #self.grad_list = torch.log10(self.grad_list).detach().cpu().numpy()
                #sns.displot(self.grad_list, kind='hist', aspect=1.2)
                #plt.savefig(f'metrics/dist_grad_{self.iter}_{int(self.total_flops*100):4d}_{int(self.total_acts*100):4d}_{loss:.4f}.png')
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
                    if hasattr(child,'group_master'):
                        child = self.name2module[child.group_master]
                    chans_o = int(child.in_mask.sum().cpu().numpy())
                else:
                    chans_o = module.out_channels
                print('{}: input_channels: {}/{}, out_channels: {}/{}'.format(
                        name, chans_i, len(module.in_mask), chans_o, len(child.in_mask)))
            for module, name in self.ln_names.items():
                if hasattr(module, 'child'):
                    child = self.name2module[module.child]
                    if hasattr(child,'group_master'):
                        child = self.name2module[child.group_master]
                    chans_o = int(child.in_mask.sum().cpu().numpy())
                else:
                    chans_o = module.out_channels
                print('{}: out_channels: {}/{}'.format(name, chans_o, len(child.in_mask)))

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
                if hasattr(child,'group_master'):
                    child = self.name2module[child.group_master]
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

    def init_accum_fishers(self):
        """Clear accumulated fisher info."""
        for module, name in self.conv_names.items():
            self.accum_fishers[module].zero_()
            self.accum_mags[module].zero_()
            self.accum_grads[module].zero_()
        for group in self.groups:
            self.accum_fishers[group].zero_()
            self.accum_mags[group].zero_()
            self.accum_grads[group].zero_()

    def find_pruning_channel(self, module, fisher, in_mask, info):
        """Find the the channel of a model to pruning.

        Args:
            module (nn.Conv | int ): Conv module of model or idx of self.group
            fisher(Tensor): the fisher information of module's in_mask
            in_mask (Tensor): the squeeze in_mask of modules
            info (dict): store the channel of which module need to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta

        Returns:
            dict: store the current least important channel
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """
        module_info = {}
        # if hasattr(module,'name'):
        #     print(module.name, fisher.min())
        # else:
        #     print(module,fisher.min())
        if in_mask.sum() > 1:
            nonzero = in_mask.nonzero().view(-1)
            fisher = fisher[nonzero]
            min_value, argmin = fisher.min(dim=0)
            min_value = float(min_value)
            if min_value < info['min']:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def single_prune(self, info, exclude=None):
        """Find the channel with smallest fisher / delta in modules not in
        group.

        Args:
            info (dict): Store the channel of which module need
                to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta
            exclude (list): List contains all modules in group.
                Default: None

        Returns:
            dict: store the channel of which module need to be pruned
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """
        for module, name in self.conv_names.items():
            if exclude is not None and module in exclude:
                continue
            fisher = self.accum_fishers[module]
            mag = self.accum_mags[module]
            grad = self.accum_grads[module]
            in_mask = module.in_mask.view(-1)
            ancestors = self.conv2ancest[module]
            if self.delta == 'flops':
                # delta_flops is a value indicate how much flops is
                # reduced in entire forward process after we set a
                # zero in `in_mask` of a specific conv_module.
                # this affects both current and ancestor module
                # flops per channel
                if hasattr(module, 'child'):
                    child = self.name2module[module.child]
                    if hasattr(child,'group_master'):
                        child = self.name2module[child.group_master]
                    chans_o = child.in_mask.sum()
                else:
                    chans_o = module.out_channels
                delta_flops = self.flops[module] * chans_o / (
                    module.in_channels * module.out_channels)
                for ancestor in ancestors:
                    delta_flops += self.flops[ancestor] * ancestor.in_mask.sum(
                    ) / (ancestor.in_channels * ancestor.out_channels)
                fisher /= (float(delta_flops) / 1e9)
                mag /= (float(delta_flops) / 1e9)
                grad /= (float(delta_flops) / 1e9)
            if self.delta == 'acts':
                # activation only counts ancestors
                delta_acts = 0
                for ancestor in ancestors:
                    delta_acts += self.acts[ancestor] / ancestor.out_channels
                fisher /= (float(max(delta_acts, 1.)) / 1e6)
                mag /= (float(max(delta_acts, 1.)) / 1e6)
                grad /= (float(max(delta_acts, 1.)) / 1e6)
            self.fisher_list = torch.cat((self.fisher_list,fisher[in_mask.bool()].view(-1)))
            self.mag_list = torch.cat((self.mag_list,mag[in_mask.bool()].view(-1)))
            self.grad_list = torch.cat((self.grad_list,grad[in_mask.bool()].view(-1)))
            info.update(
                self.find_pruning_channel(module, fisher, in_mask, info))
                
        return info

    def channel_prune(self):
        """Select the channel in model with smallest fisher / delta set
        corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e15}
        self.fisher_list = torch.tensor([]).cuda()
        self.mag_list = torch.tensor([]).cuda()
        self.grad_list = torch.tensor([]).cuda()
        self.fisher_reg = None
        info.update(self.single_prune(info, self.group_modules))
        for group in self.groups:
            # they share the same in mask
            in_mask = self.groups[group][0].in_mask.view(-1)
            fisher = self.accum_fishers[group].double()
            mag = self.accum_mags[group].double()
            grad = self.accum_grads[group].double()
            if self.delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
                mag /= float(self.flops[group] / 1e9)
                grad /= float(self.flops[group] / 1e9)
            elif self.delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
                mag /= float(self.acts[group] / 1e6)
                grad /= float(self.acts[group] / 1e6)
            self.fisher_list = torch.cat((self.fisher_list,fisher[in_mask.bool()].view(-1)))
            self.mag_list = torch.cat((self.mag_list,mag[in_mask.bool()].view(-1)))
            self.grad_list = torch.cat((self.grad_list,grad[in_mask.bool()].view(-1)))
            info.update(self.find_pruning_channel(group, fisher, in_mask, info))
                
        module, channel = info['module'], info['channel']
        if self.trained_mask or self.noise_mask:
            pass
        else:
            # only modify in_mask is sufficient
            if isinstance(module, int):
                # the case for multiple modules in a group
                for m in self.groups[module]:
                    m.in_mask[channel] = 0
            elif module is not None:
                # the case for single module
                module.in_mask[channel] = 0
            
    def accumulate_fishers(self):
        """Accumulate all the fisher during self.interval iterations."""

        for module, name in self.conv_names.items():
            self.accum_fishers[module] += self.batch_fishers[module]
            self.accum_mags[module] += self.batch_mags[module]
            self.accum_grads[module] += self.batch_grads[module]
        for group in self.groups:
            self.accum_fishers[group] += self.batch_fishers[group]
            self.accum_mags[group] += self.batch_mags[group]
            self.accum_grads[group] += self.batch_grads[group]

    def group_fishers(self):
        """Accumulate all module.in_mask's fisher and flops in same group."""
        # the case for groups
        for group in self.groups:
            self.flops[group] = 0
            self.acts[group] = 0
            # impact on group members
            for module in self.groups[group]:
                # accumulate fisher per channel per batch
                module_fisher = self.temp_fisher_info[module]
                self.temp_fisher_info[group] += module_fisher 
                # accumulate flops per in_channel per batch for each group
                if hasattr(module, 'child'):
                    child = self.name2module[module.child]
                    if hasattr(child,'group_master'):
                        child = self.name2module[child.group_master]
                    chans_o = child.in_mask.sum()
                else:
                    chans_o = module.out_channels
                delta_flops = self.flops[module] // module.in_channels // \
                    module.out_channels * chans_o
                self.flops[group] += delta_flops

            # sum along the dim of batch
            self.batch_fishers[group] = self.temp_fisher_info[group]**2
            self.batch_mags[group] = self.temp_mag_info[module]
            self.batch_grads[group] = self.temp_grad_info[module]

            # impact on group ancestors, whose out channels are coupled with its
            # in_channels
            for module in self.ancest[group]:
                delta_flops = self.flops[module] // module.out_channels // \
                    module.in_channels * module.in_mask.sum()
                self.flops[group] += delta_flops
                acts = self.acts[module] // module.out_channels
                self.acts[group] += acts
        # the case for single modules
        for module, name in self.conv_names.items():
            self.batch_fishers[module] = self.temp_fisher_info[module]**2
            self.batch_mags[module] = self.temp_mag_info[module]
            self.batch_grads[module] = self.temp_grad_info[module]

    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.conv_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self):
        """Clear fisher info of single conv and group."""
        for module, name in self.conv_names.items():
            self.temp_fisher_info[module].zero_()
            self.temp_mag_info[module].zero_()
            self.temp_grad_info[module].zero_()
        for group in self.groups:
            self.temp_fisher_info[group].zero_()
            self.temp_mag_info[group].zero_()
            self.temp_grad_info[group].zero_()

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

        def backward_hook(grad_feature):
            def compute_fisher(input, grad_input, layer_name):
                # information per mask channel per module
                grads = input * grad_input
                if layer_name in ['Conv2d']:
                    grads = grads.sum(-1).sum(-1).sum(0)
                else:
                    print('Unrecognized in compute_fisher:',layer_name)
                    exit(0)
                return grads
                
            def compute_mag(input, grad_input, layer_name):
                # information per mask channel per module
                grads = torch.abs(input)
                if layer_name in ['Conv2d']:
                    grads = grads.sum(-1).sum(-1).sum(0)
                else:
                    print('Unrecognized in compute_fisher:',layer_name)
                    exit(0)
                return grads
                
            def compute_grad(input, grad_input, layer_name):
                # information per mask channel per module
                grads = torch.abs(grad_input)
                if layer_name in ['Conv2d']:
                    grads = grads.sum(-1).sum(-1).sum(0)
                else:
                    print('Unrecognized in compute_fisher:',layer_name)
                    exit(0)
                return grads

            layer_name = type(module).__name__
            feature = self.conv_inputs[module].pop(-1)[0]
            self.temp_fisher_info[module] += compute_fisher(feature, grad_feature, layer_name)
            self.temp_mag_info[module] += compute_mag(feature, grad_feature, layer_name)
            self.temp_grad_info[module] += compute_grad(feature, grad_feature, layer_name)
            
        # alternate way to compute fisher information
        #if inputs[0].requires_grad:
            #inputs[0].register_hook(backward_hook)
        #    self.conv_inputs[module].append(inputs)

    def compute_fisher_backward(self, module):
        # there are some bugs in torch, not using the backward hook
        """
        Args:
            module (nn.Module): module register hooks
            grad_input (tuple): tuple contains grad of input and parameters,
                grad_input[0]is the grad of input in Pytorch 1.3, it seems
                has changed in Higher version
        """
        def compute_fisher(weight, grad_weight, layer_name):
            # information per mask channel per module
            grads = weight*grad_weight
            if layer_name in ['Conv2d']:
                grads = grads.sum(-1).sum(-1).sum(0)
            else:
                print('Unrecognized in compute_fisher:',layer_name)
                exit(0)
            return grads
        
        def compute_mag(weight, grad_weight, layer_name):
            # information per mask channel per module
            grads = torch.abs(weight)
            if layer_name in ['Conv2d']:
                grads = grads.sum(-1).sum(-1).sum(0)
            else:
                print('Unrecognized in compute_fisher:',layer_name)
                exit(0)
            return grads
            
        def compute_grad(weight, grad_weight, layer_name):
            # information per mask channel per module
            grads = torch.abs(grad_weight)
            if layer_name in ['Conv2d']:
                grads = grads.sum(-1).sum(-1).sum(0)
            else:
                print('Unrecognized in compute_fisher:',layer_name)
                exit(0)
            return grads

        layer_name = type(module).__name__
        weight = module.weight
        self.temp_fisher_info[module] += compute_fisher(weight, weight.grad, layer_name)
        self.temp_mag_info[module] += compute_mag(weight, weight.grad, layer_name)
        self.temp_grad_info[module] += compute_grad(weight, weight.grad, layer_name)

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
        ln2ancest = {}
        for n, m in model.named_modules():
            if type(m).__name__ not in ['Conv2d','BatchNorm2d']:
                continue
            # independent nets
            if 'conv1' == n:
                ancest_name = []
            elif 'conv' in n:
                a,b,c = re.findall(r'\d+',n)
                if c == '1':
                    if b == '0':
                        if a == '1':
                            ancest_name = ['conv1']
                        elif a == '2':
                            ancest_name = ['layer1.8.conv2']
                        else:
                            ancest_name = ['layer2.8.conv2']
                    else:
                        if a == '1':
                            ancest_name = ['layer1.0.conv2',f'layer1.{int(b)-1}.conv2']
                        elif a == '2':
                            ancest_name = ['layer2.0.conv2',f'layer2.{int(b)-1}.conv2']
                        else:
                            ancest_name = ['layer3.0.conv2',f'layer3.{int(b)-1}.conv2']
                else:
                    ancest_name = [f'layer{a}.{b}.conv1']

            if type(m).__name__ in ['Conv2d']:
                conv2ancest[m] = []
            else:
                ln2ancest[m] = []
            for name in ancest_name:
                if type(m).__name__ in ['Conv2d']:
                    conv2ancest[m] += [self.name2module[name]]
                else:
                    ln2ancest[m] += [self.name2module[name]]
                    
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
        self.ln2ancest = ln2ancest

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
        module.trained_mask = self.trained_mask
        limit = float(10)
        module.noise_mask = self.noise_mask
        module.finetune = not pruning
        if type(module).__name__ == 'Conv2d':
            all_ones = module.weight.new_ones(module.in_channels,)
            half_ones_zeros = module.weight.new_ones(module.in_channels)
            half_ones_zeros[module.in_channels//2:] = 0
            module.register_buffer('in_mask', half_ones_zeros)
            if self.trained_mask:
                module.register_buffer(
                    'soft_mask', torch.nn.Parameter(torch.randn(module.in_channels)).to(module.weight.device))
            def modified_forward(m, x):
                if self.use_mask:
                    if not m.finetune:
                        if m.trained_mask:
                            if hasattr(m, 'group_master'):
                                mask = F.sigmoid(self.name2module[m.group_master].soft_mask)
                            else:
                                mask = F.sigmoid(m.soft_mask)
                            m.in_mask[:] = mask.data
                            mask = mask.view(1,-1,1,1)
                            x = x * mask.to(x.device)
                        elif m.noise_mask:
                            mask = m.in_mask.view(1,-1,1,1).to(x.device)
                            noise = torch.empty_like(x).uniform_(-limit, limit)*mask
                            x = x + noise
                        else:
                            mask = m.in_mask.view(1,-1,1,1)
                            x = x * mask.to(x.device)
                    else:
                        # if it has no ancestor
                        # we need to mask it
                        if x.size(1) == len(m.in_mask):
                            x = x[:,m.in_mask.bool(),:,:]
                output = F.conv2d(x, m.weight, m.bias, m.stride, m.padding, m.dilation, m.groups)
                m.output_size = output.size()
                return output
            module.forward = MethodType(modified_forward, module) 
        if  type(module).__name__ == 'BatchNorm2d':
            # no need to modify layernorm during pruning since it is not computed over channels
            pass


    def deploy_pruning(self, model):
        """To speed up the finetune process, We change the shape of parameter
        according to the `in_mask` and `out_mask` in it."""

        for name, module in model.named_modules():
            if type(module).__name__ == 'Conv2d':
                module.finetune = True
                requires_grad = module.weight.requires_grad
                if hasattr(module, 'child'):
                    child = self.name2module[module.child]
                    if hasattr(child,'group_master'):
                        child = self.name2module[child.group_master]
                    out_mask = child.in_mask.bool()
                else:
                    out_mask = None
                in_mask = module.in_mask.bool()
                if hasattr(module, 'bias') and module.bias is not None and out_mask is not None:
                    module.bias = nn.Parameter(module.bias.data[out_mask])
                    module.bias.requires_grad = requires_grad
                if out_mask is not None:
                    temp_weight = module.weight.data[out_mask]
                    module.weight = nn.Parameter(temp_weight[:, in_mask].data)
                    module.weight.requires_grad = requires_grad

            elif type(module).__name__ == 'BatchNorm2d':
                if hasattr(module, 'child'):
                    child = self.name2module[module.child]
                    if hasattr(child,'group_master'):
                        child = self.name2module[child.group_master]
                    out_mask = child.in_mask.bool()
                else:
                    out_mask = None
                requires_grad = module.weight.requires_grad
                if out_mask is not None:
                    module.normalized_shape = (int(out_mask.sum()),)
                    module.weight = nn.Parameter(module.weight.data[out_mask].data)
                    module.bias = nn.Parameter(module.bias.data[out_mask].data)
                    module.weight.requires_grad = requires_grad
                    module.bias.requires_grad = requires_grad