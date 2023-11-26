import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lpips
from collections import OrderedDict
from timeit import default_timer as timer
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

from data.utils import IMG_INV_TRANSFORMS, DEPTH_INV_TRANSFORMS
from modules.utils import get_model_size

import tinycudann as tcnn
import commentjson as json

from models.FactorFields import FactorFields as Dif
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from .common import *

def fourier_mapping(x, B):
    """Fourier mapping.
    
    Args:
    ----------
        x (torch.Tensor): Input tensor.
        B (torch.Tensor): Fourier basis.
        
    Returns:
    ----------
        torch.Tensor: Fourier mapped tensor.
    
    Notes:
    ----------
        Adapted from Tancik et al. (2020) `<https://github.com/tancik/fourier-feature-networks>`_.
"""
    
    if B is None:
        return x
    else:
        x_proj = (2.*torch.pi*x) @ B.T
        return torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SineLayer(nn.Module):
    """Sine layer.
    
    Args:
    ----------
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to use bias. Defaults to True.
        is_first (bool, optional): Whether it is the first layer. Defaults to False.
        omega_0 (int, optional): Omega_0. Defaults to 30.
        
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_."""
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    """Siren model.
    
    Args:
    ----------
        in_features (int): 
            Number of input features.
        hidden_features (int): 
            Number of hidden features.
        hidden_layers (int): 
            Number of hidden layers.
        out_features (int): 
            Number of output features.
        outermost_linear (bool, optional): 
            Whether to use linear activation for the outermost layer. Defaults to False.
        first_omega_0 (int, optional): 
            Omega_0 for the first layer. Defaults to 30.
        hidden_omega_0 (int, optional): 
            Omega_0 for the hidden layers. Defaults to 30.
        fourier_scale (float, optional): 
            Fourier scale. Defaults to None.
        
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_.
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., fourier_scale=None):
        super().__init__()
        
        self.net = []
        if fourier_scale is None:
            self.B = None
            self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        elif fourier_scale == 'eye':
            self.B = torch.eye(2)
            self.net.append(SineLayer(in_features*2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        else:
            self.B = torch.normal(0, fourier_scale, (32, in_features))
            self.net.append(SineLayer(32*2, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
        if fourier_scale is not None:
            self.weights_init()
    
    def weights_init(self):
        self.B = nn.Parameter(self.B, requires_grad=False)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(fourier_mapping(coords, self.B))
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def net_laplace(y, x):
    """Compute the Laplacian.
    
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_.
    """
    grad = net_gradient(y, x)
    return net_divergence(grad, x)


def net_divergence(y, x):
    """"Compute the divergence
    
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_.
    """
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def net_gradient(y, x, grad_outputs=None):
    """Compute the gradient.
    
    Notes:
    ----------
        Adapted from official SIREN implementation`<https://github.com/vsitzmann/siren>`_."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad




class CoordinateBasedMLP(nn.Module):
    """Coordinate-based MLP.
    
    Args:
    ----------
        in_features (int):
            Number of input features.
        hidden_features (int):
            Number of hidden features.
        hidden_layers (int):
            Number of hidden layers.
        out_features (int):
            Number of output features.
        outermost_linear (bool, optional):
            Whether to use linear activation for the outermost layer. Defaults to False.
        fourier_scale (float, optional):
            Fourier scale. Defaults to None.
        
    Notes:
    ----------
        Adapted from Tancik et al. (2020) `<https://github.com/tancik/fourier-feature-networks>`_. 
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, fourier_scale=None):
        super().__init__()
        
        self.net = []
        if fourier_scale is None:
            self.B = None
            self.net.append(nn.Linear(in_features, hidden_features))
        else:
            self.B = torch.normal(0, fourier_scale, (hidden_features//2, in_features))
            # self.B = torch.randn(hidden_features//2, in_features) * fourier_scale
            self.net.append(nn.Linear(hidden_features, hidden_features))


        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU())

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            self.net.append(final_linear)
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*self.net)
        if fourier_scale is not None:
            self.weights_init()
    
    def weights_init(self):
        self.B = nn.Parameter(self.B, requires_grad=False)
    
    def forward(self, x):
        x = x.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(fourier_mapping(x, self.B))
        return output, x    


class SobelLayer(nn.Module):
    """Sobel filter layer.
    
    Args:
    ----------
        p_norm (int): p-norm to use.
    """
    def __init__(self, p_norm) -> None:
        super().__init__()
        self.gx_filter = nn.Conv2d(3, 1, (3, 3), bias=False, padding=(3//2, 3//2), padding_mode='replicate')
        self.gy_filter = nn.Conv2d(3, 1, (3, 3), bias=False, padding=(3//2, 3//2), padding_mode='replicate')
        self.p_norm = p_norm
        self.weights_init()
    
    def weights_init(self):
        """Initialize weights of the Sobel filter."""
        weights_gy = torch.stack([1/3 * torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) for _ in range(3)])
        weights_gx = torch.stack([1/3 * torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) for _ in range(3)])
        self.gx_filter.weight = nn.Parameter(weights_gx.view(1,3,3,3).type(torch.FloatTensor), requires_grad=False)
        self.gy_filter.weight = nn.Parameter(weights_gy.view(1,3,3,3).type(torch.FloatTensor), requires_grad=False)

    def forward(self, x):
        """Forward pass.
        
        Args:
        ----------
            x (torch.Tensor): Input tensor.

        Returns:
        ----------
            torch.Tensor: Gradient magnitude."""
        gx = self.gx_filter(x)
        gy = self.gy_filter(x)

        return torch.norm(torch.cat([gx,gy], dim=1), dim=1, p=self.p_norm, keepdim=True)


class GradientFilter(nn.Module):
    """Gradient filter layer.
    
    Args:
    ----------
        p_norm (int): p-norm to use.
    """
    def __init__(self, p_norm) -> None:
        super().__init__()
        self.gx_filter = nn.Conv2d(3, 1, (1, 3), bias=False, padding=(1,0), padding_mode='replicate')
        self.gy_filter = nn.Conv2d(3, 1, (3, 1), bias=False, padding=(0,1), padding_mode='replicate')
        self.p_norm = p_norm
        self.weights_init()
    
    def weights_init(self):
        """Initialize weights of the Sobel filter."""
        weights_gy = torch.stack([torch.tensor([[-1/3, 0, 1/3]]) for _ in range(3)])
        weights_gx = torch.stack([torch.tensor([[-1/3], [0], [1/3]]) for _ in range(3)])
        self.gx_filter.weight = nn.Parameter(weights_gx.view(1,3,3,1).type(torch.FloatTensor), requires_grad=False)
        self.gy_filter.weight = nn.Parameter(weights_gy.view(1,3,1,3).type(torch.FloatTensor), requires_grad=False)

    def forward(self, x):
        """Forward pass.
        
        Args:
        ----------
            x (torch.Tensor): Input tensor.

        Returns:
        ----------
            """
        gx = self.gx_filter(x)
        gy = self.gy_filter(x)

        return torch.norm(torch.cat([gx,gy], dim=1), dim=1, p=self.p_norm, keepdim=True)


class Dictionary_field(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, fourier_scale=None):
        super().__init__()
        base_conf = OmegaConf.load('./configs/defaults.yaml')
        second_conf = OmegaConf.load('./configs/image.yaml')
        cfg = OmegaConf.merge(base_conf, second_conf)
       
        H = 256
        W = 192
        cfg.dataset.aabb = [[0,0], [H,W]]
        self.net = Dif(cfg,'cuda')

    def forward(self, x):
        feats, _ = self.net.get_coding(x.to('cuda'))
        output = self.net.linear_mat(feats)    
        return output, x 


nn_dict = {
    'SIREN': Siren,
    'FOURIER_MAPPED_MLP': CoordinateBasedMLP,
    'DICTIONARY_FIELD' : Dictionary_field
}

def dif_PSNR(a,b):
    if type(a).__module__ == np.__name__:
        mse = np.mean((a-b)**2)
    else:
        mse = torch.mean((a-b)**2).item()
    psnr = -10.0 * np.log(mse) / np.log(10.0)
    return psnr

def skip(
        num_input_channels=2, num_output_channels=3, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True):
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    last_scale = n_scales - 1 

    cur_depth = None

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip, deeper))
        else:
            model_tmp.add(deeper)
        
        model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.add(bn(num_channels_skip[i]))
            skip.add(act(act_fun))
            
        # skip.add(Concat(2, GenNoise(nums_noise[i]), skip_part))

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))


        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model
        


def deblurring(blur_dataset, blur_nn: nn.Module, deblur_params:dict, device:torch.DeviceObjType, load_ckpt:bool=False, ckpt_path:str=None):
    """Sharp neural representations from blur function.
    
    Args:
    ----------
        blur_dataset (data.datasets.BaseCMBFitting): Blur dataset.
        blur_nn (nn.Module): Blurring neural network.
        deblur_params (dict): Deblurring parameters.
        device (torch.DeviceObjType): Device to use.
        load_ckpt (bool): Whether to load checkpoint.
        ckpt_path (str): Checkpoint path.

    Returns:
    ----------
        dict: Deblurring evaluation metrics.
        numpy.ndarray: Deblurred image.
        numpy.ndarray: Ground truth image.
        nn.Module: Implicit neural representation.
    """

    # Set up dataloader
    batch_size = 1
    train_dataloader = DataLoader(blur_dataset, batch_size=batch_size, pin_memory=True, num_workers=0)
    # lpips network
    lpips_fn = lpips.LPIPS(net='vgg')

    start_time = timer()
    # Set up implicit neural network
    
    if deblur_params['nn_model'] == 'HASH_ENCODING':    
        with open("configs/config_hash.json") as f:
                config = json.load(f)
        
        mlp_nn = tcnn.NetworkWithInputEncoding(n_input_dims=2, n_output_dims=3, encoding_config=config["encoding"], network_config=config["network"])
        num_epochs = 200
    else:
        nn_model = nn_dict[deblur_params['nn_model']]
        mlp_nn = nn_model(in_features=2, out_features=3, **deblur_params[deblur_params['nn_model']], outermost_linear=True)
        mlp_nn.to(device)
        num_epochs = 30 *deblur_params['num_epochs']
    

    mlp_nn.to(device)
    
    # Set up optimization parameters

    optimizer = torch.optim.Adam(lr=deblur_params['lr'], params=mlp_nn.parameters())
    clip_grad = deblur_params['clip_grad']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=deblur_params['scheduler_eta_min'])
    


    criterion = nn.MSELoss()
    if deblur_params['gradient_fn'] == 'filter':
        grad_nn = GradientFilter(p_norm=deblur_params['p_norm'])
        grad_nn.to(device)
    elif deblur_params['gradient_fn'] == 'net_grad':
        grad_nn = net_gradient
    elif deblur_params['gradient_fn'] == 'u-net':
        input_depth = 3

        unet = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        unet = unet.type(torch.cuda.FloatTensor).to(device)
    else:
        raise ValueError("Not implemented 'gradient_fn': {}".format(deblur_params['gradient_fn']))

    beta = deblur_params['gradient_weight']
    blur_nn.to(device)
    
    padding = (blur_nn.ks//2)
    img_padded_size = list(map(lambda x: x + 2*padding, blur_dataset.img_size))
    patch_padded_size = list(map(lambda x: x + 2*padding, blur_dataset.patch_size))

    if not load_ckpt:
        # Training loop
        for _ in tqdm(range(num_epochs)):
            # Iterate over batches
            for sample in train_dataloader:
                input_coords, gt_blur = sample['coords'].to(device), sample['blurry'].to(device)
                idx = sample['idx']
                optimizer.zero_grad()
                # Evaluate implicit MLP
                if deblur_params['nn_model'] == 'HASH_ENCODING':    
                    mlp_sharp_ravel = mlp_nn(input_coords.permute(0,2,3,1).view(-1,2))
                    mlp_coords = input_coords.type(torch.FloatTensor)
                else:
                    mlp_sharp_ravel, mlp_coords = mlp_nn(input_coords.permute(0,2,3,1).view(-1,2))

                
                mlp_sharp = mlp_sharp_ravel.view(1,*patch_padded_size,3).permute(0,3,1,2)
                # Evaluate blur function
                pred_blur = blur_nn(mlp_sharp.type(torch.cuda.FloatTensor), idx)
                # Compute loss
                
                
                loss = criterion(pred_blur, gt_blur)
                if deblur_params['gradient_fn'] == 'filter':
                    mlp_grad = grad_nn(mlp_sharp.type(torch.cuda.FloatTensor))
                    mlp_grad = torch.squeeze(mlp_grad)
                    mlp_grad = mlp_grad.permute(1, 0).reshape((-1, 1))
                    loss += beta * mlp_grad.mean()
                elif deblur_params['gradient_fn'] == "net_grad":
                    mlp_grad = grad_nn(mlp_sharp_ravel, mlp_coords)
                    loss += beta * torch.norm(mlp_grad, dim=1, p=deblur_params['p_norm']).mean()
                elif deblur_params['gradient_fn'] == 'u-net':                    
                    mlp_grad = unet(mlp_sharp.type(torch.cuda.FloatTensor))
                    loss += beta * mlp_grad.mean()
                loss.backward()
                
                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(mlp_nn.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(mlp_nn.parameters(), max_norm=clip_grad)
                optimizer.step()
            scheduler.step()
    else:
        mlp_nn.load_state_dict(torch.load(ckpt_path))
        mlp_nn.eval()

    pred_sharp = []
    # Evaluate sharp image from implicit MLP


    with torch.no_grad():
        # Iterate over batches

        for sample in train_dataloader:
            input_coords = sample['coords'].to(device)
            if deblur_params['nn_model'] == 'HASH_ENCODING':
                mlp_sharp = mlp_nn(input_coords.permute(0,2,3,1).view(-1,2))
            else:
                mlp_sharp, _ = mlp_nn(input_coords.permute(0,2,3,1).view(-1,2))
            mlp_sharp = mlp_sharp.view(1,*patch_padded_size,3).permute(0,3,1,2)
            pred_sharp.append(mlp_sharp[0,:,padding:-padding,padding:-padding])

    # Reconstruct sharp image
    
    sharp_est = torch.stack(pred_sharp, dim=-1).view(1,-1,len(blur_dataset))
    sharp_est = F.fold(sharp_est, output_size=blur_dataset.img_size, kernel_size=blur_dataset.patch_size, stride=blur_dataset.patch_size)
    sharp_est = IMG_INV_TRANSFORMS(sharp_est[0]).astype(np.float64)
    end_time = timer()
    
    
    
    # Compute metrics
    sharp_gt = blur_dataset.sharp

    
    psnr = dif_PSNR(sharp_est, sharp_gt) #peak_signal_noise_ratio(sharp_gt, sharp_est)
    ssim = structural_similarity(sharp_gt, sharp_est,  channel_axis=2)
    lpips_value = lpips_fn.forward(lpips.im2tensor(sharp_gt), lpips.im2tensor(sharp_est)).detach().numpy().squeeze()
    elapsed_time = end_time - start_time
    model_size = get_model_size(mlp_nn)
    # Concatenate metrics

    perf = {
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips_value,
        'elapsedTime': elapsed_time,
        'modelSize': model_size
    }
    return perf, sharp_est, sharp_gt, mlp_nn


    
