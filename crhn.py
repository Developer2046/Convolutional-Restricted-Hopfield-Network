#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 13:20:43 2026

@author: edward
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from .rhn import Nonlinear_RHN
import os

class SignSTE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(input >= 0,
                           torch.ones_like(input),
                           -torch.ones_like(input))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        # Clipped STE: derivative = 1 if |x| <= 1
        grad_input = grad_output.clone()
        grad_input[input.abs() > 1] = 0

        return grad_input

class Con_RHN(nn.Module):
    def __init__(self, kernel_sizes=[3, 3, 3, 3, 3, 3], num_filters=[3, 4, 8, 16, 8, 4], \
                 fc_dims=[512, 256, 128], rhn_arch=[128, 100], input_size=32, device='cuda'):
        
        super(Con_RHN, self).__init__()
        
        assert fc_dims[-1] == rhn_arch[0], f"Mismatch: crhn_last={fc_dims[-1]}, but rhn first={rhn_arch[0]}"
        
        self.device = device
        self.cov_rhn = CRHN(kernel_sizes=kernel_sizes, num_filters=num_filters, \
                            fc_dims=fc_dims, input_size=input_size, act_name="tanh", device=self.device)
        self.rhn = Nonlinear_RHN(arch = rhn_arch, act_type='tanh', boost=False, device=self.device)
        
        self.convergence_threshold = 1e-3
        self.max_iterations = 50
        

    def save_models(self, index, dataset, path):
        # create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
    
        conv_path = os.path.join(
            path, f"conv_rhn_{dataset}_{self.cov_rhn.arch}_{index}.pth"
        )
        rhn_path = os.path.join(
            path, f"rhn_{dataset}_{self.rhn.arch}_{index}.pth"
        )
    
        torch.save(self.cov_rhn.state_dict(), conv_path)
        torch.save(self.rhn.state_dict(), rhn_path)
    
        return True
    
    
    def load_models(self, index, dataset, path, device="cpu"):
        conv_path = os.path.join(
            path, f"conv_rhn_{dataset}_{self.cov_rhn.arch}_{index}.pth"
        )
        rhn_path = os.path.join(
            path, f"rhn_{dataset}_{self.rhn.arch}_{index}.pth"
        )
    
        try:
            self.cov_rhn.load_state_dict(torch.load(conv_path, map_location=device))
            self.rhn.load_state_dict(torch.load(rhn_path, map_location=device))
        
            print(f"[SUCCESS] Models loaded for dataset='{dataset}', index={index}")
            print(f"  -> Conv: {conv_path}")
            print(f"  -> RHN : {rhn_path}")
            return True
    
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            return False
        
    
    def get_rhn_input(self, x):
        x = x.to(self.device)
        input_rhn = self.cov_rhn.forward_to_rhn(x)
        return input_rhn
        
    
    def forward(self, x):
        
        x = x.to(self.device)
        
        input_rhn = self.cov_rhn.forward_to_rhn(x)
        
        output_cov = self.rhn.forward(input_rhn, logit=False)
        
        input_cov = SignSTE.apply(output_cov)
        
        recon = self.cov_rhn.recon_from_rhn(input_cov)
        
        return recon
    
    def train_all_model(self, x, epochs, lr):
        
        self.cov_rhn.train_model(x, epochs=epochs, lr=lr)
        
        input_rhn = self.cov_rhn.forward_to_rhn(x)
        
        
        for ix in range(100):
            self.rhn.right_train_layer(input_rhn)
            self.rhn.left_train_layer(input_rhn)
            
        return True
    

    @torch.no_grad()
    def query(self, x, max_iterations=None, threshold=None, return_history=False):
        """
        Iterative auto-associative retrieval.
        Corrupted input -> repeated reconstruction -> stable memory attractor.
        """
        prev = x.to(self.device)

        if max_iterations is None:
            max_iterations = self.max_iterations
        if threshold is None:
            threshold = self.convergence_threshold

        #prev = x.clone()
        history = [prev.clone()] if return_history else None
        with torch.no_grad():
            for ix in range(max_iterations):
                new_x = self.forward(prev)
                delta = F.mse_loss(new_x, prev)
    
                print(f"Iteration {ix+1}, Loss: {delta.item():.6f}")
    
                if return_history:
                    history.append(new_x.clone())
    
                if delta.item() < threshold:
                    return (new_x, history) if return_history else new_x
    
                prev = new_x

        return (prev, history) if return_history else prev
    


class CRHN(nn.Module):
    """
    Convolutional Restricted Hopfield-style Auto-Associative Memory.

    Design goals:
    1. Encode input image -> compact latent memory code
    2. Decode latent memory code -> reconstruct original image
    3. Tied conv/deconv weights
    4. Tied fc encoder/decoder weights
    5. Flexible kernel sizes with automatic output_padding computation
    6. Exact output spatial size matching input_size
    """

    def __init__(
        self,
        kernel_sizes=[3, 3, 3],
        num_filters=[3, 16, 32, 64],
        fc_dims=[512, 256],
        input_size=96,
        in_channels=3,
        stride=2,
        act_name="tanh",
        use_sign_latent=True,
        device="cuda"
    ):
        super().__init__()

        assert len(num_filters) >= 2, "num_filters must have at least 2 entries."
        assert len(kernel_sizes) == len(num_filters) - 1, \
            "len(kernel_sizes) must equal len(num_filters)-1"

        if num_filters[0] != in_channels:
            raise ValueError(
                f"num_filters[0] must equal in_channels. Got {num_filters[0]} vs {in_channels}."
            )

        self.input_size = input_size
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.num_layers = len(num_filters) - 1
        self.stride = stride
        self.device = device
        self.use_sign_latent = use_sign_latent
        self.arch = {
            "kernel_sizes": kernel_sizes,
            "num_filters": num_filters,
            "fc_dims": fc_dims,
            "input_size": input_size,
            "stride": stride,
            "act_name": act_name,
            "use_sign_latent": use_sign_latent,
        }

        self.max_iterations = 50
        self.convergence_threshold = 1e-3

        if act_name == "tanh":
            self.act = torch.tanh
        elif act_name == "relu":
            self.act = nn.LeakyReLU(negative_slope=0.1)
        else:
            raise ValueError("act_name must be 'tanh' or 'relu'.")

        # --------------------------------------------------
        # Build encoder shape schedule
        # --------------------------------------------------
        self.encoder_shapes = []   # list of (C, H, W) after each conv block
        h = input_size
        w = input_size

        for i in range(self.num_layers):
            k = kernel_sizes[i]
            p = k // 2
            h = math.floor((h + 2 * p - k) / stride + 1)
            w = math.floor((w + 2 * p - k) / stride + 1)
            self.encoder_shapes.append((num_filters[i + 1], h, w))

        self.conv_out_channels, self.final_h, self.final_w = self.encoder_shapes[-1]
        self.conv_output_dim = self.conv_out_channels * self.final_h * self.final_w

        # --------------------------------------------------
        # Shared convolution kernels
        # --------------------------------------------------
        self.shared_kernels = nn.ParameterList([
            nn.Parameter(
                torch.randn(
                    num_filters[i + 1],
                    num_filters[i],
                    kernel_sizes[i],
                    kernel_sizes[i]
                ) * 0.1
            )
            for i in range(self.num_layers)
        ])

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            k = kernel_sizes[i]
            p = k // 2
            conv = nn.Conv2d(
                in_channels=num_filters[i],
                out_channels=num_filters[i + 1],
                kernel_size=k,
                stride=stride,
                padding=p,
                bias=False
            )
            conv.weight = nn.Parameter(self.shared_kernels[i])
            self.convs.append(conv)

        # --------------------------------------------------
        # Fully connected memory layers
        # --------------------------------------------------
        full_fc_dims = [self.conv_output_dim] + fc_dims

        self.shared_fc_weights = nn.ParameterList([
            nn.Parameter(
                torch.nn.init.orthogonal_(
                    torch.empty(full_fc_dims[i + 1], full_fc_dims[i])
                )
            )
            for i in range(len(full_fc_dims) - 1)
        ])

        self.fcs = nn.ModuleList([
            nn.Linear(full_fc_dims[i], full_fc_dims[i + 1], bias=False)
            for i in range(len(full_fc_dims) - 1)
        ])
        for i, fc in enumerate(self.fcs):
            fc.weight = nn.Parameter(self.shared_fc_weights[i])

        # decoder fc: tied transpose weights
        self.fcs_output = nn.ModuleList([
            nn.Linear(full_fc_dims[i + 1], full_fc_dims[i], bias=False)
            for i in reversed(range(len(full_fc_dims) - 1))
        ])
        for out_idx, src_idx in enumerate(reversed(range(len(full_fc_dims) - 1))):
            self.fcs_output[out_idx].weight = nn.Parameter(self.shared_fc_weights[src_idx].T)

        # latent dimension = last fc dim
        self.latent_dim = full_fc_dims[-1]

        # --------------------------------------------------
        # Build deconvs with automatic output_padding
        # --------------------------------------------------
        # target spatial sizes in reverse:
        # final_h -> previous_h -> ... -> input_size
        # decoder_target_shapes = [(in_channels, input_size, input_size)]
        # decoder_target_shapes = [(num_filters[i],) + self._shape_before_layer(i) for i in range(self.num_layers)]
        # decoder_target_shapes gives target output of each transpose conv in forward order
        # i = num_layers-1 ... 0 during construction

        self.deconvs = nn.ModuleList()
        current_h, current_w = self.final_h, self.final_w

        for i in reversed(range(self.num_layers)):
            k = kernel_sizes[i]
            p = k // 2
            target_h, target_w = self._shape_before_layer(i)

            base_h = (current_h - 1) * stride - 2 * p + k
            base_w = (current_w - 1) * stride - 2 * p + k

            out_pad_h = target_h - base_h
            out_pad_w = target_w - base_w

            if out_pad_h < 0 or out_pad_h >= stride or out_pad_w < 0 or out_pad_w >= stride:
                raise ValueError(
                    f"Invalid output_padding at layer {i}: "
                    f"(h={out_pad_h}, w={out_pad_w}) with stride={stride}. "
                    f"Try different kernel sizes or stride."
                )

            deconv = nn.ConvTranspose2d(
                in_channels=num_filters[i + 1],
                out_channels=num_filters[i],
                kernel_size=k,
                stride=stride,
                padding=p,
                output_padding=(out_pad_h, out_pad_w),
                bias=False
            )
            deconv.weight = nn.Parameter(self.shared_kernels[i])
            self.deconvs.append(deconv)

            current_h, current_w = target_h, target_w

        self.to(device)

    def _shape_before_layer(self, layer_idx):
        """
        Spatial shape before encoder conv[layer_idx].
        For layer 0, it is input_size x input_size.
        For layer i>0, it is encoder_shapes[i-1][1:].
        """
        if layer_idx == 0:
            return self.input_size, self.input_size
        return self.encoder_shapes[layer_idx - 1][1], self.encoder_shapes[layer_idx - 1][2]

    def _match_input_size(self, x):
        """
        Final safety correction: crop if slightly oversized.
        """
        target_h = self.input_size
        target_w = self.input_size
        return x[:, :, :target_h, :target_w]


    def encode_conv(self, x):
        x = x.to(self.device)
        for conv in self.convs:
            x = self.act(conv(x))
        return x

    def encode(self, x):
        """
        Input image -> latent memory code
        """
        x = self.encode_conv(x)
        x = x.reshape(x.size(0), -1)

        pre_x = None
        for fc in self.fcs:
            pre_x = fc(x)
            x = self.act(pre_x)

        if self.use_sign_latent:
            x = SignSTE.apply(pre_x)
        else:
            x = pre_x

        return x

    def decode(self, z):
        """
        Latent memory code -> reconstructed image
        """
        x = z.to(self.device)
        for fc_output in self.fcs_output:
            x = self.act(fc_output(x))

        x = x.reshape(z.size(0), self.conv_out_channels, self.final_h, self.final_w)

        # deconvs were appended from deep->shallow, so use them in listed order
        for deconv in self.deconvs:
            x_pre = deconv(x)
            x = self.act(x_pre)

        x = self._match_input_size(x_pre)
        return x

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon

    def forward_to_rhn(self, x):
        """
        Alias: image -> latent memory state
        """
        return self.encode(x)

    def recon_from_rhn(self, z):
        """
        Alias: latent memory state -> image
        """
        return self.decode(z)

    def loss(self, outputs, targets, ortho_lambda=0.0):
        rec = F.mse_loss(outputs, targets)
        return rec

    def train_model(self, images, epochs=100, lr=1e-3, verbose_every=100):
        images = images.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.forward(images)

            if outputs.shape != images.shape:
                raise RuntimeError(
                    f"Output shape {outputs.shape} does not match target shape {images.shape}"
                )

            loss = self.loss(outputs, images)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % verbose_every == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")
            

        return True

    @torch.no_grad()
    def query(self, x, max_iterations=None, threshold=None, return_history=False):
        """
        Iterative auto-associative retrieval.
        Corrupted input -> repeated reconstruction -> stable memory attractor.
        """
        prev = x.to(self.device)

        if max_iterations is None:
            max_iterations = self.max_iterations
        if threshold is None:
            threshold = self.convergence_threshold

        #prev = x.clone()
        history = [prev.clone()] if return_history else None
        with torch.no_grad():
            for ix in range(max_iterations):
                new_x = self.forward(prev)
                delta = F.mse_loss(new_x, prev)
    
                print(f"Iteration {ix+1}, Loss: {delta.item():.6f}")
    
                if return_history:
                    history.append(new_x.clone())
    
                if delta.item() < threshold:
                    return (new_x, history) if return_history else new_x
    
                prev = new_x

        return (prev, history) if return_history else prev
