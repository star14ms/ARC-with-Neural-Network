import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelRelativeVectorExtractor(nn.Module):
    def __init__(self, n_range_search, W_kernel_max, H_kernel_max, pad_class_initial=0, pad_n_head=None, pad_dim_feedforward=1, dropout=0.1, pad_num_layers=1, bias=False, n_class=10):
        super().__init__()
        self.n_range_search = n_range_search
        self.W_kernel_max = W_kernel_max
        self.H_kernel_max = H_kernel_max
        self.pad_class_initial = pad_class_initial

        # self.attn_C = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(n_class+1, pad_n_head if pad_n_head is not None else n_class+1, pad_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
        #     num_layers=pad_num_layers,
        #     enable_nested_tensor=False,
        # )

    def forward(self, x):
        N, C, H, W = x.shape
        H_PAD = self.n_range_search if self.n_range_search != -1 else H - 1
        W_PAD = self.n_range_search if self.n_range_search != -1 else W - 1
        HL = 1 + 2*H_PAD
        WL = 1 + 2*W_PAD
        
        if self.n_range_search != 0:
            x_C10 = torch.zeros_like(x[:,:1], dtype=x.dtype, device=x.device)
            x_C10 = F.pad(x_C10, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)
            
            if self.pad_class_initial == 0:
                # add one more color dimension meaning padding or not
                x_C0 = F.pad(x[:, self.pad_class_initial:self.pad_class_initial+1], (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=1)
                x_C1to9 = F.pad(x[:, self.pad_class_initial+1:], (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=0)
                x = torch.cat([x_C0, x_C1to9, x_C10], dim=1)
            elif self.pad_class_initial == -1:
                x = F.pad(x, (W_PAD, W_PAD, H_PAD, H_PAD), mode='constant', value=0)
                x = torch.cat([x, x_C10], dim=1)

            ### TODO: if self.pad_class_initial != 0:

        x = x.permute(0, 2, 3, 1)
        C = x.shape[3]
        x = torch.stack([x[n].unfold(0, HL, 1).unfold(1, WL, 1) for n in range(N)], dim=0) # [N, H, W, C, HL, WL]
        x = x.view(N*H*W*C, HL, WL) # [N, S, C, L] where S = H*W

        # Padding to Max Length
        x_max = torch.full([N*H*W*C, self.H_kernel_max, self.W_kernel_max], fill_value=0, dtype=x.dtype, device=x.device)
        x_max[:,:HL,:WL] = x
        x = x_max.view(N*H*W, C, -1)

        # # Predict Padding Colors
        # if self.n_range_search != 0:
        #     x = x.transpose(1, 2)
        #     mask_to_inf = (x == 0).all(dim=2)
        #     mask_to_0 = (x[:, :, -1:] == 1)
        #     colored_padding = self.attn_C(x, src_key_padding_mask=mask_to_inf) # [L, C+1] <- [L, C+1]
        #     colored_padding = colored_padding[:, :, :-1].softmax(dim=2)
        #     colored_padding = colored_padding * mask_to_0
        #     x = x[:, :, :-1] + colored_padding
        #     x = x.transpose(1, 2)
        #     C = x.shape[1]

        if self.n_range_search != 0:
            x = x[:, :-1] # [N*H*W, C, L]
            C = x.shape[1]

        ### TODO: PixelVector = Pixels Located Relatively + Pixels Located Absolutely (363442ee, 63613498, aabf363d) + 3 Pixels with point-symmetric and line-symmetric relationships (3631a71a, 68b16354)
        ### TODO: PixelEachSubstitutorOverTime (045e512c, 22168020, 22eb0ac0, 3bd67248, 508bd3b6, 623ea044), 

        return x.view(N*H*W, C, self.H_kernel_max*self.W_kernel_max)


class PixelAbsoluteVectorExtractor(nn.Module):
    def __init__(self, W_max=30, H_max=30):
        super().__init__()
        self.W_max = W_max
        self.H_max = H_max

    def forward(self, x, output_shape):
        N, C, H, W = x.shape
        H_OUT, W_OUT = output_shape
        H_MAX, W_MAX = self.H_max, self.W_max

        x_max = torch.full([N, C, H_MAX, W_MAX], fill_value=0, dtype=x.dtype, device=x.device)
        x_max[:,:,:H,:W] = x
        x_broadcasted = x_max.repeat(H_OUT, W_OUT, 1, 1, 1, 1)
        
        return x_broadcasted.permute(2, 0, 1, 3, 4, 5).reshape(N, H_OUT, W_OUT, C, H_MAX, W_MAX).reshape(N*H_OUT*W_OUT, C, H_MAX*W_MAX)


class PixelVectorExtractor(nn.Module):
    def __init__(self, n_range_search, W_kernel_max, H_kernel_max, vec_abs=True, W_max=30, H_max=30, pad_class_initial=0, pad_n_head=None, pad_dim_feedforward=1, dropout=0.1, pad_num_layers=1, bias=False, n_class=10):
        super().__init__()
        self.vec_abs = vec_abs
        self.extract_rel_vec = PixelRelativeVectorExtractor(n_range_search, W_kernel_max, H_kernel_max, pad_class_initial, pad_n_head=pad_n_head, pad_dim_feedforward=pad_dim_feedforward, dropout=dropout, pad_num_layers=pad_num_layers, bias=bias, n_class=n_class)

        if vec_abs:
            self.extract_abs_vec = PixelAbsoluteVectorExtractor(W_max=W_max, H_max=H_max)

    def forward(self, x, output_shape=None):
        if output_shape is None:
            output_shape = x.shape[2:]

        x_vec = self.extract_rel_vec(x) # (29 + 1 + 29)**2 = 2704

        if self.vec_abs:
            x_abs = self.extract_abs_vec(x, output_shape) # 30**2 = 900
            x_vec = torch.cat([x_vec, x_abs], dim=2)

        return x_vec
