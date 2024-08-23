import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelVectorExtractor(nn.Module):
    def __init__(self, n_range_search, max_width, max_height, pad_class_initial=0, pad_n_head=None, pad_dim_feedforward=1, dropout=0.1, pad_num_layers=1, bias=False, n_class=10):
        super().__init__()
        self.n_range_search = n_range_search
        self.max_width = max_width
        self.max_height = max_height
        self.pad_class_initial = pad_class_initial

        self.attn_C = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(n_class+1, pad_n_head if pad_n_head is not None else n_class+1, pad_dim_feedforward, dropout=dropout, batch_first=True, bias=bias),
            num_layers=pad_num_layers,
            enable_nested_tensor=False,
        )

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
        x_max = torch.full([N*H*W*C, self.max_height, self.max_width], fill_value=0, dtype=x.dtype, device=x.device)
        x_max[:,:HL,:WL] = x
        x = x_max.view(N*H*W, C, -1)

        # Predict Padding Colors
        if self.n_range_search != 0:
            x = x.transpose(1, 2)
            mask_to_inf = (x == 0).all(dim=2)
            mask_to_0 = (x[:, :, -1:] == 1)
            colored_padding = self.attn_C(x, src_key_padding_mask=mask_to_inf) # [L, C+1] <- [L, C+1]
            colored_padding = colored_padding[:, :, :-1].softmax(dim=2)
            colored_padding = colored_padding * mask_to_0
            x = x[:, :, :-1] + colored_padding
            x = x.transpose(1, 2)
            C = x.shape[1]

        # if self.n_range_search != 0:
        #     x = x[:, :-1] # [N*H*W, C, L]
        #     C = x.shape[1]

        ### TODO: PixelVector = Pixels Located Relatively + Pixels Located Absolutely (363442ee, 63613498, aabf363d) + 3 Pixels with point-symmetric and line-symmetric relationships (3631a71a, 68b16354)
        ### TODO: PixelEachSubstitutorOverTime (045e512c, 22168020, 22eb0ac0, 3bd67248, 508bd3b6, 623ea044), 

        return x.view(N*H*W, C, self.max_height, self.max_width)
