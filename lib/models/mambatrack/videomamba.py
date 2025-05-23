# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from lib.models.mambatrack.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, stride=16, in_chans=3, embed_dim=768, norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        drop_path=0.,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        layer_idx=None,
        bimamba=True,
        device=None,
        dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class VisionMamba(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            stride=16,
            depth=24,
            embed_dim=192,
            channels=3,
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            device=None,
            dtype=None,
            add_z_seg=False,
            z_vocab_size=200,
            add_cls_token=False
    ):
        factory_kwargs = {"device": device, "dtype": dtype}  # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, stride=stride, in_chans=channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        # ---------- new ------------ #
        self.add_z_seg = add_z_seg
        self.add_cls_token = add_cls_token
        self.template_segment_embed = nn.Embedding(z_vocab_size, self.embed_dim) if self.add_z_seg else None
        self.iou_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) if self.add_cls_token else None

        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        # self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def resize_pos_embed(self, new_size, pos_embed_checkpoint):
        model_pos_embed_len = int(new_size ** 2)

        if model_pos_embed_len != pos_embed_checkpoint.shape[-2] - 1:
            cls_embed_checkpoint = pos_embed_checkpoint[:, :1]
            patch_pos_embed = pos_embed_checkpoint[:, 1:]
            embedding_size = pos_embed_checkpoint.shape[-1]

            orig_size = int((pos_embed_checkpoint.shape[-2] - 1) ** 0.5)
            # new_size = int((model_pos_embed_len - 1) ** 0.5)

            patch_pos_embed = patch_pos_embed.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            patch_pos_embed = torch.nn.functional.interpolate(
                patch_pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False)
            new_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            return new_pos_embed
        else:
            return pos_embed_checkpoint

    def interpolate_pos_embed(self, cfg):
        search_size = cfg.DATA.SEARCH.SIZE
        template_size = cfg.DATA.TEMPLATE.SIZE
        patch_size = self.patch_embed.patch_size[0]
        template_feat_size = template_size // patch_size
        search_feat_size = search_size // patch_size

        # self.pos_embed
        pos_embed_t = self.resize_pos_embed(template_feat_size, self.pos_embed)
        pos_embed_s = self.resize_pos_embed(search_feat_size, self.pos_embed)
        self.pos_embed_t = nn.Parameter(pos_embed_t)
        self.pos_embed_s = nn.Parameter(pos_embed_s)

    def forward_features(self, z, x, z_frame_ids, concat_mode='tsts', scan_mode='spatial_first', inference_params=None):
        # z is a list containing multiple templates
        z = torch.stack(z, dim=1)
        _, T_z, C_z, H_z, W_z = z.shape
        z = z.flatten(0, 1)
        t_rgb = z[:, :3, :, :]
        t_X = z[:, 3:, :, :]

        B, H, W = x.shape[0], x.shape[2], x.shape[3]  # b,6,h,w
        s_rgb = x[:, :3, :, :]
        s_X = x[:, 3:, :, :]

        t_rgb = self.patch_embed(t_rgb)
        t_X = self.patch_embed(t_X)
        s_rgb = self.patch_embed(s_rgb)
        s_X = self.patch_embed(s_X)

        t_rgb = t_rgb + self.pos_embed_t
        s_rgb = s_rgb + self.pos_embed_s
        t_X = t_X + self.pos_embed_t
        s_X = s_X + self.pos_embed_s

        if T_z > 1:  # multiple memory frames
            t_rgb = t_rgb.view(B, T_z, -1, t_rgb.size()[-1]).contiguous()
            t_X = t_X.view(B, T_z, -1, t_X.size()[-1]).contiguous()
        else:
            t_rgb = t_rgb.view(B, 1, -1, t_rgb.size()[-1]).contiguous()
            t_X = t_X.view(B, 1, -1, t_X.size()[-1]).contiguous()

        if self.add_z_seg and z_frame_ids is not None:
            t_ids = z_frame_ids.to(t_rgb.device)
            segment_embed = self.template_segment_embed(t_ids).unsqueeze(-2)
            t_rgb = t_rgb + segment_embed
            t_X = t_X + segment_embed


        # concat all tokens
        if concat_mode == "tsts":
            if scan_mode == 'temporal_first':
                t_rgb = t_rgb.transpose(1, 2)
                t_X = t_X.transpose(1, 2)

            t_rgb = t_rgb.flatten(1, 2)
            t_X = t_X.flatten(1, 2)
            x = torch.cat([t_rgb, s_rgb, t_X, s_X], dim=1)
        elif concat_mode == "ttss":
            t_rgb = t_rgb.flatten(1, 2)
            t_X = t_X.flatten(1, 2)
            x = torch.cat([t_rgb, t_X, s_rgb, s_X], dim=1)
        elif concat_mode == "crosst":
            crosst_list = []
            for i in range(t_rgb.shape[1]):
                crosst_list.append(t_rgb[:, i])
                crosst_list.append(t_X[:, i])
            crosst_list.extend([s_rgb, s_X])
            x = torch.cat(crosst_list, dim=1)
        else:
            raise NotImplementedError(f'no implement such {concat_mode} mode!')

        if self.add_cls_token:
            iou_token = self.iou_token.expand(B, -1, -1)
            iou_token = iou_token + self.pos_embed[:, :1, :]
            x = torch.cat([iou_token, x], dim=1)

        x = self.pos_drop(x)

        # mamba impl
        residual = None
        hidden_states = x
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token
        return hidden_states

    def forward(self, z, x, z_frame_ids, concat_mode, scan_mode, inference_params=None):
        x = self.forward_features(z, x, z_frame_ids, concat_mode, scan_mode, inference_params=inference_params)
        # x = self.head(x)
        return x


def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(pretrained)['model']
        del ckpt["head.weight"]
        del ckpt["head.bias"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('Load pretrained videomamba missing_keys: ', missing_keys)
        print('Load pretrained videomamba unexpected_keys: ', unexpected_keys)
    return model


def videomamba_small(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=384,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(pretrained)['model']
        del ckpt["head.weight"]
        del ckpt["head.bias"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('Load pretrained videomamba missing_keys: ', missing_keys)
        print('Load pretrained videomamba unexpected_keys: ', unexpected_keys)
    return model


def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=576,
        depth=32,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(pretrained)['model']
        del ckpt["head.weight"]
        del ckpt["head.bias"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('Load pretrained videomamba missing_keys: ', missing_keys)
        print('Load pretrained videomamba unexpected_keys: ', unexpected_keys)
    return model


def videomamba_base(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16,
        embed_dim=768,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        ckpt = torch.load(pretrained)['model']
        del ckpt["head.weight"]
        del ckpt["head.bias"]
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        print('Load pretrained videomamba missing_keys: ', missing_keys)
        print('Load pretrained videomamba unexpected_keys: ', unexpected_keys)
    return model


if __name__ == '__main__':
    img_size = 576
    x = torch.rand(1, 3, img_size, img_size).cuda()
    model = videomamba_middle(img_size=img_size)
    ckpt = torch.load('videomamba_m16_in1k_res224to448to576.pth')['model']
    del ckpt["head.weight"]
    del ckpt["head.bias"]
    model.load_state_dict(ckpt, strict=True)
    model = model.cuda()
    y = model(x)
    print(y.shape)
