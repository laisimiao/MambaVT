import os
import sys

prj_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if prj_path not in sys.path:
    sys.path.append(prj_path)

import argparse
import time
import torch
import torch.nn as nn
from thop import profile
from thop.utils import clever_format

from lib.models.mambatrack.mamba_simple import Mamba
from lib.models.mambatrack.vit import Attention

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='mambatrack', help='training script name')
    parser.add_argument('--config', type=str, default='mambavt_m256_ep20', help='yaml configure file name')
    args = parser.parse_args()

    return args

def get_complexity_MAMBA(m: Mamba, x, y):
    """(B, L, D): batch size, sequence length, dimension"""
    batch, seqlen, d_model = x[0].shape
    d_inner = m.d_inner
    d_state = m.d_state
    # import ipdb; ipdb.set_trace()
    """compute flops"""
    total_ops = 0
    # in_proj
    total_ops += 2 * d_inner * d_model * batch * seqlen
    # out_proj
    total_ops += d_inner * d_model * batch * seqlen
    # compute bi-SSM
    # see: https://github.com/state-spaces/mamba/issues/110
    total_ops += 2 * (9 * batch * seqlen * d_model * d_state + batch * d_model * seqlen)
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def get_complexity_SelfAttn(m: Attention, x, y):
    """(B, L, D): batch size, sequence length, dimension"""
    total_ops = 0
    B, N, C = x[0].shape
    total_ops += B * 4 * N * C ** 2
    total_ops += B * 2 * C * N ** 2
    m.total_ops += torch.DoubleTensor([int(total_ops)])

def get_data(bs, sz):
    img_patch = torch.randn(bs, 6, sz, sz)
    return img_patch

def evaluate_mambat(model, template_list, search_list, custom_ops, verbose=False):
    '''Speed Test'''
    macs1, params1 = profile(model, inputs=(template_list, search_list),
                             custom_ops=custom_ops, verbose=verbose)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    T_w = 50
    T_t = 100
    print("testing speed ...")
    torch.cuda.synchronize()
    with torch.no_grad():
        # overall
        for i in range(T_w):
            _ = model(template_list, search_list)
        start = time.time()
        for i in range(T_t):
            _ = model(template_list, search_list)
        torch.cuda.synchronize()
        end = time.time()
        avg_lat = (end - start) / T_t
        print("The average overall latency is %.2f ms" % (avg_lat * 1000))
        print("FPS is %.2f fps" % (1. / avg_lat))

if __name__ == "__main__":
    import importlib
    device = "cuda:1"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = f"{prj_path}/experiments/{args.script}/{args.config}.yaml"
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''import mambatrack network module'''
    model_module = importlib.import_module('lib.models.mambatrack')
    model_constructor = model_module.build_mambatrack
    model = model_constructor(cfg, training=False)
    # for name, module in model.named_modules():
    #     print(name)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    # z_num = 10  # cfg.TEST.TEMPLATE_NUMBER + 1

    model = model.to(device)
    template = get_data(bs, z_sz).to(device)
    search = get_data(bs, x_sz).to(device)

    for z_num in [7]:
        template_list = [template] * z_num
        search_list = [search]
        custom_ops = {Attention: get_complexity_SelfAttn} if 'vit' in args.config else {Mamba: get_complexity_MAMBA}
        verbose = True
        token_length = int(z_sz//16)**2 * z_num * 2 + int(x_sz//16)**2 * 2
        print(f"bs:{bs}, z_sz:{z_sz}, x_sz:{x_sz}, z_num:{z_num}, token_length:{token_length}")

        evaluate_mambat(model, template_list, search_list, custom_ops, verbose=verbose)

