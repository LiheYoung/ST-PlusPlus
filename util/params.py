def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6
