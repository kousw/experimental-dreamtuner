
def freeze_params(params):
    for param in params:
        param.requires_grad_(False)

def unfreeze_params(params):
    for param in params:
        param.requires_grad_(True)