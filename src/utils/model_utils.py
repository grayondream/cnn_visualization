import torch

def load_state_dict(model, pt):
    if pt != None:
        pretrained_dict = torch.load(pt)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
