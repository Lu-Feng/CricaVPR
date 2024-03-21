dependencies = ['torch']

import torch
import network


def trained_model():
    model = network.CricaVPRNet()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/CricaVPR/releases/download/v1.0/CricaVPR.pth', map_location=torch.device('cpu'))["model_state_dict"]
    )
    return model
