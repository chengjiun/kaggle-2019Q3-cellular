import numpy as np 
import pandas as pd
import torch


@torch.no_grad()
def prediction(model, loader, device='cuda'):
    preds = np.empty(0)
    for x, _ in loader: 
        x = x.to(device)
        output = model(x)
        idx = output.max(dim=-1)[1].cpu().numpy()
        preds = np.append(preds, idx, axis=0)
    return preds

