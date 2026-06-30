import torch
import torch.nn as nn
from models import DecoderTransformer

def count_non_embedding_params(model):
    return sum(p.numel() for name, p in model.named_parameters()
               if 'embed' not in name and 'lm_head' not in name)

@torch.no_grad()
def estimate_loss(model, split, eval_iters, batch_size, block_size, custom_data=None):
    model.eval()
    losses = torch.zeros(eval_iters)
    source_data = train if split == 'train' else validation
    if custom_data is not None and split == 'train':
        source_data = custom_data
        
    for k in range(eval_iters):
        ix = torch.randint(len(source_data) - block_size, (batch_size,))
        x = torch.stack([source_data[i:i+block_size] for i in ix]).cuda()
        y = torch.stack([source_data[i+1:i+block_size+1] for i in ix]).cuda()
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()
