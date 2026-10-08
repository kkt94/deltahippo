import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from utils.backbone import get_backbone, obtain_features

def get_classifier(params, feature_dim, out_dim_list):
    if params.classifier == 'None':
        return None
    elif params.classifier == 'CosineLinear':
        return nn.ModuleList([CosineLinear(in_features=feature_dim,out_features=out_dim_list[t_id]) for t_id in range(len(out_dim_list))])
    elif params.classifier == 'Linear':
        return nn.ModuleList([nn.Linear(in_features=feature_dim,out_features=out_dim_list[t_id]) for t_id in range(len(out_dim_list))])
    elif params.classifier == 'MaskedCosineLinear':          
        # CosineLinear layers; the mask wrapper is applied in build_classifier
        return nn.ModuleList([CosineLinear(feature_dim, out_dim_list[t]) for t in range(len(out_dim_list))])
    else:
        raise NotImplementedError()
    
def get_real_weight(layer):
    """
    Return the weight tensor regardless of whether the layer is a WeightMaskWrapper
    """
    if isinstance(layer, WeightMaskWrapper):
        return layer.base_layer.weight
    return layer.weight

# Per-weight importance of a classifier layer (Fisher or MAS), averaged over the loader
def compute_importance(layer, loader, *, model, tokenizer, params,
                       metric="fisher", device=None):

    device  = device or next(model.parameters()).device
    weight  = get_real_weight(layer)
    imp_acc = torch.zeros_like(weight)

    layer = layer.to(device)
    model.eval(); layer.eval()

    for lm_input in loader:
        # ---------- 1) Feature extraction (backbone): no grad needed ----------
        with torch.no_grad():
            feats = obtain_features(params=params,
                                    model=model,
                                    lm_input=lm_input,
                                    tokenizer=tokenizer).to(device)

        # ---------- 2) Classifier forward: grad required ----------
        if metric == "fisher":
            logits = layer(feats)
            prob   = torch.softmax(logits, dim=-1)

            for i in range(prob.size(0)):
                layer.zero_grad(set_to_none=True)
                prob[i].log().sum().backward(retain_graph=True)
                imp_acc += weight.grad.abs()**2      # (∂log p)^2

        elif metric == "mas":
            layer.zero_grad(set_to_none=True)
            out = layer(feats)
            (out**2).sum().backward()
            imp_acc += weight.grad.abs()            # |∂ ‖F‖²|

        else:
            raise ValueError("metric must be 'fisher' or 'mas'")

    imp_acc /= len(loader.dataset)
    return imp_acc.detach()
    
class MultiProtoCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, num_proto=5):
        super(MultiProtoCosineLinear, self).__init__()
        self.num_proto = num_proto
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features*num_proto, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))
        return out
        

class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2,dim=1), F.normalize(self.weight, p=2, dim=1))

        return out

# ===== Mask wrapper around an existing Linear/CosineLinear layer =====
class WeightMaskWrapper(nn.Module):
    """
    Wraps any Linear/CosineLinear layer so that
    - only weights whose mask value is 1 are trained
    - weights whose mask value is 0 have their gradients zeroed
    """
    def __init__(self, base_layer: nn.Module, init_mask: torch.Tensor):
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("mask", init_mask.clone().float())

        # Mask gradients during backward
        self.base_layer.weight.register_hook(lambda g: g * self.mask)

    # Called when the mask needs to be replaced
    @torch.no_grad()
    def update_mask(self, new_mask: torch.Tensor):
        assert new_mask.shape == self.mask.shape
        self.mask.copy_(new_mask.float())

    def forward(self, x):
        w = self.base_layer.weight                  # Parameters actually being trained
        w_masked = w * self.mask                   # Mask values with 0/1 only

        # ── For CosineLinear, rewrite the same formula ──
        if isinstance(self.base_layer, CosineLinear):
            out = F.linear(F.normalize(x, dim=1),
                           F.normalize(w_masked, dim=1))
        else:   # Linear, etc.
            out = F.linear(x, w_masked, self.base_layer.bias)

        # Gradient masking is already handled via register_hook
        return out