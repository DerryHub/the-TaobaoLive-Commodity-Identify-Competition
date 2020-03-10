import torch
from arcface.head import Arcface
from config import get_args_arcface
from arcface.utils import l2_norm

opt = get_args_arcface()

model = Arcface(opt)

model.load_state_dict(torch.load('trained_models/arcface_resnet_ir_se_50.pth.b'))

print(torch.sum(torch.sum(l2_norm(model.kernel.detach(), axis=0), dim=1)**2))
