import torch
from models.pfld_vovnet import vovnet_pfld
from models.pfld import PFLDInference

from pthflops import count_ops

device = 'cuda:0'
model = PFLDInference().to(device)
inp = torch.rand(1,3,112,112).to(device)
count_ops(model,inp)