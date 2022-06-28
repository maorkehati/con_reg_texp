import random
import torch

def parse_config_line(l):
    l = l.strip()
    if not l or l.startswith("#"):
        return False
    
    if '#' in l:
        l = l[:l.index("#")]
        
    l = [i.strip() for i in l.split("=")]
    if len(l) != 2:
        return False
        
    return l

def isnumeric(s):
    return s.replace('.','',1).isdigit()

def rand_interval(mini, maxi):
    return mini+random.random()*(maxi-mini)
    
def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(0)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu