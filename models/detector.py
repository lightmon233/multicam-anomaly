import torch
import torch.nn as nn


def compute_loss(fused, recon, mem_attn):
    # 重构损失
    recon_loss = nn.functional.mse_loss(recon, fused.detach())
    # 记忆稀疏性损失（鼓励每次只激活少数记忆槽）
    sparsity_loss = (-mem_attn * torch.log(mem_attn + 1e-8)).mean()
    return recon_loss + 0.01 * sparsity_loss