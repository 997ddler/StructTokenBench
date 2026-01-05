import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
from sklearn.cluster import KMeans
import math

from einops import rearrange, einsum

def get_codebook_utility(input_ids, codebook_embed, eps=1e-8):
    index_count = torch.bincount(input_ids, minlength=len(codebook_embed))
    # normalize frequency to probs
    probs = index_count / torch.sum(index_count)

    # perplexity
    perplexity = torch.exp(-torch.sum(probs * torch.log(probs + eps), dim=-1))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)

    # the percentage of used indices
    num_total = len(index_count)
    use_ratio = torch.count_nonzero(index_count) / num_total

    return {
        "perplexity": perplexity,
        "perplexity_normalized": perplexity / len(codebook_embed),
        "entropy": entropy,
        "entropy_normalized": entropy / len(codebook_embed),
        "use_ratio": use_ratio,
    }


class BaseQuantizer(nn.Module):

    def __init__(self, codebook_size: int=None, codebook_embed_size: int=None, 
        loss_weight: dict=None, _need_init: bool=True, 
        freeze_codebook: bool=False, use_linear_project: bool=False, 
        use_kmeans_init: bool=False, **kwargs):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_embed_size = codebook_embed_size
        self.codebook = nn.Embedding(self.codebook_size, self.codebook_embed_size)

        self.loss_weight = loss_weight

        self._need_init = _need_init
        self.freeze_codebook = freeze_codebook
        self.use_kmeans_init = use_kmeans_init

        self.use_linear_project = use_linear_project
        if self.use_linear_project:
            self.linear_proj = nn.Linear(self.codebook_embed_size, self.codebook_embed_size)
    
    @torch.no_grad()
    def get_codebook(self,):
        return self.codebook.weight

    def indices2embedding(self, indices: torch.IntTensor) -> torch.Tensor:
        z_q = self.codebook[indices]
        return z_q
    
    def forward(self, z: torch.Tensor, cur_iter: int | None) -> (torch.Tensor, torch.IntTensor, float):
        """Return: quantized_z, detached codes, commitment_loss
        """
        raise NotImplementedError
    
    def embedding2indices(self, z: torch.Tensor) -> torch.IntTensor:
        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        
        # calculate the distance for each representation w.r.t. the codebook
        if self.use_linear_project:
            weight = self.linear_proj(self.codebook.weight)
        else:
            weight = self.codebook.weight
        dist = (torch.sum(flat_z ** 2, dim=1, keepdim=True) 
                + torch.sum(weight ** 2, dim=1) # NOTE
                - 2 * torch.matmul(flat_z, weight.t())) # [B * L, codebook_size]
        
        # get indices of the closest embedding in the codebook
        quantized_indices = torch.argmin(dist, dim=1)
        quantized_indices = rearrange(quantized_indices, "(b l) -> b l", b=batch_size, l=seq_length).detach()

        return quantized_indices

class StraightThroughQuantizer(BaseQuantizer):

    """
    Reference: https://github.com/SerezD/vqvae-vqgan-pytorch-lightning/blob/7a08d332f9fe9f275cdbfa82dc739fdcebad3398/vqvae/modules/vector_quantizers.py#L8
    """

    def __init__(self, soft_ae_training=False, soft_ae_iters=10000, 
                 soft_ae_scheduler='cosine', lambda_loss=0.5, 
                 soft_loss=False, use_soft_representation=False, **kwargs):
        super().__init__(**kwargs)
        
        self.soft_ae_training = soft_ae_training
        self.soft_ae_iters = soft_ae_iters
        self.soft_ae_scheduler = soft_ae_scheduler
        self.lambda_loss = lambda_loss
        self.soft_loss = soft_loss
        self.use_soft_representation = use_soft_representation
        
        if self.soft_ae_training:
            
            self.register_buffer('sche_a', torch.tensor(1.0))
            self.register_buffer('cur_iter', torch.tensor(0))

    
    def _tile(self, x):
        """
        Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/layers/codebook.py#L34
        """
        d, ew = x.shape
        if d < self.codebook_size:
            n_repeats = (self.codebook_size + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """
        Reference: https://github.com/evolutionaryscale/esm/blob/2efdadfe77ddbb7f36459e44d158531b4407441f/esm/layers/codebook.py#L43
        """
        # z: [B, L, hidden_dim]
        self._need_init = False

        flat_inputs = z.view(-1, self.codebook_embed_size) # [B * L, hidden_dim]
        
        if self.use_kmeans_init:
            # K-means initialization: gather embeddings from all GPUs, cluster, and distribute
            if dist.is_initialized():
                # Gather embeddings from all GPUs to rank 0
                world_size = dist.get_world_size()
                rank = dist.get_rank()
                
                # Prepare list for gathering (all_gather requires same shape, so we assume same batch size)
                gathered_inputs = [torch.zeros_like(flat_inputs) for _ in range(world_size)]
                dist.all_gather(gathered_inputs, flat_inputs)
                
                # Only rank 0 performs k-means clustering
                if rank == 0:
                    # Concatenate all gathered embeddings: [world_size * B * L, hidden_dim]
                    print(f"Gathering embeddings from all GPUs to rank 0, world_size: {world_size}, rank: {rank}")
                    all_embeddings = torch.cat(gathered_inputs, dim=0)
                    
                    # Convert to numpy for sklearn KMeans
                    all_embeddings_np = all_embeddings.cpu().numpy()
                    
                    # Perform k-means clustering
                    kmeans = KMeans(n_clusters=self.codebook_size, random_state=0, n_init=10)
                    kmeans.fit(all_embeddings_np)
                    centroids = torch.from_numpy(kmeans.cluster_centers_).to(z.device).to(z.dtype)
                else:
                    # Other ranks create placeholder tensor for receiving centroids
                    centroids = torch.zeros(self.codebook_size, self.codebook_embed_size, 
                                          device=z.device, dtype=z.dtype)
                
                # Broadcast centroids from rank 0 to all GPUs
                dist.broadcast(centroids, 0)
            else:
                # Single GPU case
                all_embeddings = flat_inputs
                all_embeddings_np = all_embeddings.cpu().numpy()
                
                # Perform k-means clustering
                kmeans = KMeans(n_clusters=self.codebook_size, random_state=0, n_init=10)
                kmeans.fit(all_embeddings_np)
                centroids = torch.from_numpy(kmeans.cluster_centers_).to(z.device).to(z.dtype)
            
            self.codebook.weight.detach().copy_(centroids)
        else:
            # Original random initialization
            y = self._tile(flat_inputs)
            _k_rand = y[torch.randperm(y.shape[0])][: self.codebook_size]

            if dist.is_initialized():
                dist.broadcast(_k_rand, 0)
            self.codebook.weight.detach().copy_(_k_rand)
        
        if self.freeze_codebook:
            for name, p in self.codebook.named_parameters():
                p.requires_grad = False
    
    def update_sche_a(self, cur_iter):
        """Update the schedule parameter for soft quantization warmup."""
        if not self.soft_ae_training:
            return
        
        if cur_iter >= self.soft_ae_iters:
            self.sche_a.fill_(0.0)
            return
        
        progress = cur_iter / self.soft_ae_iters
        
        if self.soft_ae_scheduler == 'cosine':
            self.sche_a.fill_(0.5 * (1 + math.cos(math.pi * progress)))
        elif self.soft_ae_scheduler == 'linear':
            self.sche_a.fill_(1 - progress)
        elif self.soft_ae_scheduler == 'exponential':
            self.sche_a.fill_(math.exp(-5 * progress))
        else:
            raise ValueError(f"Invalid scheduler: {self.soft_ae_scheduler}")
    
    def forward(self, z: torch.Tensor, cur_iter=None):
        # z: [B, L, hidden_dim]
        
        if self._need_init and self.training:# and not self.freeze_codebook:
            self._init_embeddings(z)
        
        # Update schedule if soft_ae_training is enabled and cur_iter is provided
        if self.training and self.soft_ae_training and cur_iter is not None:
            self.update_sche_a(cur_iter)
        
        # get indices of the closest embedding in the codebook
        quantized_indices = self.embedding2indices(z)

        batch_size, seq_length, dim_size = z.shape
        flat_z = rearrange(z, "b l h -> (b l) h")
        flat_indices = rearrange(quantized_indices, "b l -> (b l)")
        
        quantized_z_pos = torch.zeros((flat_indices.shape[0], self.codebook_size), device=z.device)
        quantized_z_pos = quantized_z_pos.scatter_(1, flat_indices.unsqueeze(1), 1) # [B * L, codebook_size]
        if self.use_linear_project:
            quantized_z = torch.matmul(quantized_z_pos, self.linear_proj(self.codebook.weight)) # [B * L, hidden_dim = 128]
        else:
            quantized_z = torch.matmul(quantized_z_pos, self.codebook.weight) # [B * L, hidden_dim = 128]

        # loss functions
        metrics = {}
        # Reference: Eqn. (3) in https://arxiv.org/pdf/1711.00937
        commitment_loss = F.mse_loss(quantized_z.detach(), flat_z)
        quantization_loss = F.mse_loss(quantized_z, flat_z.detach())
        
        # Apply soft loss weighting during warmup if enabled
        if self.soft_ae_training and self.soft_loss:
            sche_a_val = self.sche_a.item()
            loss_weight = (1 - self.lambda_loss) * (1 - sche_a_val) + self.lambda_loss
            commitment_loss = loss_weight * commitment_loss
            quantization_loss = loss_weight * quantization_loss
        
        loss = self.loss_weight["commitment_loss_weight"] * commitment_loss
        metrics["commitment_loss"] = commitment_loss
        loss += self.loss_weight["quantization_loss_weight"] * quantization_loss
        metrics["quantization_loss"] = quantization_loss

        metrics["sche_a"] = self.sche_a.detach()

        # Apply soft quantization during warmup
        if self.training and self.soft_ae_training:
            sche_a_val = self.sche_a.item()
            if self.use_soft_representation and sche_a_val > 0.0:
                # During warmup: use soft representation
                quantized_z = self.sche_a * flat_z + (1 - self.sche_a) * (flat_z + (quantized_z - flat_z).detach())
                quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)
            else:
                # After warmup or if not using soft representation: standard straight-through
                quantized_z = flat_z + (quantized_z - flat_z).detach()
                quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)
        else:
            # Standard straight through gradient
            quantized_z = flat_z + (quantized_z - flat_z).detach()
            quantized_z = rearrange(quantized_z, "(b l) h -> b l h", b=batch_size, l=seq_length, h=dim_size)

        return quantized_z, quantized_indices, loss, metrics