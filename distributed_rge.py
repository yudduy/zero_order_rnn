#!/usr/bin/env python3: 
# To run smoke test:
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py
# 7B param LSTM
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --tokenizer hf --model_type LSTM
# or
# python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 distributed_rge.py  



# To train 1B param LSTM on OWT
# python -m torch.distributed.launch --nproc_per_node=10 distributed_rge.py --tokenizer hf --model_type LSTM --mode train --batch_size 1024 --meta_perturbations 10 --model_scale 1 --hidden_size 297500 --input_size 1024 --num_heads 350 --min_seq_len 10 --learning_rate 0.001  --verbose false


# 
# To run distributed speed tests tests for LSTM and num perturbations 96 (which is 8x12 metaperturbations), on tiny (100k), small (1m), medium (10m), large (100m), xlarge models (1.1B) on a 8xA40 server, run:
# tiny: (~100k params, 0.05s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 12 --model_scale 1 --hidden_size 240 --input_size 100 --num_heads 12 --model_type LSTM --min_seq_len 10 --learning_rate 0.1  --verbose false
# small: (~1m params, 0.06s / step )
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 12 --model_scale 1 --hidden_size 1600 --input_size 100 --num_heads 32 --model_type LSTM --min_seq_len 10 --learning_rate 0.1  --verbose false

# medium: (~10m params, 0.07s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 12 --model_scale 1 --hidden_size 9600 --input_size 100 --num_heads 64 --model_type LSTM --min_seq_len 10 --learning_rate 0.01  --verbose false

# large: (~100m params, 0.4s / step )
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 12 --model_scale 1 --hidden_size 66000 --input_size 100 --num_heads 220 --model_type LSTM --min_seq_len 10 --learning_rate 0.01  --verbose false

# xlarge: (~1B params, 4.4s / steps )
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 12 --model_scale 1 --hidden_size 297500 --input_size 100 --num_heads 350 --model_type LSTM --min_seq_len 10 --learning_rate 0.005  --verbose false

# xxlarge: (~4.5B params, 17s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 12 --model_scale 1 --hidden_size 1000000 --input_size 100 --num_heads 1000 --model_type LSTM --min_seq_len 10 --learning_rate 0.0001  --verbose false



# To run distributed speed tests tests for LSTM and num perturbations 512 (which is 8x64 metaperturbations), on tiny (100k), small (1m), medium (10m), large (100m), xlarge models (1.1B) on a 8xA40 server, run:
# tiny: (~100k params, 0.24s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 64 --model_scale 1 --hidden_size 240 --input_size 100 --num_heads 12 --model_type LSTM --min_seq_len 10 --learning_rate 0.5  --verbose false
# small: (~1m params, 0.25s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 64 --model_scale 1 --hidden_size 1600 --input_size 100 --num_heads 32 --model_type LSTM --min_seq_len 10 --learning_rate 0.1  --verbose false

# medium: (~10m params, 0.3s / step )
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 64 --model_scale 1 --hidden_size 9600 --input_size 100 --num_heads 64 --model_type LSTM --min_seq_len 10 --learning_rate 0.1  --verbose false

# large: (~100m params, 0.3s / step )
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 64 --model_scale 1 --hidden_size 66000 --input_size 100 --num_heads 220 --model_type LSTM --min_seq_len 10 --learning_rate 0.1  --verbose false

# xlarge: (~1B params, 19s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 64 --model_scale 1 --hidden_size 297500 --input_size 100 --num_heads 350 --model_type LSTM --min_seq_len 10 --learning_rate 0.01  --verbose false

# xxlarge: (~4.5B params, 17s / step)
# python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py --batch_size 1024 --meta_perturbations 64 --model_scale 1 --hidden_size 1000000 --input_size 100 --num_heads 1000 --model_type LSTM --min_seq_len 10 --learning_rate 0.001  --verbose false



from __future__ import annotations
from contextlib import contextmanager
from typing import Dict, List, Sequence

import os, argparse, time
import string
import torch
import torch.nn as nn
import torch.distributed as dist
import datetime
import wandb
import math
from datasets import load_dataset
import random
import numpy as np
import warnings


from pathlib import Path
from simpletokenizers.simpletokenizers import CharTokenizer, NumericTokenizer, get_tiktoken
from models.models        import LSTM, DNC
from tasks.tasks          import generate_openwebtext_task, compute_task_loss, compute_task_accuracy


try:
    # Requires:  flashrnn ≥ 0.1  (pip install flash-rnn or pip install -e . from repo RECOMMENDED!) OR it will fallback to pytroch and be slow.
    from flashrnn.flashrnn import flashrnn           # official package name
    FLASH_OK = True
except ModuleNotFoundError:
    FLASH_OK = False
    warnings.warn("flashrnn not found – LSTM will fall back to PyTorch LSTM.",
                  RuntimeWarning)

CHUNK_SIZE = 2**22 


def teacher_forcing_loss_emb_parallel(model, x_ids, y_ids_unpadded, criterion, chunk_size=32, x_emb=None, return_predictions=False):

    try:
        
        if x_emb == None:
            x_emb = model.embed(x_ids)
            
        if x_emb.dtype != next(model.parameters()).dtype:
            x_emb = x_emb.to(dtype=next(model.parameters()).dtype)
            
        B, Lx, E = x_emb.shape
        Ly = y_ids_unpadded.shape[1]
    
        if return_predictions:
            all_preds = []
        hidden = None
        memory = None
        total_loss = 0.0
        total_predicted_tokens = 0
        
        # Process input sequence first
        pos = 0
        while pos < Lx:
            chunk_end = min(pos + chunk_size, Lx)
            input_chunk = x_emb[:, pos:chunk_end, :]
            
            out_chunk, mem_new, hidden_new = model(input_chunk, hidden=hidden, memory=memory)
            hidden = hidden_new
            memory = mem_new
            pos = chunk_end

        
        # Now process target sequence chunk by chunk
        pos = 0
        while pos < Ly - 1:  # -1 because we don't embed the last target token
            chunk_end = min(pos + chunk_size, Ly - 1)
            # Only embed the current chunk of target sequence
            y_chunk = y_ids_unpadded[:, pos:chunk_end]
            y_emb_chunk = model.embed(y_chunk)
            
            out_chunk, mem_new, hidden_new = model(y_emb_chunk, hidden=hidden, memory=memory)
            
            # Update states
            hidden = hidden_new
            memory = mem_new
    
            # Compute loss for this chunk
            out_chunk = out_chunk.reshape(-1, out_chunk.size(-1))
            targets = y_ids_unpadded[:, pos+1:chunk_end+1].reshape(-1)  # shift by 1 for next-token prediction
            
            if targets.size(0) > 0:  # ensure we have targets
                chunk_loss = criterion(out_chunk.to(torch.float64), targets)
                out_chunk = out_chunk.to(torch.float32)  # Optional: demote if reused below
                chunk_loss = criterion(out_chunk, targets)
                total_loss += chunk_loss * targets.size(0)
                total_predicted_tokens += targets.size(0)
            
            pos = chunk_end
            if return_predictions:
                pred_chunk = torch.argmax(out_chunk, dim=-1).reshape(targets.shape).detach()
                all_preds.append(pred_chunk)

    

        if total_predicted_tokens == 0:
            import pdb
            pdb.set_trace()
            return 0.0 if not return_predictions else (0.0, None)
        
        avg_loss = total_loss / total_predicted_tokens
        avg_loss = avg_loss.to(torch.float32)  # demote back

        if return_predictions:
            preds = torch.cat(all_preds, dim=-1).reshape(y_ids_unpadded.size(0), -1)
            return avg_loss, preds
        else:
            return avg_loss

    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM in teacher_forcing_loss_emb: {str(e)}")
        torch.cuda.empty_cache()
        raise  # Re-raise to be caught by train_micro_batch

    except Exception as e:
        print(f"Error in teacher_forcing_loss_emb: {str(e)}")
        torch.cuda.empty_cache()
        raise  # Re-raise to be caught by train_micro_batch




# =============================================================================
# Logging Helpers
# =============================================================================
def current_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log_start(stage, rank):
    print(f"[Rank {rank}] {current_time()} - START {stage}\n", end='', flush=True)

def log_end(stage, rank):
    print(f"[Rank {rank}] {current_time()} - END {stage}\n", end='', flush=True)

def log_msg(stage, rank, msg):
    print(f"[Rank {rank}] {current_time()} - {stage}: {msg}\n", end='', flush=True)

# =============================================================================
# Persistent Group: Create once for all ranks except rank 0
# =============================================================================
def create_group_except_rank0():
    world_size = dist.get_world_size()
    ranks = list(range(1, world_size))
    return dist.new_group(ranks=ranks)

# =============================================================================
# Helper: Broadcast within a given group (persistent group used)
# =============================================================================
def broadcast_in_group(tensor, src_rank, group):
    dist.broadcast(tensor, src=src_rank, group=group)
    # dist.barrier()
    return tensor

### FOR CD-RGE DIST FUNCTION



    
    
# --------------------------------------------------------------------------- #
#  Timing helper                                                  #
# --------------------------------------------------------------------------- #
@contextmanager
def timed(label: str, rank: int, bucket: Dict[str, float], *, sync: bool = True, verbose=False):
    """
    Timing context with explicit START / END logs per rank per stage.
    """
    if verbose:
        print(f"START [rank {rank:02d}] {label}")
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    bucket[label] = bucket.get(label, 0.0) + dt
    if verbose:
        print(f" END  [rank {rank:02d}] {label:30s}: {dt:8.4f}s")




# --------------------------------------------------------------------------- #
#  Low-level helpers                  #
# --------------------------------------------------------------------------- #
# def _same_device(params: Sequence[torch.nn.Parameter]) -> torch.device:
#     dev = params[0].device
#     for p in params:
#         if p.device != dev:
#             raise RuntimeError("parameters live on multiple devices")
#     return dev


def generate_perturbation(ref: torch.Tensor, scale: float, distribution: str, seed: int) -> torch.Tensor:
    g = torch.Generator(device=ref.device).manual_seed(int(seed))
    if distribution == "rad":
        z = torch.zeros_like(ref).bernoulli_(0.5, generator=g).mul_(2).sub_(1).mul_(scale)
    elif distribution == "normal":
        z = torch.randn_like(ref, generator=g) * scale
    else:  # uniform
        z = (torch.rand_like(ref, generator=g) * 2 - 1) * scale
    return z


import torch
import torch.distributed as dist
from typing import Sequence, List, Dict, Optional, Tuple

# --------------------------------------------------------------------------- #
#  Utility: apply a single probe                                              #
# --------------------------------------------------------------------------- #
def apply_probe(params: Sequence[torch.nn.Parameter],
                scale: float,
                base_seed: int,
                distn: str,
                rolling_sum_weighted_probe: Optional[List[torch.Tensor]] = None,
                coeff: Optional[float] = None):
    """
    θ ← θ + scale * δ     where δ has unit magnitude (Rademacher or Normal).

    If `rolling_sum_weighted_probe` and `coeff` are provided we accumulate
        rolling_sum_weighted_probe[i] += coeff * δ
    so a one-shot gradient step can be taken later.

    Note: `torch.add_(tensor, alpha=scale)` is already a fused kernel; `addcmul_`
    would need two tensors and offers no savings here.
    """
    for i, p in enumerate(params):
        delta = generate_perturbation(p, 1.0, distn, base_seed + i)  # unit δ
        p.data.add_(delta, alpha=scale)                              # fused add

        if rolling_sum_weighted_probe is not None and coeff is not None:
            rolling_sum_weighted_probe[i].add_(delta, alpha=coeff)



# --------------------------------------------------------------------------- #
#  Helper for broadcasting variable-length 2-D Long tensors                   #
# --------------------------------------------------------------------------- #
def broadcast_tensor_list(src_list: List[torch.Tensor], per_rank: int, device: torch.device) -> List[torch.Tensor]:
    """
    Rank-0 holds `src_list` of length `per_rank`; each entry is a 2-D Long tensor.
    Other ranks get allocated tensors with identical shapes & receive the data.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    out: List[torch.Tensor] = []
    for k in range(per_rank):
        if rank == 0:
            t = src_list[k].to(device, non_blocking=True)
            shape = torch.tensor(t.shape, dtype=torch.int32, device=device)
        else:
            shape = torch.empty(2, dtype=torch.int32, device=device)
            t = None

        dist.broadcast(shape, src=0) if dist.is_initialized() else None
        rows, cols = int(shape[0]), int(shape[1])

        if rank != 0:
            t = torch.empty((rows, cols), dtype=torch.long, device=device)
        dist.broadcast(t, src=0) if dist.is_initialized() else None
        out.append(t)
    return out



# =============================================================================
# Main Class: RGEOptimizerDistributed
# =============================================================================
#
# Roles:
#   - Rank 0 ("adam rank"): does not create a model; obtains parameter meta from Rank 1,
#         then initializes adam_m and adam_v (on GPU) based on that meta.
#   - Rank 1 ("clean rank"): creates the full model and input data.
#   - Dirty ranks (>=2): create the model structure only (their parameters will be overwritten).

class RGEOptimizerDistributed:
    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 probe_dropout_rate=0.99, 
                 epsilon_tying_ratio=1.0,
                 beta1=0.9, 
                 beta2=0.999,
                 adam=True,
                 adaptive=False,
                 probe_normalization=False,
                 gradient_normalization=False,
                 meta_perturbations=1,
                 weight_decay=0.0,
                 scatter_batch=False,
                 normal_distribution =True,
                 l1_norm = False,
                 antithetic = True,
                 verbose=True):
        self.normal_distribution = normal_distribution
        self.l1_norm = l1_norm
        self.learning_rate = learning_rate
        self.probe_dropout_rate = probe_dropout_rate
        self.epsilon_tying_ratio = epsilon_tying_ratio 
        self.beta1 = beta1
        self.beta2 = beta2
        self.adam = adam
        self.antithetic = antithetic
        self.weight_decay = weight_decay
        
        self.adaptive = adaptive
        self.probe_normalization = probe_normalization
        self.gradient_normalization = gradient_normalization
        self.meta_perturbations = meta_perturbations
        self.scatter_batch = scatter_batch
        self.verbose = verbose

        # On Rank 1 and dirty ranks, model is provided.
        self.model = model  

        rank = dist.get_rank()
        
        self.param_list = list(self.model.parameters())
        self.d = sum(p.numel() for p in self.param_list)
        
        self.adam_m = None  # this will exist only on rank 0
        self.adam_v = None  # this will exist only on rank 0
        
        self.adam_ratio_list = None # this will exist only on rank 1
        self.probe_list = None  # this will exist only on rank 2+

        if self.adam:
            self.clean_rank = 1
        else:
            self.clean_rank = 0
        

        # Create a persistent group for all ranks except rank 0.
        if self.adam: 
            if rank == 0: # adam rank, hold both adam moments
                self.group_except_zero = None
                self.adam_m = [torch.zeros_like(p) for p in self.param_list]
                del self.param_list
                self.param_list = None
                self.adam_v = [torch.zeros_like(p) for p in self.adam_m]
            else:
                
                self.group_except_zero = create_group_except_rank0()
                if rank == 1: # clean rank, hold model + adam_ratio
                    self.adam_ratio_list = [torch.zeros_like(p, dtype=p.dtype, device=p.device) for p in self.param_list]
                elif rank>=2: # dirty ranks, hold model + probe
                    self.probe_list = [torch.zeros_like(p, dtype=p.dtype, device=p.device) for p in self.param_list]
        

    
    def dist_cdrge_step(
        self,
        x_ids_list: List[torch.Tensor],
        y_list:     List[torch.Tensor],
        criterion,
        iteration:  int,          # unused but kept for API parity
        cache_roll = True
    ) -> float:
        # 0.  set-up ----------------------------------------------------------------
        distributed  = dist.is_available() and dist.is_initialized()
        rank         = dist.get_rank()       if distributed else 0
        world_size   = dist.get_world_size() if distributed else 1
        per_rank     = self.meta_perturbations
        n_total      = per_rank * world_size
        macro_bs     = int(getattr(self, "macro_batch_size", 1))
        distn        = "normal" if getattr(self, "normal_distribution", False) else "rad"
        epsilon      = self.epsilon_tying_ratio * self.learning_rate
        param_list   = self.param_list
        device       = param_list[0].device
        verbose      = self.verbose
    
        times: Dict[str, float] = {}
    
        # 1. optional rolling buffer -----------------------------------------------
        rolling_sum_weighted_probe = (
            [torch.zeros_like(p.data) for p in param_list] if cache_roll else None
        )

        
        # 2. broadcast θ ------------------------------------------------------------
        with timed("broadcast_theta", rank, times, verbose=verbose):
            if distributed and world_size > 1:
                for p in param_list:
                    dist.broadcast(p.data, src=0)
    
        # 3. broadcast batch --------------------------------------------------------
        with timed("broadcast_batch", rank, times, verbose=verbose):
            x_ids_list = broadcast_tensor_list(x_ids_list, per_rank, device)
            y_list     = broadcast_tensor_list(y_list,     per_rank, device)
    
        # 4. scatter seeds ----------------------------------------------------------
        with timed("seed_scatter", rank, times, verbose=verbose):
            seeds_local = torch.zeros(per_rank, dtype=torch.int32, device=device)
            if rank == 0:
                full_seeds = torch.randint(0, 2**31 - 1, (n_total,),
                                           dtype=torch.int32, device=device)
                chunks = list(full_seeds.chunk(world_size, dim=0))
            else:
                full_seeds = torch.empty(n_total, dtype=torch.int32, device=device)
                chunks = None
            if distributed and world_size > 1:
                dist.scatter(seeds_local, chunks, src=0)
            else:
                seeds_local.copy_(full_seeds)
    
        # 5. local ±ε evaluations ---------------------------------------------------
        loss_pairs_local = torch.zeros(per_rank, 2, dtype=torch.float32, device=device)

        for m in range(per_rank):
            seed_m  = int(seeds_local[m].item())
            x_ids   = x_ids_list[m]
            y       = y_list[m]
    
            with timed("forward_loops", rank, times, verbose=verbose):
                # +ε
                apply_probe(param_list, +epsilon, seed_m, distn)
                L_plus = sum(
                    teacher_forcing_loss_emb_parallel(self.model, x_ids, y, criterion).item()
                    for _ in range(macro_bs)
                ) / macro_bs
    
                # −ε
                apply_probe(param_list, -2.0 * epsilon, seed_m, distn)
                L_minus = sum(
                    teacher_forcing_loss_emb_parallel(self.model, x_ids, y, criterion).item()
                    for _ in range(macro_bs)
                ) / macro_bs
    
                # coef and restore (+ε again)
                coef          = (L_plus - L_minus) / (2.0 * n_total)
                restore_coeff = -coef                           # GD direction
                apply_probe(
                    param_list, +epsilon, seed_m, distn,
                    rolling_sum_weighted_probe=rolling_sum_weighted_probe,
                    coeff=restore_coeff,
                )
    
                loss_pairs_local[m, 0] = L_plus
                loss_pairs_local[m, 1] = L_minus
    
        # 6. gather losses (logging only) ------------------------------------------
        with timed("gather_losses", rank, times, verbose=verbose):
            if distributed and world_size > 1:
                gather_buf = (
                    [torch.empty_like(loss_pairs_local) for _ in range(world_size)]
                    if rank == 0 else None
                )
                dist.gather(loss_pairs_local, gather_buf, dst=0)
                if rank == 0:
                    loss_pairs_full = torch.cat(gather_buf, dim=0)
            else:
                loss_pairs_full = loss_pairs_local

        
        # 7. parameter update -------------------------------------------------------
        with timed("param_update", rank, times, verbose=verbose):
            if cache_roll:
                if distributed and world_size > 1:
                    for buf in rolling_sum_weighted_probe:
                        dist.reduce(buf, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    for p, acc in zip(param_list, rolling_sum_weighted_probe):
                        p.data.add_(acc)                       # fused add
            else:  # fallback (slow loop)
                if rank == 0:
                    if world_size == 1:
                        full_seeds = seeds_local.clone()
                    for i in range(n_total):
                        coef   = (loss_pairs_full[i, 0] - loss_pairs_full[i, 1]) \
                                 / (2.0 * n_total)
                        seed_i = int(full_seeds[i].item())
                        apply_probe(param_list, -coef.item(), seed_i, distn)
    
        # 8. timing summary (optional) ---------------------------------------------
        if rank == 0 and verbose:
            key_order = ["broadcast_theta", "broadcast_batch", "seed_scatter",
                         "forward_loops", "gather_losses", "param_update"]
            t_vec = [times.get(k, 0.0) for k in key_order]
            print("\n--- timing summary (sec) ---")
            for k, v in zip(key_order, t_vec):
                print(f"{k:<25s}: {v:7.3f}")
            print("--------------------------------\n")
    
        # 9. return loss ------------------------------------------------------------
        mean_loss = float(
            (loss_pairs_full if rank == 0 else loss_pairs_local).mean().item()
        )
        dist.barrier()
        return mean_loss if rank == 0 else {}


    


# ------------------------------------------------------------
# save_distributed_checkpoint  (model only, no Adam tensors)
# ------------------------------------------------------------
def save_distributed_checkpoint(optimizer, run_name, save_dir, rank):
    """
    Rank-0: write <run_name>_model.pt  containing only the model state
    Rank-1+: do nothing
    """
    os.makedirs(save_dir, exist_ok=True)

    if rank != 0:          # only rank-0 does I/O
        return

    model = optimizer.model
    ckpt_path = os.path.join(save_dir, f"{run_name}_model.pt")

    # collect *existing* architecture fields so this works for DNC and LSTM
    wanted = ("input_size", "output_size", "hidden_size",
              "memory_size", "head_size", "num_reads")
    model_args = {k: getattr(model, k) for k in wanted if hasattr(model, k)}

    torch.save({
        "model_state_dict": model.state_dict(),
        "model_args":       model_args,
    }, ckpt_path)

    print(f"[rank{rank}] model checkpoint saved to {ckpt_path}")


# ------------------------------------------------------------
# load_distributed_checkpoint 
# ------------------------------------------------------------
def load_distributed_checkpoint(optimizer, ckpt_path, device, rank):
    """
    Load a checkpoint directly from a full path.
    Only rank 0 reads from disk and loads the model state.
    """
    if rank != 0:
        return True  # other ranks do nothing

    if ckpt_path is None:
        print(f"[rank{rank}] No checkpoint provided. Starting from scratch.")
        return False

    if not os.path.exists(ckpt_path):
        print(f"[rank{rank}] WARNING – checkpoint {ckpt_path} not found. Starting from scratch.")
        return False

    ckpt = torch.load(ckpt_path, map_location=device)
    optimizer.model.load_state_dict(ckpt["model_state_dict"])

    # Sanity check architecture
    loaded_args = ckpt.get("model_args", {})
    current_args = {k: getattr(optimizer.model, k)
                    for k in loaded_args.keys()}

    for k, v in loaded_args.items():
        if current_args.get(k) != v:
            print(f"[rank{rank}] WARNING – mismatch {k}: ckpt={v}, current={current_args.get(k)}")

    print(f"[rank{rank}] Loaded checkpoint from {ckpt_path}")
    return True


# =============================================================================
# Main Routine
# =============================================================================
def main():
    import os
    os.environ["WANDB_API_KEY"] = ""

    parser = argparse.ArgumentParser()
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training.")
    parser.add_argument("--mode", type=str, choices=["test", "train"], default="train", help="Run mode: test or train.")
    parser.add_argument("--max_iters", type=int, default=1e10, help="Maximum iterations for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--weight_decay", type=float, default=0.00001, help="Standard weight decay.")
    parser.add_argument("--beta1", type=float, default=0.99, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--beta2", type=float, default=0.999, help="Base learning rate (and eps, tied 1:1).")
    parser.add_argument("--epsilon_tying_ratio", type=float, default=1.0, help="Perturbation scale epsilon (tied to learning rate).")
    parser.add_argument("--probe_dropout_rate", type=float, default=0., help="Dropout rate for probe vector.")
    parser.add_argument("--wandb_proj", type=str, default=None, help="WandB project name (optional)")
    parser.add_argument("--wandb_run", type=str, default=None, help="WandB run name (optional)")
    parser.add_argument("--warmup_iters", type=int, default=100, help="Warmup iterations.")
    parser.add_argument("--cosine_wavelength", type=int, default=1000, help="Cosine LR wavelength, init to very high.")
    parser.add_argument("--val_iters", type=int, default=10, help="Val iters, when we run val and log to wandb, and potentially checkpoint.")
    parser.add_argument("--meta_perturbations", type=int, default=12, help="Number of Perturbations for all ranks per step.")
    parser.add_argument("--scatter_batch", type=str, choices=["true", "false"], default="false", help="whether each perturbation should be on a different batch, if true, we sample (world_size-2)*batch_size x_ids and y per iter and scatter it to the appropriate ranks.")
    
    # New CLI arguments for model configuration
    parser.add_argument("--model_scale", type=int, default=1, help="Scaling factor for model dimensions.")
    parser.add_argument("--num_heads", type=int, default=16, help="# dnc heads.")
    parser.add_argument("--memory_size", type=int, default=128, help="memory_size.")
    parser.add_argument("--hidden_size", type=int, default=128, help="hidden_size.")
    parser.add_argument("--input_size", type=int, default=128, help="Input size.")
    parser.add_argument("--head_size", type=int, default=128, help="head_size .")
    
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size. BE SURE TO REDUCE LR AS YOU INCREASE BATCH SIZE BY SQRT(BATCHSIZE) as stability increases the exp delta loss decreases. So needs lower LR.")
    parser.add_argument("--min_seq_len", type=int, default=4, help="Min sequence length.")
    parser.add_argument("--max_seq_len", type=int, default=100000, help="Max sequence length.")
    parser.add_argument("--step_seq_len", type=int, default=2, help="How much to step the sequence length.")
    parser.add_argument("--step_seq_len_loss_thresh", type=float, default=3.0, help="At what threshold to check the loss to step sequence length.")
    parser.add_argument("--patience_seq_len", type=int, default=50, help="How patient to be before stepping sequence length.")
    parser.add_argument("--tokenizer", type=str, default="hf", choices=["hf", None], 
                    help="Tokenizer to use. If 'hf', will use HuggingFace tokenizer. If None, will use character tokenizer.")
    parser.add_argument("--probe_normalization", type=str, choices=["true", "false"], default="false", help="normalizes the probe before applying to the model before query. helps limit the magnitude of the probe to the epsilon-hypersphere.")
    parser.add_argument("--gradient_normalization", type=str, choices=["true", "false"], default="false", help="normalizes the final gradient before updating the model weights.")
    parser.add_argument("--adaptive", type=str, choices=["true", "false"], default="false", help="if true biases the sampling by the adam_ratio, otherwise 0 mean sampling.")
    parser.add_argument("--adam", type=str, choices=["true", "false"], default="false", help="if true use adam, else use vanilla sgd.")
    parser.add_argument("--use_different_batch_per_meta_perturbation", type=str, choices=["true", "false"], default="false", help="if true use a different minibatch per meta_perturbation, else use all the same.")
    parser.add_argument("--normal_distribution", type=str, choices=["true", "false"], default="false", help="if true use normal distribution, otherwise use rademacher +/- 1.")
    parser.add_argument("--l1_norm", type=str, choices=["true", "false"], default="false", help="if true use L1 norm, else use L2 norm for grad normalization and for probe normalization (may help stablize for very large models).")
    parser.add_argument("--antithetic", type=str, choices=["true", "false"], default="true", help="if true use antithetic sampling for forward diff (not for central as its redundant), else dont.. it will double convergence rate!")
    parser.add_argument("--central_difference", type=str, choices=["true", "false"], default="true", help="if true use central difference, else use forward diff.")
    parser.add_argument("--learn_rate_schedule", type=str, choices=["true", "false"], default="true", help="if we want a lr schedule.")
    parser.add_argument("--model_type", type=str, choices=["DNC", "LSTM"], default="LSTM", help="Type of model to use.")
    parser.add_argument("--load_from_checkpoint", type=str, default=None,
                    help="Path to a .pt model checkpoint to load before training.")
    parser.add_argument("--verbose", type=str, choices=["true", "false"], default="false", help="verbosity settings.")
    args = parser.parse_args()
    
    
    # # TEST OVERFIT FAST DEMO! 
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for distributed training.")
    # parser.add_argument("--mode", type=str, choices=["test", "train"], default="test", help="Run mode: test or train.")
    # parser.add_argument("--max_iters", type=int, default=1e10, help="Maximum iterations for training.")
    # parser.add_argument("--learning_rate", type=float, default=0.01, help="Base learning rate (and eps, tied 1:1).")
    # parser.add_argument("--beta1", type=float, default=0.0, help="Base learning rate (and eps, tied 1:1).")
    # parser.add_argument("--beta2", type=float, default=0.0, help="Base learning rate (and eps, tied 1:1).")
    # parser.add_argument("--epsilon_tying_ratio", type=float, default=1.0, help="Perturbation scale epsilon (tied to learning rate).")
    # parser.add_argument("--weight_decay", type=float, default=0.0, help="Standard weight decay.")
    # parser.add_argument("--probe_dropout_rate", type=float, default=0., help="Dropout rate for probe vector.")
    # parser.add_argument("--wandb_proj", type=str, default=None, help="WandB project name (optional)")
    # parser.add_argument("--wandb_run", type=str, default=None, help="WandB run name (optional)")
    # parser.add_argument("--warmup_iters", type=int, default=0, help="Warmup iterations.")
    # parser.add_argument("--cosine_wavelength", type=int, default=100000000, help="Cosine LR wavelength, init to very high.")
    # parser.add_argument("--val_iters", type=int, default=10, help="Val iters, when we run val and log to wandb, and potentially checkpoint.")
    # parser.add_argument("--meta_perturbations", type=int, default=12, help="Number of Perturbations for all ranks per step.")
    # parser.add_argument("--scatter_batch", type=str, choices=["true", "false"], default="false", help="whether each perturbation should be on a different batch, if true, we sample (world_size-2)*batch_size x_ids and y per iter and scatter it to the appropriate ranks.")
    # parser.add_argument("--model_scale", type=int, default=1, help="Scaling factor for model dimensions.")
    # parser.add_argument("--num_heads", type=int, default=2**8, help="# dnc heads.")
    # parser.add_argument("--memory_size", type=int, default=128, help="memory_size.")
    # parser.add_argument("--hidden_size", type=int, default=2**17, help="hidden_size.")
    # parser.add_argument("--input_size", type=int, default=100, help="Input size.")
    # parser.add_argument("--head_size", type=int, default=0, help="head_size .")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size. BE SURE TO REDUCE LR AS YOU INCREASE BATCH SIZE BY SQRT(BATCHSIZE) as stability increases the exp delta loss decreases. So needs lower LR.")
    # parser.add_argument("--min_seq_len", type=int, default=10, help="Min sequence length.")
    # parser.add_argument("--max_seq_len", type=int, default=10, help="Max sequence length.")
    # parser.add_argument("--step_seq_len", type=int, default=10, help="How much to step the sequence length.")
    # parser.add_argument("--step_seq_len_loss_thresh", type=float, default=0.0, help="At what threshold to check the loss to step sequence length.")
    # parser.add_argument("--patience_seq_len", type=int, default=100, help="How patient to be before stepping sequence length.")    
    # parser.add_argument("--tokenizer", type=str, default=None, choices=["hf", None], 
    #                 help="Tokenizer to use. If 'hf', will use HuggingFace tokenizer. If None, will use character tokenizer.")
    # parser.add_argument("--probe_normalization", type=str, choices=["true", "false"], default="false", help="normalizes the probe before applying to the model before query. helps limit the magnitude of the probe to the epsilon-hypersphere.")
    # parser.add_argument("--gradient_normalization", type=str, choices=["true", "false"], default="false", help="normalizes the final gradient before updating the model weights.")
    # parser.add_argument("--adaptive", type=str, choices=["true", "false"], default="false", help="if true biases the sampling by the adam_ratio, otherwise 0 mean sampling.")
    # parser.add_argument("--adam", type=str, choices=["true", "false"], default="false", help="if true use adam, else use vanilla sgd.")
    # parser.add_argument("--use_different_batch_per_meta_perturbation", type=str, choices=["true", "false"], default="false", help="if true use a different minibatch per meta_perturbation, else use all the same.")
    # parser.add_argument("--normal_distribution", type=str, choices=["true", "false"], default="false", help="if true use normal distribution, otherwise use rademacher +/- 1.")
    # parser.add_argument("--l1_norm", type=str, choices=["true", "false"], default="false", help="if true use L1 norm, else use L2 norm for grad normalization and for probe normalization (may help stablize for very large models).")
    # parser.add_argument("--antithetic", type=str, choices=["true", "false"], default="false", help="if true use antithetic sampling, else dont.. it will double convergence rate!")
    # parser.add_argument("--central_difference", type=str, choices=["true", "false"], default="true", help="if true use central difference, else use forward diff.")
    # parser.add_argument("--learn_rate_schedule", type=str, choices=["true", "false"], default="false", help="if we want a lr schedule.")
     # parser.add_argument("--model_type", type=str, choices=["DNC", "LSTM"], default="LSTM", help="Type of model to use.")
    # parser.add_argument("--load_from_checkpoint", type=str, default=None,
    #                 help="Path to a .pt model checkpoint to load before training.")

    # parser.add_argument("--verbose", type=str, choices=["true", "false"], default="false", help="verbosity settings.")

    
    args = parser.parse_args()

    
    # TEMP OVERRIDE FOR NOW SO WE CAN DEBUG
    # args.wandb_proj = None
    args.scatter_batch = True if args.scatter_batch == "true" else False 
    args.central_difference = True if args.central_difference == "true" else False 
    args.use_different_batch_per_meta_perturbation = True if args.use_different_batch_per_meta_perturbation == "true" else False 
    args.learn_rate_schedule = True if args.learn_rate_schedule == "true" else False 

    verbose = True if args.verbose == "true" else False     

    #####################################################################################
    # SETUP TRAINING
    #####################################################################################

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{args.local_rank}")

    assert torch.cuda.is_available(), f"Rank {rank}: CUDA not available!"
    print(f"Rank {rank} using device {torch.cuda.current_device()}")

    # set the random seed differently per rank
    torch.manual_seed(torch.initial_seed() + rank) 

    if args.tokenizer == "hf":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.model_max_length = int(1e10)
        special_tokens_dict = {
            'bos_token': '<bos>',
            'eos_token': '<eos>',
            'pad_token': '<pad>'  # optional but good to have
        }
        
        tokenizer.add_special_tokens(special_tokens_dict)
        vocab_size = len(tokenizer)
        char_to_id = None
        id_to_char = None
        print("BOS ID:", tokenizer.bos_token_id, tokenizer.bos_token)
        print("EOS ID:", tokenizer.eos_token_id, tokenizer.eos_token)
        
        print("Decode BOS:", tokenizer.decode([tokenizer.bos_token_id]))
        print("Decode EOS:", tokenizer.decode([tokenizer.eos_token_id]))
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id).to(device)
    else:
        tokenizer = CharTokenizer()
        # vocab_list = tokenizer.vocab_list
        # char_to_id = tokenizer.char_to_id
        # id_to_char = tokenizer.id_to_char
        vocab_size = len(tokenizer.vocab_list)
        criterion = nn.CrossEntropyLoss().to(device)
        
    # vocab_list, char_to_id, id_to_char = get_char_vocab()
    # vocab_size = len(vocab_list)
    
    args.vocab_size = vocab_size

    # Derived values based on model_scale
    args.hidden_size = args.hidden_size * args.model_scale
    args.memory_size = args.memory_size * args.model_scale
    args.head_size = args.head_size * args.model_scale
    # args.input_size = args.input_size * args.model_scale
    args.num_heads = args.num_heads * args.model_scale

    if args.central_difference:
        run_name = f"cent_{args.mode}_lr{args.learning_rate}_scale{args.model_scale}_pdrop{args.probe_dropout_rate}"
    else:
        run_name = f"fwd_{args.mode}_lr{args.learning_rate}_scale{args.model_scale}_pdrop{args.probe_dropout_rate}"

    
    # Add model architecture details
    model_params = f"_h{args.hidden_size}"
    
    # Add training configuration
    train_params = f"_bs{args.batch_size}_seq{args.min_seq_len}_seq{args.max_seq_len}_b1_{args.beta1}_b2_{args.beta2}"
    
    # Add optimization details
    opt_params = f"_coswav_{args.cosine_wavelength}_wu{args.warmup_iters}_mp{args.meta_perturbations}"
    
    # Combine all parts
    if args.wandb_run=="" or args.wandb_run is None:
         args.wandb_run = run_name + model_params + train_params + opt_params
    
    
    log_msg("Trying first dist.barrier(), if hanging here, no infiniband likely on node, need to turn off p2p",rank,"if so, run export NCCL_P2P_DISABLE=1")
    dist.barrier()
    

    log_start("INIT MODEL", rank)

    # with torch.inference_mode():
    if args.model_type == "LSTM":
        embed = nn.Embedding(args.vocab_size, args.input_size, device=device)
        model = LSTM(
            input_size  = args.input_size,
            output_size = args.vocab_size,
            hidden_size = args.hidden_size,
            memory_size = args.memory_size,
            head_size   = args.head_size,
            num_heads   = args.num_heads,
            embed       = embed,
            device      = device,
        )
    elif args.model_type == "DNC":
        torch.backends.cudnn.enabled = False
        # time.sleep(rank) # we stagger the launch of DNC formation prevent RAM issues
        embed = nn.Embedding(args.vocab_size, args.input_size,device=device)
        model = DNC(input_size=args.input_size, output_size=args.vocab_size, hidden_size=args.hidden_size, memory_size=args.memory_size, head_size=args.head_size, num_heads=args.num_heads, embed=embed, device=device)
        # model.controller.flatten_parameters()
        
    else:
        raise Exception(f"Unk model type: {args.model_type}")
    
    
    
    distributed_optimizer = RGEOptimizerDistributed(
        model=model,
        learning_rate=args.learning_rate,
        probe_dropout_rate=args.probe_dropout_rate,
        epsilon_tying_ratio=args.epsilon_tying_ratio,
        beta1=args.beta1,
        beta2=args.beta2,
        adam=args.adam=="true",
        adaptive=args.adaptive=="true",
        weight_decay=args.weight_decay,
        probe_normalization=args.probe_normalization=="true",
        gradient_normalization=args.gradient_normalization=="true",
        meta_perturbations=args.meta_perturbations,
        scatter_batch = args.scatter_batch,
        normal_distribution =args.normal_distribution=="true",
        antithetic =args.antithetic=="true",
        l1_norm = args.l1_norm=="true",
        verbose=verbose
    )
    
    dist.barrier()

    if rank == 0:
        # load checkpoint on rank 0, and will get broadcasted to the rest
        if args.load_from_checkpoint is not None:
            loaded = load_distributed_checkpoint(distributed_optimizer, args.load_from_checkpoint, device, rank)
            if loaded:
                print(f"[rank{rank}] ✅ Checkpoint loaded successfully. Continuing training.")
            else:
                print(f"[rank{rank}] ⚠️ Failed to load checkpoint. Training from scratch.")
        else:
            print(f"[rank{rank}] No checkpoint specified. Training from raw weights.")
        

    model.eval()
    
    # if rank != 0:

    #     x_ids = []
    #     y = []
        
    #     for j in range(args.meta_perturbations):
    #         if j==0:
    #             # if scatter_batch, we want to sample only one batch and use that same batch for all perturbations on each rank each meta_pert. This only is important if args.mode == test really.
    #             x_ids_temp = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device, dtype=torch.long) # PLACEHOLDER
    #             y_temp = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device) # PLACEHOLDER
    #         x_ids.append(x_ids_temp)
    #         y.append(y_temp)
        

        
    #     if rank == 1:
    #         num_params = sum(p.numel() for p in model.parameters())
    #         num_layers = len(list(model.children()))
    #         print(f"[Init] Model has {num_params} parameters across {num_layers} layers.")
            

    

    # elif rank == 0:
    #     x_ids, y = None,  None
    #     pass

    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_layers = len(list(model.children()))
    print(f"[Init] Model has {num_params} parameters (trainable=={trainable_params} across {num_layers} layers.")
    
    dist.barrier()
    log_end("INIT MODEL", rank)
    
    

    
    if rank == distributed_optimizer.clean_rank:
        # Loss EMA tracking - one fast, one slow
        loss_ema_fast = None
        loss_ema_slow = None
        # ema_alpha_fast = 0.9  # Faster EMA
        # ema_alpha_slow = 0.999  # Slower EMA
        ema_alpha_fast = 1  # Faster EMA
        ema_alpha_slow = 1  # Slower EMA
        
        # Cosine learning rate scheduling parameters
        base_lr = args.learning_rate
        min_lr = base_lr * 0.001
        cosine_wavelength = args.cosine_wavelength #1000  # Length of each cosine cycle
        schedule_iteration = 0
        patience_counter = 0
        
        # Track previous loss
        prev_loss = float('inf')
        
    
    if rank == distributed_optimizer.clean_rank and args.wandb_proj is not None and wandb is not None:
        # wandb.init(project=args.wandb_proj, name=args.wandb_run)
        wandb.init(project=args.wandb_proj,name=args.wandb_run)
        wandb.config.update( vars(args) )

        print("[Rank 1] Initialized wandb logging.")

    #####################################################################################
    # Load the dataset on all ranks (dont need rank 0, but thats ok)
    #####################################################################################
    # generate OWT 
    ds = None
    iterr = 0
    while True:
        try:
            ds = load_dataset(
                "haritzpuerto/the_pile_00_OpenWebText2",
                # split="train",
                # cache_dir="/hf_cache/",
                # use_auth_token=False,  # Disable authentication to avoid API calls
                download_mode="reuse_dataset_if_exists"  # Reuse the cached dataset
            )
            # now we can use:
            # ds['train']
            # ds['validation']
            # ds['test']
            break
        except Exception as e:
            print("Hugging face issue...")
            print(e)
            time.sleep(5)
            iterr+=1
            if iterr>100:
                raise Exception("HUGGING FACE ISSUES AGAIN!")
    print("Got the OWT dataset!")
                
    #####################################################################################
    # START TRAINING
    #####################################################################################
    val_loss = 1e4 # placeholder value
    
    # Initialize seq_len scheduler variables
    current_max_seq_len = args.min_seq_len
    steps_below_thresh = 0

    start_time = datetime.datetime.now()
    with torch.inference_mode():
        
        for i in range(int(args.max_iters) ):
            #####################################################################################
            # Sample x_ids,y from dataset
            #####################################################################################
            
            if args.mode == "train" or (args.mode == "test"  and i==0):
                x_ids = []
                y = []

                for j in range(args.meta_perturbations):

                    if (args.mode == "train" and args.use_different_batch_per_meta_perturbation) or (args.mode == "test" and j==0) or (args.mode == "train" and j==0 and not args.use_different_batch_per_meta_perturbation):
                     
                        sampled_seq_len = random.randint(args.min_seq_len, current_max_seq_len)


                        x_ids_temp, y_temp = generate_openwebtext_task(
                            num_samples     = args.batch_size,
                            ds              = ds["train"],
                            tokenizer       = tokenizer,
                            min_tokens      = args.min_seq_len,
                            max_tokens      = sampled_seq_len,
                            device          = device
                        )
                        
                        # x_ids_temp, y_temp = generate_openwebtext_task_unified(
                        #     num_samples=args.batch_size,
                        #     ds=ds["train"],
                        #     tokenizer=hf_tokenizer if args.tokenizer == "hf" else None,
                        #     min_tokens=args.min_seq_len,
                        #     max_tokens=sampled_seq_len,
                        #     char_tokenizer=(args.tokenizer != "hf"),
                        #     char_to_id=char_to_id,
                        #     str_to_tensor=str_to_tensor,
                        #     return_strings=False,
                        #     device=device
                        # )
                        
                            
        
                    x_ids.append(x_ids_temp)
                    y.append(y_temp)
                        
                    if verbose:
                        print(f" x_ids {x_ids_temp.shape} y {y_temp.shape} x_ids dtype {x_ids_temp.dtype} y dtype {y_temp.dtype}" )

                    
            
                    
            dist.barrier()
                    
            #####################################################################################
            # TRAIN THE MODEL
            #####################################################################################
            # if distributed_optimizer.adam:
            
            if args.central_difference:
                # central_difference_distributed
                train_loss = distributed_optimizer.dist_cdrge_step(x_ids, y, criterion, iteration=i)
            else:
                train_loss = distributed_optimizer.forward_difference_distributed(x_ids, y, criterion, iteration=i)

            
            # else: # todo, need to implement this to be more efficient with GPUs
            #     train_loss = distributed_optimizer.distributed_step_SPSA(x_ids, y, criterion, iteration=i)

            

            #####################################################################################
            # CHECKPOINT THE MODEL RARELY
            #####################################################################################
            if args.mode == "train" and (i+1) % (args.val_iters) == 0:
                
                save_distributed_checkpoint(distributed_optimizer, 
                                            args.wandb_run, 
                                            "rnn_checkpoints", 
                                            rank)
                dist.barrier()
            
                
            
            if rank == distributed_optimizer.clean_rank:
                #####################################################################################
                # UPDATE THE LEARNING RATE WITH OUR FAST/SLOW EMA COSINE LR SCHEDULE
                #####################################################################################
            
                if loss_ema_fast is None:
                    # Initialize both EMAs with the current loss value
                    loss_ema_fast = train_loss
                    loss_ema_slow = train_loss

                
                loss_ema_fast = ema_alpha_fast * loss_ema_fast + (1 - ema_alpha_fast) * train_loss
                loss_ema_slow = ema_alpha_slow * loss_ema_slow + (1 - ema_alpha_slow) * train_loss
                if args.learn_rate_schedule:
                    if i < args.warmup_iters:
                        # Linear warmup
                        distributed_optimizer.learning_rate = base_lr * (i / args.warmup_iters)
                        distributed_optimizer.learning_rate = max(1e-8,distributed_optimizer.learning_rate)
                    else:
                        
                        # Check if the fast EMA is higher than the slow EMA (by a small threshold)
                        # if loss_ema_fast > (loss_ema_slow + 1e-5):
                        #     patience_counter += 1
                        # else:
                        #     patience_counter = 0  # reset if condition is not met
                    
                        # # Only step the cosine schedule if we have been patient enough
                        # if patience_counter >= args.schedule_patience:
                        #     schedule_iteration += 1
                        #     patience_counter = 0  # reset the counter after stepping
                    
                        # Compute the position within the cosine cycle.
                        # Here, schedule_iteration is used to determine how far we are along the cycle.
                        # cycle_iteration = schedule_iteration % cosine_wavelength
                        
                        # progress = schedule_iteration / cosine_wavelength
                        
                        progress = i / cosine_wavelength
                        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                        distributed_optimizer.learning_rate = min_lr + (base_lr - min_lr) * cosine_factor
    
    
                    #####################################################################################
                    # UPDATE THE LEARNING RATE WITH OUR FAST/SLOW EMA COSINE LR SCHEDULE
                    #####################################################################################
                    if loss_ema_fast < args.step_seq_len_loss_thresh:
                        steps_below_thresh += 1
                    else:
                        steps_below_thresh = 0
                
                    if steps_below_thresh >= args.patience_seq_len:
                        if current_max_seq_len + args.step_seq_len <= args.max_seq_len:
                            current_max_seq_len += args.step_seq_len
                            print("="*50)
                            print(f"[INFO] Increased current_max_seq_len from {current_max_seq_len-args.step_seq_len} to {current_max_seq_len} at iter {i} (loss_ema_fast={loss_ema_fast:.4f})")
                            print("="*50)
                        else:
                            current_max_seq_len = args.max_seq_len
                        steps_below_thresh = 0
                        
                

                #####################################################################################
                # RUN VALIDATION
                #####################################################################################
                if rank == distributed_optimizer.clean_rank and (i+1) % args.val_iters == 0:

                    if args.mode == "train":
        
                        # # GENERATE VAL BATCH and run
                        # x_strs, y_strs = generate_openwebtext_task_str(
                        #                             args.batch_size,
                        #                             1,
                        #                             ds,
                        #                             train = False,
                        #                             total_seq_len=args.seq_len,
                        #                             verbose = False
                        #                         )  
                        # if args.tokenizer == "hf":
                        #     x_ids = hf_tokenizer(x_strs, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
                        #     y = hf_tokenizer(y_strs, return_tensors='pt', padding=True, truncation=True).input_ids.to(device)
                        # else:
                        #     x_ids = str_to_tensor(x_strs, char_to_id).to(device)
                        #     y = str_to_tensor(y_strs, char_to_id).to(device)

                        val_x_ids, val_y  = generate_openwebtext_task(
                            num_samples     = args.batch_size,
                            ds              = ds["validation"],
                            tokenizer       = tokenizer,
                            min_tokens      = args.min_seq_len,
                            max_tokens      = sampled_seq_len,
                            device          = device
                        )
                        # val_x_ids, val_y = generate_openwebtext_task_unified(
                        #     num_samples=args.batch_size,
                        #     ds=ds["validation"],
                        #     tokenizer=hf_tokenizer if args.tokenizer == "hf" else None,
                        #     min_tokens=args.min_seq_len,
                        #     max_tokens=sampled_seq_len,
                        #     char_tokenizer=(args.tokenizer != "hf"),
                        #     char_to_id=char_to_id,
                        #     str_to_tensor=str_to_tensor,
                        #     return_strings=False,
                        #     device=device
                        # )

                        
                        val_loss, val_preds = teacher_forcing_loss_emb_parallel(model, val_x_ids, val_y, criterion, return_predictions=True)

                        if args.tokenizer == "hf":
                            decode_fn = lambda ids: tokenizer.batch_decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)

                        else:
                            decode_fn = lambda ids: ["".join([id_to_char[i.item()] for i in seq if i.item() in id_to_char]) for seq in ids]
                        
                        decoded_preds = decode_fn(val_preds)
                        decoded_targets = decode_fn(val_y)
                        decoded_inputs = decode_fn(val_x_ids)
                        
                        print("="*30)
                        print("Validation predictions:")
                        for jjj in range(min(len(decoded_preds), 5)):  # show up to 5 samples
                            print(f"[Sample {jjj}] Input:    '{decoded_inputs[jjj]}'")
                            print(f"[Sample {jjj}] Target:   '{decoded_targets[jjj]}'")
                            print(f"[Sample {jjj}] Predicted:'{decoded_preds[jjj]}'")
                        print("="*30)
                    
                    #####################################################################################
                    # log to wandb every val_iters iterations.
                    #####################################################################################
                    if args.wandb_proj is not None and wandb is not None:
                        # Compute a dummy weight decay loss (if applicable)
                        weight_decay_loss = 0.0
                        for param in model.parameters():
                            if param.requires_grad:
                                weight_decay_loss += (1e-2 / 2) * torch.sum(param ** 2)  # using 1e-2 as dummy weight_decay
    
    
                        
                        log_data = {
                            "train_loss": train_loss, 
                            "val_loss": val_loss, 
                            "loss_ema_fast":loss_ema_fast,
                            "loss_ema_slow":loss_ema_slow,
                            "current_max_seq_len":current_max_seq_len,
                            "lr": distributed_optimizer.learning_rate,
                            "weight_decay_loss": weight_decay_loss.item(),
                        }
                        
                        try:
                            wandb.log(log_data, step=i)
                        except Exception as e:
                            print(e)
                    
                
                #####################################################################################
                # Log to stdout
                #####################################################################################
                print("="*50)
                average_time_per_iter = (datetime.datetime.now() - start_time)
                start_time = datetime.datetime.now()
                print(f"[Train] Iteration {i }, train_loss = {train_loss}, loss_ema_fast = {loss_ema_fast}, loss_ema_slow = {loss_ema_slow}, lr = {distributed_optimizer.learning_rate}, val_loss = {val_loss}, current_max_seq_len = {current_max_seq_len} time per step (TOTAL) = { average_time_per_iter }")
                

            dist.barrier()
            if rank == distributed_optimizer.clean_rank:
                if train_loss < 0.1:

                    end_time = datetime.datetime.now()
                    print("="*50)
                    print("="*50)
                    print("="*50)
                    log_msg("FINISHED TRAINING", rank, f"in {i} iterations acheived {train_loss} loss in {average_time_per_iter} seconds per iter.")
                    print(f"[Init] Model has {num_params} parameters across {num_layers} layers.")
            
                    print("="*50)
                    print("="*50)
                    print("="*50)
                    time.sleep(200)
                    break

            
        dist.destroy_process_group()
        

if __name__ == "__main__":
    main()
