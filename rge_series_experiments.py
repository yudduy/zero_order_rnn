#!/usr/bin/env python3
"""
In series RGE experiments on LSTM and DNC training on a few tasks: overfit, copy, sort, reverse, add, penn treebank, etc. 
python rge_series_experiments.py --unit_test --max_iterations 10 --micro_batch_size 1 --seq_length 10 --device cuda:1 --solver 1SPSA
 choices=["BPTT", "1SPSA", "1.5-SPSA", "2SPSA" ... ] )
"""

import argparse
import os
import random
import string
import time
import math
import json
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
from flashrnn.flashrnn import flashrnn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from simpletokenizers.simpletokenizers import CharTokenizer, NumericTokenizer, get_tiktoken
from models.models        import LSTM, DNC
from tasks.tasks          import get_examples_for_task, compute_task_loss, compute_task_accuracy

import pdb

# Optional imports
try:
    import tiktoken  # Used only if tokenizer='hf_tiktoken' is specified
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

try:
    # Requires:  flashrnn ≥ 0.1  (pip install flash-rnn or pip install -e . from repo RECOMMENDED!) OR it will fallback to pytroch and be slow.
    from flashrnn.flashrnn import flashrnn           # official package name
    FLASH_OK = True
except ModuleNotFoundError:
    FLASH_OK = False
    warnings.warn("flashrnn not found – LSTM will fall back to PyTorch LSTM.",
                  RuntimeWarning)

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb is not installed. Experiment tracking will be disabled.")
    print("To enable, install with: pip install wandb")

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)




def compute_task_loss(logits, ids_np, tok, task, verbose=False):
      """Compute loss with gradients based on task type with proper shift for predictions"""
      B, T, V = logits.shape
      device = logits.device

      # Different separator tokens for different tasks
      if task in ["copy", "repeat_copy", "sort", "reverse"]:
          sep = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
      elif task == "add":
          sep = tok.token_to_id.get("=", 0)
      else:  # penn_tree_bank or any others
          # For PTB, predict next token for each position
          targets = torch.as_tensor(ids_np[:, 1:], device=device, dtype=torch.long)
          shifted_logits = logits[:, :-1, :]
          loss = torch.nn.functional.cross_entropy(
              shifted_logits.reshape(-1, V),
              targets.reshape(-1),
              reduction='mean'
          )
          return loss

      # Get space token ID
      space_id = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)

      # Initialize running total and count
      total_loss = torch.tensor(0.0, device=device, requires_grad=True)
      count = 0

      for b in range(B):
          try:
              # Find separator position
              pos = list(ids_np[b]).index(sep)
          except ValueError:
              continue

          # Get output part (everything after the separator)
          output_part = ids_np[b][pos + 1:]

          if len(output_part) <= 1:  # Need at least 2 tokens for prediction
              continue

          # Find content indices in output (non-space tokens)
          content_indices = [i for i, t in enumerate(output_part) if t != space_id]
          if not content_indices:
              continue

          # Debug prints
          if verbose and b < 3:
              input_text = tok.decode(ids_np[b][:pos+1])
              target_text = tok.decode([output_part[i] for i in content_indices if i < len(output_part)])
              print(f"\nSample {b}:")
              print(f"  Input: '{input_text}'")
              print(f"  Target: '{target_text}'")

          # The crucial correction: 
          # For each position i, use logits at position pos+1+i to predict token at position i+1
          pred_logits = []
          tgt_tokens = []

          # For each position except the last one in output_part
          for i in range(len(output_part) - 1):
              # Only include if it's a content token or space following content
              if (i in content_indices) or (i-1 in content_indices and output_part[i] == space_id):
                  if pos + 1 + i < T:  # Ensure we're within logits sequence length
                      pred_logits.append(logits[b, pos + 1 + i])
                      tgt_tokens.append(output_part[i + 1])  # Predict NEXT token

          # Skip if we have no valid targets
          if not tgt_tokens:
              continue

          # Stack logits and prepare targets
          if pred_logits:
              stacked_logits = torch.stack(pred_logits)
              targets = torch.tensor(tgt_tokens, device=device, dtype=torch.long)

              # stacked_logits = stacked_logits.to(torch.float64)   # <— or .float() or nothing.. not sure it really helps.
              stacked_logits = stacked_logits.to(torch.float32) 
              # Compute batch loss
              batch_loss = torch.nn.functional.cross_entropy(
                  stacked_logits,
                  targets,
                  reduction='sum'
              )

              total_loss = total_loss + batch_loss
              count += len(tgt_tokens)

      # If no valid samples, return dummy loss
      if count == 0:
          return logits.sum() * 0.0

      return total_loss / count

def compute_task_loss_with_gradients(logits, ids_np, tok, task, verbose=False):
      """Compute loss with gradients based on task type with proper shift for predictions"""
      B, T, V = logits.shape
      device = logits.device

      # Different separator tokens for different tasks
      if task in ["copy", "repeat_copy", "sort", "reverse"]:
          sep = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
      elif task == "add":
          sep = tok.token_to_id.get("=", 0)
      else:  # penn_tree_bank or any others
          # For PTB, predict next token for each position
          targets = torch.as_tensor(ids_np[:, 1:], device=device, dtype=torch.long)
          shifted_logits = logits[:, :-1, :]
          loss = torch.nn.functional.cross_entropy(
              shifted_logits.reshape(-1, V),
              targets.reshape(-1),
              reduction='mean'
          )
          return loss

      # Get space token ID
      space_id = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)

      # Initialize running total and count
      total_loss = torch.tensor(0.0, device=device, requires_grad=True)
      count = 0

      for b in range(B):
          try:
              # Find separator position
              pos = list(ids_np[b]).index(sep)
          except ValueError:
              continue

          # Get output part (everything after the separator)
          output_part = ids_np[b][pos + 1:]

          if len(output_part) <= 1:  # Need at least 2 tokens for prediction
              continue

          # Find content indices in output (non-space tokens)
          content_indices = [i for i, t in enumerate(output_part) if t != space_id]
          if not content_indices:
              continue

          # Debug prints
          if verbose and b < 3:
              input_text = tok.decode(ids_np[b][:pos+1])
              target_text = tok.decode([output_part[i] for i in content_indices if i < len(output_part)])
              print(f"\nSample {b}:")
              print(f"  Input: '{input_text}'")
              print(f"  Target: '{target_text}'")

          # The crucial correction: 
          # For each position i, use logits at position pos+1+i to predict token at position i+1
          pred_logits = []
          tgt_tokens = []

          # For each position except the last one in output_part
          for i in range(len(output_part) - 1):
              # Only include if it's a content token or space following content
              if (i in content_indices) or (i-1 in content_indices and output_part[i] == space_id):
                  if pos + 1 + i < T:  # Ensure we're within logits sequence length
                      pred_logits.append(logits[b, pos + 1 + i])
                      tgt_tokens.append(output_part[i + 1])  # Predict NEXT token

          # Skip if we have no valid targets
          if not tgt_tokens:
              continue

          # Stack logits and prepare targets
          if pred_logits:
              stacked_logits = torch.stack(pred_logits)
              targets = torch.tensor(tgt_tokens, device=device, dtype=torch.long)

              # Compute batch loss
              batch_loss = torch.nn.functional.cross_entropy(
                  stacked_logits,
                  targets,
                  reduction='sum'
              )

              total_loss = total_loss + batch_loss
              count += len(tgt_tokens)

      # If no valid samples, return dummy loss
      if count == 0:
          return logits.sum() * 0.0

      return total_loss / count

def compute_task_accuracy(logits, ids_np, tok, task, verbose=False):
    """Compute accuracy based on task type"""
    B, T, V = logits.shape
    device = logits.device
    
    total_correct = 0
    total_tokens = 0
    content_correct = 0
    content_tokens = 0
    eos_correct = 0
    eos_tokens = 0
    
    # Different separator tokens for different tasks
    if task in ["copy", "repeat_copy", "sort", "reverse"]:
        sep = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
    elif task == "add":
        sep = tok.token_to_id.get("=", 0)
    else:  # penn_tree_bank or any others
        # For PTB, predict next token for each position
        targets = torch.as_tensor(ids_np[:, 1:], device=device, dtype=torch.long)
        predictions = logits[:, :-1, :].argmax(dim=-1)
        correct = (predictions == targets).sum().item()
        total = targets.numel()
        return correct / total if total > 0 else 0.0
    
    # Get space token ID
    space_id = tok.char_to_id.get(" ", 0) if hasattr(tok, 'char_to_id') else tok.token_to_id.get(" ", 0)
    
    for b in range(B):
        try:
            # Find separator position
            pos = list(ids_np[b]).index(sep)
        except ValueError:
            continue
        
        # Get target tokens after separator
        output_part = ids_np[b][pos+1:]
        
        # Find content tokens (non-space) and add EOS token
        content_indices = [i for i, t in enumerate(output_part) if t != space_id]
        
        if not content_indices:
            continue
        
        # Add one more index after the last content token to include an EOS space
        last_content_idx = max(content_indices)
        eos_idx = last_content_idx + 1
            
        # Get all predictions for the output part
        pred_logits = logits[b, pos+1:pos+1+len(output_part)]
        pred_ids = pred_logits.argmax(dim=-1).cpu().numpy()
        
        # Debug prints for verifying processing
        if verbose and b < 3:  # Limit debug to first 3 samples
            input_part = ids_np[b][:pos+1]
            input_text = tok.decode(input_part)
            
            # Decode actual target and prediction
            content_target = [output_part[i] for i in content_indices if i < len(output_part)]
            content_pred = [pred_ids[i] for i in content_indices if i < len(pred_ids)]
            
            target_text = tok.decode(content_target)
            pred_text = tok.decode(content_pred)
            
            # Include EOS token if it exists
            full_target = content_target.copy()
            full_pred = content_pred.copy()
            
            if eos_idx < len(output_part):
                full_target.append(output_part[eos_idx])
            if eos_idx < len(pred_ids):
                full_pred.append(pred_ids[eos_idx])
                
            full_target_text = tok.decode(full_target)
            full_pred_text = tok.decode(full_pred)
            
            print(f"\nAccuracy Sample {b}:")
            print(f"  Input: '{input_text}'")
            print(f"  Target (content): '{target_text}'")
            print(f"  Pred (content): '{pred_text}'")
            print(f"  Target (with EOS): '{full_target_text}'")
            print(f"  Pred (with EOS): '{full_pred_text}'")
        
        # Count correct content token predictions
        for i in content_indices:
            if i < len(pred_ids):  # Make sure we're in bounds
                total_tokens += 1
                content_tokens += 1
                if pred_ids[i] == output_part[i]:
                    total_correct += 1
                    content_correct += 1
                    
        # Count correct EOS token prediction if it exists
        if eos_idx < len(output_part) and eos_idx < len(pred_ids):
            total_tokens += 1
            eos_tokens += 1
            if pred_ids[eos_idx] == output_part[eos_idx]:
                total_correct += 1
                eos_correct += 1
    
    # Report detailed stats if verbose
    if verbose:
        content_acc = content_correct / max(content_tokens, 1) * 100
        eos_acc = eos_correct / max(eos_tokens, 1) * 100
        print(f"\nDetailed accuracy stats:")
        print(f"  Content tokens: {content_correct}/{content_tokens} = {content_acc:.2f}%")
        print(f"  EOS tokens: {eos_correct}/{eos_tokens} = {eos_acc:.2f}%")
        print(f"  Overall: {total_correct}/{total_tokens} = {total_correct / max(total_tokens, 1) * 100:.2f}%")
    
    return total_correct / max(total_tokens, 1)

# ───────── Memory Estimation ─────────
def estimate_memory_usage(model_params, B, T, G, N, D, V, S):
    """
    Estimate VRAM usage for the model

    Args:
        model_params: List of model parameters
        B: Batch size
        T: Sequence length
        G: Number of gates
        N: Number of heads
        D: Dimension per head
        V: Vocabulary size
        S: Number of states (2 for LSTM)

    Returns:
        param_memory: Memory used by parameters (in MB)
        activation_memory: Memory used by activations (in MB)
        total_memory: Total memory usage (in MB)
    """
    # Count parameter memory
    param_count = sum(p.numel() for p in model_params)
    param_memory = param_count * 2 / (1024 * 1024)  # bfloat16 = 2 bytes, convert to MB

    # Estimate activation memory
    # Key activations:
    # 1. One-hot encoding: B * T * V * 2 bytes
    # 2. Input embeddings: B * T * G * N * D * 2 bytes
    # 3. Hidden states: S * B * T * N * D * 2 bytes
    # 4. Logits: B * T * V * 2 bytes

    activation_bytes = (
        B * T * V * 2 +                # One-hot encoding
        B * T * G * N * D * 2 +        # Input embeddings
        S * B * T * N * D * 2 +        # Hidden states
        B * T * V * 2                  # Logits
    )

    activation_memory = activation_bytes / (1024 * 1024)  # Convert to MB
    total_memory = param_memory + activation_memory

    return param_memory, activation_memory, total_memory

# ───────── Parameter Perturbation and Optimization ─────────
def generate_rademacher_noise(param, epsilon):
    # 1) make an int8 tensor of 0 / 1
    noise = torch.randint(
        0, 2,                        # values {0,1}
        param.shape,
        device=param.device,
        dtype=torch.int8
    )
    # 2) map {0,1} → {-1,+1}  and cast to param.dtype (fp16)
    noise = noise.mul(2).sub_(1).to(param.dtype)

    # 3) scale by epsilon
    return noise.mul_(epsilon)
    
def generate_perturbation(param, epsilon, distribution='rad', seed=None):
    """Generate perturbation according to the specified distribution with optional seed"""
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)

    if distribution == 'rad':
        # Rademacher distribution (±1)
        return generate_rademacher_noise(param, epsilon)
    elif distribution == 'normal':
        # Normal distribution
        return torch.randn_like(param) * epsilon
    elif distribution == 'uniform':
        # Uniform distribution in [-1, 1]
        return (torch.rand_like(param) * 2 - 1) * epsilon
    else:
        raise ValueError(f"Unknown distribution: {distribution}")



def SPSA1_5(model_params: List[torch.Tensor],
            loss_fn,
            lr: float,
            epsilon: float,
            iterations: int,
            num_perturbations: int = 20,
            distribution: str = 'rad',
            antithetic: bool = False,
            use_adaptive_step: bool = False,
            clip_grad_norm: float = 0.0,
            cache_gradients: bool = False,
            lambda_reg: float = 1., #1e-2,
            CHUNK_SIZE: int = 2**15,
            args=None):
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"CDRGE: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"CDRGE: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Store gradients for each parameter
    if is_verbose:
        print("CDRGE: Initializing gradients...")
    
    gradients = [torch.zeros_like(p.data) for p in model_params]


    all_seeds = [random.randint(0, 2**32 - 1) for i in range(num_perturbations)]
    grad_estimate_list = []
    sum_of_losses = 0

    if cache_gradients:
        # one tensor per param to accumulate Σ coeff·probe
        grad_buffer = [torch.zeros_like(p.data) for p in model_params]
    else:
        grad_estimate_list = []          # original storage
    
    
    if is_verbose:
        print("starting perturbations")

    total_clean_loss = 0.0
    for _ in range(macro_batch_size):
        clean_loss_batch = loss_fn()
        batch_loss = clean_loss_batch.item() if hasattr(clean_loss_batch, 'item') else float(clean_loss_batch)
        total_clean_loss += batch_loss

    clean_loss = total_clean_loss / macro_batch_size
    
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = all_seeds[j]
        
        ##################################################################
        # Apply positive perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,                
                    epsilon,
                    distribution,
                    seed=pert_seed + param_idx
                )
                chunk.add_(probe)         # in-place update of that slice only
            
            # Generate and apply perturbation
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        
        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size

        ##################################################################
        # Apply negative perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,               
                     -2*epsilon,
                    distribution,
                    seed=pert_seed + param_idx
                )
                chunk.add_(probe)         # in-place update of that slice only
            # Generate and apply perturbation
            # probe = generate_perturbation(param, -2*epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        # Compute loss with negative perturbation - average across macro batches
        total_neg_loss = 0.0
        for _ in range(macro_batch_size):
            neg_loss_batch = loss_fn()
            batch_loss = neg_loss_batch.item() if hasattr(neg_loss_batch, 'item') else float(neg_loss_batch)
            total_neg_loss += batch_loss
            
        neg_loss = total_neg_loss / macro_batch_size
        
        # 2-sided estimation: (f(x+ε) - f(x-ε))/2 
        grad_estimate = -1*(pos_loss - neg_loss) / (2*num_perturbations) # no epsilon bc we mult later so it cancels out
        curv = abs(pos_loss -2*clean_loss + neg_loss)/(epsilon**2) 
        # curvature = max( 0.001 * curv, lambda_reg)
        curvature = max( curv**args.saturating_alpha, lambda_reg)
        
        # curvature = max( np.log(curv + 1e-8), lambda_reg)
        grad_estimate_list.append(grad_estimate / curvature)
        sum_of_losses += (pos_loss + neg_loss) / (2)

        if True:#is_verbose:
          print(f"  grad_estimate {grad_estimate} curv {curv} denom {curvature} ")
        
        # for param_idx, param in enumerate(model_params):
        #     for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
        #         probe = generate_perturbation(
        #             chunk,               
        #             1,
        #             distribution,
        #             seed=pert_seed + param_idx
        #         )
        #         chunk.add_(probe,alpha=epsilon)         # in-place update of that slice only
        #         if cache_gradients:
        #             # coeff / epsilon  =  grad_estimate / (+ε)
        #             grad_chunk = grad_buffer[param_idx][chunk]      # same view
        #             grad_chunk.add_(probe, alpha=grad_estimate)

        for param_idx, param in enumerate(model_params):
            rows = param.size(0)
        
            for start in range(0, rows, CHUNK_SIZE):
                end   = min(start + CHUNK_SIZE, rows)
        
                slice_param = param.data[start:end]        # view onto weights
                probe = generate_perturbation(
                            slice_param,                   # same shape
                            1,
                            distribution,
                            seed=pert_seed + param_idx
                        )
        
                slice_param.add_(probe, alpha=epsilon)     # restore weights
        
                if cache_gradients:
                    # accumulate coeff·probe into the matching slice of grad_buffer
                    grad_buffer[param_idx][start:end].add_(probe,
                                                           alpha=grad_estimate)
            
            # # Regenerate the same noise
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)

            # # Restore parameter by adding the perturbation
            # param.data.add_(probe)
            

    # for idx, param in enumerate(model_params):
    #     # if idx == 3:                      # skip states tensor
    #     #     continue
    
    #     # -------- accumulate gradient in FP32 -------------------------
    #     grad = None
    #     for j in range(num_perturbations):
    #         seed         = all_seeds[j] + idx
    #         coeff        = grad_estimate_list[j]       # (f⁺−f⁻)/2ε
    #         probe   = generate_perturbation(
    #                            param, coeff, distribution, seed)
            
    #         grad = probe if grad is None else grad + probe

    #     param.data.add_(grad, alpha= -1./float(num_perturbations)   )
    #     # -------- SGD step in FP32, then cast back -------------------
    #     # param_fp32 = param.data.float()
    #     # param_fp32.add_( -grad )                          # **NO extra ε/n**
    #     # param.data.copy_( param_fp32.to(param.dtype) )         # back to bf16
    # # SGD update (same as before)
    # param.data.add_(grad, alpha=-1. / float(num_perturbations))

    if cache_gradients:
        for buf, param in zip(grad_buffer, model_params):
            param.data.add_(buf) 

    else: # calculate this post from the seeds
        for idx, param in enumerate(model_params):
            # grad = None
            for j in range(num_perturbations):
                seed  = all_seeds[j] + idx
                coeff = grad_estimate_list[j]                 # (f⁺−f⁻)/2n # no ε bc ε == lr and they cancel out
        
                # ── NEW: accumulate probe chunk-by-chunk ─────────────────────
                for chunk in torch.split(param, CHUNK_SIZE, dim=0):     # 8k rows
                    probe = generate_perturbation(
                                chunk, coeff, distribution, seed)
                    chunk.add_(probe) 
        
                # grad = probe if grad is None else grad + probe


    if is_verbose:
        print(f"done updating the model {time.time() - start_time}")

    # Track metrics
    elapsed = time.time() - start_time
    loss = clean_loss
    metrics['train_loss'].append(loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"CD-RGE: Optimization completed in {elapsed:.2f}s with loss {loss:.6f}")
    return metrics

    
def cdrge_optimize(model_params, 
                   loss_fn, 
                   lr, 
                   epsilon, 
                   iterations, 
                   num_perturbations=20,
                   distribution='rad', 
                   antithetic=False, 
                   use_adaptive_step=False, 
                   clip_grad_norm=0.0,
                   cache_gradients = False,
                   CHUNK_SIZE = 2**15, # TUNE THIS TO MAKE IT FIT IN VRAM BUT ALSO BE FAST
                   args=None):
    """Central Difference Random Gradient Estimation optimization

    A memory-efficient implementation that avoids parameter copies
    and uses seeded RNG for consistent perturbations.

    Args:
        model_params: List of model parameters to optimize
        loss_fn: Function that computes the loss
        lr: Learning rate
        epsilon: Perturbation scale (same as lr for numerical stability)
        iterations: Number of optimization iterations
        num_perturbations: Number of random perturbations to use
        distribution: Type of distribution for perturbations ('rad', 'normal', 'uniform')
        antithetic: Whether to use antithetic sampling (negative perturbation pairs)
        use_adaptive_step: Whether to use adaptive step sizes (default: False)
        clip_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        args: Program arguments with verbose flag

    Returns:
        Dictionary of metrics through training
    """
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"CDRGE: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"CDRGE: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Store gradients for each parameter
    if is_verbose:
        print("CDRGE: Initializing gradients...")
    


    all_seeds = [random.randint(0, 2**32 - 1) for i in range(num_perturbations)]
    grad_estimate_list = []
    sum_of_losses = 0

    
    if args.beta1>0:                        # Nesterov buffer (m starts at 0)
        coordinate_momentum   = [torch.zeros_like(p.data) for p in model_params]
        
    if args.beta2>0:                        # RMSProp accumulator (v starts at 1)
        coordinate_variance = [torch.ones_like(p.data) for p in model_params]


    if args.beta1>0 or args.beta2>0:
        cache_gradients = True  # you dont need to have this, but it ensures useability for now.. 
                                # unless you want to implement the non-cache version for momentum or variance

    
    if cache_gradients:
        # one tensor per param to accumulate Σ coeff·probe
        grad_buffer = [torch.zeros_like(p.data) for p in model_params]
            # gradients = [torch.zeros_like(p.data) for p in model_params]

        
    else:
        grad_estimate_list = []          # original storage
    
    
    if is_verbose:
        print("starting perturbations")
        
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = all_seeds[j]
        
        ##################################################################
        # Apply positive perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,                
                    epsilon,
                    distribution,
                    seed=pert_seed + param_idx
                )
                chunk.add_(probe)         # in-place update of that slice only
            
            # Generate and apply perturbation
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        
        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size

        ##################################################################
        # Apply negative perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
                probe = generate_perturbation(
                    chunk,               
                     -2*epsilon,
                    distribution,
                    seed=pert_seed + param_idx
                )
                chunk.add_(probe)         # in-place update of that slice only
            # Generate and apply perturbation
            # probe = generate_perturbation(param, -2*epsilon, distribution, seed=pert_seed + param_idx)
            # param.data.add_(probe)

        # Compute loss with negative perturbation - average across macro batches
        total_neg_loss = 0.0
        for _ in range(macro_batch_size):
            neg_loss_batch = loss_fn()
            batch_loss = neg_loss_batch.item() if hasattr(neg_loss_batch, 'item') else float(neg_loss_batch)
            total_neg_loss += batch_loss
            
        neg_loss = total_neg_loss / macro_batch_size
        
        # 2-sided estimation: (f(x+ε) - f(x-ε))/2 
        grad_estimate = -1*(pos_loss - neg_loss) / (2*num_perturbations) # no epsilon bc we mult later so it cancels out
        sum_of_losses += (pos_loss + neg_loss) / (2)
        grad_estimate_list.append(grad_estimate)
        
        # for param_idx, param in enumerate(model_params):
        #     for chunk in torch.split(param.data, CHUNK_SIZE, dim=0):   # 8 k rows each
        #         probe = generate_perturbation(
        #             chunk,               
        #             1,
        #             distribution,
        #             seed=pert_seed + param_idx
        #         )
        #         chunk.add_(probe,alpha=epsilon)         # in-place update of that slice only
        #         if cache_gradients:
        #             # coeff / epsilon  =  grad_estimate / (+ε)
        #             grad_chunk = grad_buffer[param_idx][chunk]      # same view
        #             grad_chunk.add_(probe, alpha=grad_estimate)

        for param_idx, param in enumerate(model_params):
            rows = param.size(0)
        
            for start in range(0, rows, CHUNK_SIZE):
                end   = min(start + CHUNK_SIZE, rows)
        
                slice_param = param.data[start:end]        # view onto weights
                probe = generate_perturbation(
                            slice_param,                   # same shape
                            1,
                            distribution,
                            seed=pert_seed + param_idx
                        )
        
                slice_param.add_(probe, alpha=epsilon)     # restore weights
        
                if cache_gradients:
                    # accumulate coeff·probe into the matching slice of grad_buffer
                    grad_buffer[param_idx][start:end].add_(probe,
                                                           alpha=grad_estimate)
            
            # # Regenerate the same noise
            # probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)

            # # Restore parameter by adding the perturbation
            # param.data.add_(probe)
            

    # for idx, param in enumerate(model_params):
    #     # if idx == 3:                      # skip states tensor
    #     #     continue
    
    #     # -------- accumulate gradient in FP32 -------------------------
    #     grad = None
    #     for j in range(num_perturbations):
    #         seed         = all_seeds[j] + idx
    #         coeff        = grad_estimate_list[j]       # (f⁺−f⁻)/2ε
    #         probe   = generate_perturbation(
    #                            param, coeff, distribution, seed)
            
    #         grad = probe if grad is None else grad + probe

    #     param.data.add_(grad, alpha= -1./float(num_perturbations)   )
    #     # -------- SGD step in FP32, then cast back -------------------
    #     # param_fp32 = param.data.float()
    #     # param_fp32.add_( -grad )                          # **NO extra ε/n**
    #     # param.data.copy_( param_fp32.to(param.dtype) )         # back to bf16
    # # SGD update (same as before)
    # param.data.add_(grad, alpha=-1. / float(num_perturbations))

    if cache_gradients:
        
        # for buf, param in zip(grad_buffer, model_params):
            # param.data.add_(buf) # buf already stores −∇̂θ / n
        for i, (g, param) in enumerate(zip(grad_buffer, model_params)):

            # Apply RMSProp and/or momentum 
            if args.beta1>0:
                momentum_i = coordinate_momentum[i]
                momentum_i.mul_(args.beta1).add_(g, alpha=1 - args.beta1)   # m_t
                m_hat = momentum_i                                
                # print("----- m_hat:")
                # print(m_hat[:20])
            else:
                m_hat = g
            
            if args.beta2>0:
                variances_i = coordinate_variance[i]
                variances_i.mul_(args.beta2).addcmul_(g, g, value=1 - args.beta2)  # v_t
                v_hat = variances_i.sqrt().add_(1e-8)            # ε for numerical stability
                # print("----- v_hat:")
                # print(v_hat[:20])
            else:
                v_hat = 1.0
                
            param.data.add_( m_hat / v_hat )


    

    else: # calculate this post from the seeds
        for idx, param in enumerate(model_params):
            # grad = None
            for j in range(num_perturbations):
                seed  = all_seeds[j] + idx
                coeff = grad_estimate_list[j]                 # (f⁺−f⁻)/2n # no ε bc ε == lr and they cancel out
        
                # ── NEW: accumulate probe chunk-by-chunk ─────────────────────
                for chunk in torch.split(param, CHUNK_SIZE, dim=0):     # 8k rows
                    probe = generate_perturbation(
                                chunk, coeff, distribution, seed)
                    chunk.add_(probe) 
        
                # grad = probe if grad is None else grad + probe


    if is_verbose:
        print(f"done updating the model {time.time() - start_time}")

    # Track metrics
    elapsed = time.time() - start_time
    loss = sum_of_losses/num_perturbations
    metrics['train_loss'].append(loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"CD-RGE: Optimization completed in {elapsed:.2f}s with loss {loss:.6f}")
    return metrics



def cdrge_no_chunking(model_params, loss_fn, lr, epsilon, iterations, num_perturbations=20,
                   distribution='rad', antithetic=False, use_adaptive_step=False, clip_grad_norm=0.0,
                   args=None):
    """Central Difference Random Gradient Estimation optimization

    A memory-efficient implementation that avoids parameter copies
    and uses seeded RNG for consistent perturbations.

    Args:
        model_params: List of model parameters to optimize
        loss_fn: Function that computes the loss
        lr: Learning rate
        epsilon: Perturbation scale (same as lr for numerical stability)
        iterations: Number of optimization iterations
        num_perturbations: Number of random perturbations to use
        distribution: Type of distribution for perturbations ('rad', 'normal', 'uniform')
        antithetic: Whether to use antithetic sampling (negative perturbation pairs)
        use_adaptive_step: Whether to use adaptive step sizes (default: False)
        clip_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        args: Program arguments with verbose flag

    Returns:
        Dictionary of metrics through training
    """
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"CDRGE: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"CDRGE: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Current loss at start of iteration
    if is_verbose:
        print("CDRGE: Computing initial loss...")
    
    # Store gradients for each parameter
    if is_verbose:
        print("CDRGE: Initializing gradients...")
    gradients = [torch.zeros_like(p.data) for p in model_params]


    all_seeds = [random.randint(0, 2**32 - 1) for i in range(num_perturbations)]
    grad_estimate_list = []
    sum_of_losses = 0
    if is_verbose:
        print("starting perts")
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = all_seeds[j]
        
        ##################################################################
        # Apply positive perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            # Generate and apply perturbation
            probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)
            param.data.add_(probe)

        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size

        ##################################################################
        # Apply negative perturbations using the seed
        ##################################################################
        for param_idx, param in enumerate(model_params):
            # Generate and apply perturbation
            probe = generate_perturbation(param, -2*epsilon, distribution, seed=pert_seed + param_idx)
            param.data.add_(probe)

        # Compute loss with negative perturbation - average across macro batches
        total_neg_loss = 0.0
        for _ in range(macro_batch_size):
            neg_loss_batch = loss_fn()
            batch_loss = neg_loss_batch.item() if hasattr(neg_loss_batch, 'item') else float(neg_loss_batch)
            total_neg_loss += batch_loss
            
        neg_loss = total_neg_loss / macro_batch_size
        
        # 2-sided estimation: (f(x+ε) - f(x-ε))/2 
        grad_estimate = (pos_loss - neg_loss) / (2) # no epsilon bc we mult later so it cancels out
        sum_of_losses += (pos_loss + neg_loss) / (2)
        grad_estimate_list.append(grad_estimate)
        
        for param_idx, param in enumerate(model_params):
            # Regenerate the same noise
            probe = generate_perturbation(param, epsilon, distribution, seed=pert_seed + param_idx)

            # Restore parameter by adding the perturbation
            param.data.add_(probe)
            

    for idx, param in enumerate(model_params):
        # if idx == 3:                      # skip states tensor
        #     continue
    
        # -------- accumulate gradient in FP32 -------------------------
        grad = None
        for j in range(num_perturbations):
            seed         = all_seeds[j] + idx
            coeff        = grad_estimate_list[j]       # (f⁺−f⁻)/2ε
            probe   = generate_perturbation(
                               param, coeff, distribution, seed)
            
            grad = probe if grad is None else grad + probe

        param.data.add_(grad, alpha= -1./float(num_perturbations)   )
        # -------- SGD step in FP32, then cast back -------------------
        # param_fp32 = param.data.float()
        # param_fp32.add_( -grad )                          # **NO extra ε/n**
        # param.data.copy_( param_fp32.to(param.dtype) )         # back to bf16


    if is_verbose:
        print(f"done updating the model {time.time() - start_time}")

    # Track metrics
    elapsed = time.time() - start_time
    loss = sum_of_losses/num_perturbations
    metrics['train_loss'].append(loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"CD-RGE: Optimization completed in {elapsed:.2f}s with loss {loss:.6f}")
    return metrics


def fdras_optimize(model_params, loss_fn, lr, epsilon, iterations, num_perturbations=20,
                   distribution='rad', antithetic=False, use_adaptive_step=False, clip_grad_norm=0.0,
                   args=None):
    """Finite Difference Random Adaptive Sampling (FDRAS) optimization

    A memory-efficient implementation that avoids parameter copies
    and uses seeded RNG for consistent perturbations.

    Args:
        model_params: List of model parameters to optimize
        loss_fn: Function that computes the loss
        lr: Learning rate
        epsilon: Perturbation scale (same as lr for numerical stability)
        iterations: Number of optimization iterations
        num_perturbations: Number of random perturbations to use
        distribution: Type of distribution for perturbations ('rad', 'normal', 'uniform')
        antithetic: Whether to use antithetic sampling (negative perturbation pairs)
        use_adaptive_step: Whether to use adaptive step sizes (default: False)
        clip_grad_norm: Maximum gradient norm for clipping (0 = no clipping)
        args: Program arguments with verbose flag

    Returns:
        Dictionary of metrics through training
    """
    # Add progress prints only if verbose mode is enabled
    is_verbose = args is not None and hasattr(args, 'verbose') and args.verbose
    if is_verbose:
        print(f"FDRAS: Starting optimization with {num_perturbations} perturbations")

    metrics = {
        'iterations': [],
        'train_loss': [],
        'elapsed_time': [],
        'grad_norm': []
    }

    device = model_params[0].device
    
    # Get macro_batch_size - default to 1 if not provided
    macro_batch_size = args.macro_batch_size if args is not None and hasattr(args, 'macro_batch_size') else 1
    if is_verbose and macro_batch_size > 1:
        print(f"FDRAS: Using macro batch size of {macro_batch_size}")

    start_time = time.time()

    # Current loss at start of iteration
    if is_verbose:
        print("FDRAS: Computing initial loss...")
    
    # Compute initial loss as average across all macro batches
    total_clean_loss = 0.0
    for _ in range(macro_batch_size):
        clean_loss_batch = loss_fn()  # Current model state loss
        batch_loss = clean_loss_batch.item() if hasattr(clean_loss_batch, 'item') else float(clean_loss_batch)
        total_clean_loss += batch_loss
        
    clean_loss = total_clean_loss / macro_batch_size
    
    if is_verbose:
        print(f"FDRAS: Initial loss: {clean_loss:.6f}")

    # Store gradients for each parameter
    if is_verbose:
        print("FDRAS: Initializing gradients...")
    gradients = [torch.zeros_like(p.data) for p in model_params]

    antithetic_mult = 1
    
    # Evaluate perturbations
    for j in range(num_perturbations):
        # Set a seed for this perturbation (to reproduce the same noise later)
        pert_seed = random.randint(0, 2**32 - 1)

        if antithetic and j%2==1:
            antithetic_mult = -1
            
        # Apply positive perturbations using the seed
        for param_idx, param in enumerate(model_params):
            # Generate and apply perturbation
            probe = generate_perturbation(param, antithetic_mult*epsilon, distribution, seed=pert_seed + param_idx)
            param.data.add_(probe)

        # Compute loss with positive perturbation - average across macro batches
        total_pos_loss = 0.0
        for _ in range(macro_batch_size):
            pos_loss_batch = loss_fn()
            batch_loss = pos_loss_batch.item() if hasattr(pos_loss_batch, 'item') else float(pos_loss_batch)
            total_pos_loss += batch_loss
            
        pos_loss = total_pos_loss / macro_batch_size
        
        # Single-sided estimation: (f(x+ε) - f(x))/ε
        grad_estimate = (pos_loss - clean_loss) / epsilon

        for param_idx, param in enumerate(model_params):
            # Reset seed to get the same perturbation for restoring

            # Regenerate the same noise
            probe = generate_perturbation(param, antithetic_mult, distribution, seed=pert_seed + param_idx)

            # Update gradient accumulation - avoid division by zero
            gradients[param_idx].add_(probe * grad_estimate)

            # Restore parameter by negating the perturbation
            param.data.add_(-probe*epsilon)

    # Average gradients across perturbations
    for grad in gradients:
        grad.div_(num_perturbations) # not actual_perturbs

    # Compute total gradient norm
    # total_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients]))

    # Use a default clip_norm if none provided
    # effective_clip_norm = 10.0 if clip_grad_norm <= 0 else clip_grad_norm

    # # Apply clipping if gradient norm is too large
    # if total_norm > effective_clip_norm:
    #     clip_coef = effective_clip_norm / (total_norm + 1e-8)
    #     for g in gradients:
    #         g.mul_(clip_coef)

    # Store gradient norm for metrics
    grad_norm = torch.norm(torch.stack([torch.norm(g.detach()) for g in gradients])).item()
    metrics['grad_norm'].append(grad_norm)

    # Default step size
    current_lr = lr

    # if use_adaptive_step:
    #     # Use adaptive step sizes only if loss doesn't decrease enough
    #     # Try different step sizes and keep the best
    #     loss_decreased = False

    #     # Standard step sizes to try
    #     step_sizes = [lr * 0.1, lr * 0.5, lr, lr * 2.0, lr * 5.0]

    #     best_step_loss = float('inf')
    #     best_step_lr = lr

    #     # Save current parameters - more efficient approach using list comprehension
    #     orig_params_values = [p.data.clone() for p in model_params]

    #     for step_size in step_sizes:
    #         # Apply gradient with this step size
    #         for param_idx, (param, grad) in enumerate(zip(model_params, gradients)):
    #             # Start from clean copy
    #             param.data.copy_(orig_params_values[param_idx])
    #             # Apply gradient (subtract because we want to minimize)
    #             param.data.add_(grad, alpha=-step_size)

    #         # Evaluate parameters
    #         step_loss_val = loss_fn()
    #         step_loss = step_loss_val.item() if hasattr(step_loss_val, 'item') else float(step_loss_val)

    #         # Loss decreased significantly
    #         if step_loss < (clean_loss - 0.001):
    #             loss_decreased = True

    #         # Keep track of best step size
    #         if step_loss < best_step_loss:
    #             best_step_loss = step_loss
    #             best_step_lr = step_size

    #     # If loss decreased with any step size, use that step size
    #     if loss_decreased:
    #         current_lr = best_step_lr

    #     # Restore original parameters before applying the best step size
    #     for param_idx, param in enumerate(model_params):
    #         param.data.copy_(orig_params_values[param_idx])

    # Apply gradient with the chosen step size (simple SGD)
    for param_idx, param in enumerate(model_params):
        param.data.add_(gradients[param_idx], alpha=-current_lr)

    # Check if loss improved with the update
    # new_loss_val = loss_fn()
    # new_loss = new_loss_val.item() if hasattr(new_loss_val, 'item') else float(new_loss_val)

    # Reverse out the update if it made us worse? Maybe we should just let it do its thing..
    # if new_loss > best_loss:
    #     # Restore best parameters if we've seen better ones
    #     print("NOT BETTER!")
    #     for param_idx, (param, grad) in enumerate(zip(model_params, gradients)):
    #         param.data.add_(grad, alpha=current_lr)

    # Track metrics
    elapsed = time.time() - start_time
    metrics['train_loss'].append(clean_loss)
    metrics['elapsed_time'].append(elapsed)


    # print(f"FDRAS: Optimization completed in {elapsed:.2f}s with loss {clean_loss:.6f}")

    # # Log progress
    # if iterations <= 10 or (iter_i + 1) % max(1, iterations // 10) == 0:
    #     print(f"Iter {iter_i+1}/{iterations}: Loss {clean_loss:.4f} (time: {elapsed:.2f}s, grad_norm: {grad_norm:.4f})")

    return metrics





import torch, time, random
from typing import List

# ────────────────────────────────────────────────────────────────────────────────
# Second‑Order SPSA (2SPSA) Solver
# ────────────────────────────────────────────────────────────────────────────────
def SPSA2(model_params: List[torch.Tensor],
          loss_fn,
          lr: float,
          epsilon: float,
          iterations: int,
          num_perturbations: int = 20,
          distribution: str = 'rad',
          antithetic: bool = False,        # kept for interface parity (ignored)
          use_adaptive_step: bool = False, # kept for interface parity (ignored)
          clip_grad_norm: float = 0.0,
          cache_gradients: bool = False,
          lambda_reg: float = 1.0,
          CHUNK_SIZE: int = 2 ** 15,
          args=None):
    """
    Second‑Order SPSA (2SPSA) Zeroth‑Order Optimizer

    Implements the Newton‑style update

        Δθ = (1 / 2n) Σ_i  [(f(θ+εv_i) - f(θ−εv_i)) v_i] /
                           { [(f(θ+εv_i+εu_i) − f(θ+εv_i−εu_i)
                              −f(θ−εv_i+εu_i) + f(θ−εv_i−εu_i))
                             (v_iᵀu_i) / (4 ε²)]  + λ_reg }

    using *n* independent probe pairs (v_i , u_i), each drawn from the
    Rademacher distribution.  Six forward passes are therefore required
    per probe:

        θ ± εv_i ,
        θ + εv_i ± εu_i ,
        θ − εv_i ± εu_i .

    All memory‑saving tricks from SPSA1/1.5 are retained: parameters are
    perturbed in‑place chunk‑by‑chunk, a seeded RNG reproduces the probes,
    and only per‑probe scalar coefficients are kept.
    """
    # ───── House‑keeping & logging ────────────────────────────────────────────
    is_verbose = args is not None and getattr(args, 'verbose', False)
    if is_verbose:
        print(f"2SPSA: Starting optimisation with {num_perturbations} perturbations")

    metrics = {'iterations': [], 'train_loss': [], 'elapsed_time': [], 'grad_norm': []}
    device   = model_params[0].device
    mb_size  = getattr(args, 'macro_batch_size', 1)

    # ───── Helpers ────────────────────────────────────────────────────────────
    OFFSET_U = 2 ** 20           # deterministic offset so u_seeds != v_seeds

    def _apply_noise(scale_v: float, scale_u: float,
                     seeds_v, seeds_u):
        """
        Add (scale_v * v  +  scale_u * u) to the parameters in‑place.
        Both scales are real numbers (can be ±epsilon, ±2*epsilon, etc.).
        """
        for p_idx, p in enumerate(model_params):
            for chunk in torch.split(p.data, CHUNK_SIZE, dim=0):
                if scale_v:
                    v_probe = generate_perturbation(
                        chunk, scale_v, distribution, seed=seeds_v[p_idx])
                    chunk.add_(v_probe)
                if scale_u:
                    u_probe = generate_perturbation(
                        chunk, scale_u, distribution, seed=seeds_u[p_idx])
                    chunk.add_(u_probe)

    def _eval_offset(scale_v: float, scale_u: float,
                     seeds_v, seeds_u) -> float:
        """
        Apply the requested offset, compute (macro‑)batched loss, then revert.
        Returns the averaged scalar loss value.
        """
        _apply_noise(scale_v, scale_u, seeds_v, seeds_u)

        tot = 0.0
        for _ in range(mb_size):
            l = loss_fn()
            tot += float(l.item() if hasattr(l, 'item') else l)

        _apply_noise(-scale_v, -scale_u, seeds_v, seeds_u)  # revert
        return tot / mb_size

    # ───── Pre‑compute "clean" loss ───────────────────────────────────────────
    clean_loss = sum(float(loss_fn().item()) for _ in range(mb_size)) / mb_size

    # ───── Storage for per‑probe scalar coefficients ‑‑ one per perturbation ‑─
    grad_coeffs: List[float] = []
    seeds_list = [random.randint(0, 2 ** 32 - 1) for _ in range(num_perturbations)]

    # ───── Begin optimisation ‑‑ one 2SPSA probe at a time ────────────────────
    t0 = time.time()
    for j in range(num_perturbations):

        base_seed = seeds_list[j]
        # Per‑parameter deterministic seeds for v and u
        seeds_v = [base_seed + idx          for idx in range(len(model_params))]
        seeds_u = [base_seed + OFFSET_U + idx for idx in range(len(model_params))]

        # ─── Step 1 & 2: θ ± εv_i (gradient numerator pieces) ───────────────
        pos_v = _eval_offset(+epsilon, 0.0, seeds_v, seeds_u)
        neg_v = _eval_offset(-epsilon, 0.0, seeds_v, seeds_u)

        # ─── Steps 3–6: Hessian‑like denominator pieces ─────────────────────
        pos_v_pos_u = _eval_offset(+epsilon, +epsilon, seeds_v, seeds_u)
        pos_v_neg_u = _eval_offset(+epsilon, -epsilon, seeds_v, seeds_u)
        neg_v_pos_u = _eval_offset(-epsilon, +epsilon, seeds_v, seeds_u)
        neg_v_neg_u = _eval_offset(-epsilon, -epsilon, seeds_v, seeds_u)

        # ─── Compute vᵀu for this probe (needs only signs, so scale 1) ──────
        # vTu = 0.0
        # for p_idx, p in enumerate(model_params):
        #     for chunk in torch.split(p.data, CHUNK_SIZE, dim=0):
        #         v_signs = generate_perturbation(chunk, 1.0, distribution,
        #                                         seed=seeds_v[p_idx])
        #         u_signs = generate_perturbation(chunk, 1.0, distribution,
        #                                         seed=seeds_u[p_idx])
        #         vTu += float((v_signs * u_signs).sum().item())

        # ─── Assemble numerator & denominator ‑‑ equation (C) ───────────────
        num  =  (pos_v - neg_v)                      # scalar
        # curv = ((pos_v_pos_u - pos_v_neg_u
        #         -neg_v_pos_u + neg_v_neg_u) * vTu) / (4.0 * epsilon ** 2)

        # denom = max(curv, lambda_reg)
        # coeff = - (num / denom) / (2.0 * num_perturbations)  # minus for descent

        raw_curv = ((pos_v_pos_u - pos_v_neg_u
            -neg_v_pos_u + neg_v_neg_u) ) / (4 * epsilon ** 2)

        #denom =  math.copysign(max(abs(raw_curv), lambda_reg), raw_curv)

        #denom = max( (raw_curv)**args.saturating_alpha, lambda_reg)

        if abs(raw_curv)<1.:
            print("raw_curve small")
            raw_curv = 1.
        denom = raw_curv
        
        coeff = -(pos_v - neg_v) / (2 * num_perturbations * denom)

        grad_coeffs.append(coeff)

        if True: #is_verbose:
            print(f"  pert: {j} "
                  f"num = {num:.8f}, "
                  f"coeff = {coeff:.8f}, "
                  f"curv = {raw_curv:.4f}, denom = {denom:.4f} "
                  f"pos_v={pos_v:.4f} neg_v={neg_v:.4f} "
                  f"pos_v_pos_u={pos_v_pos_u:.4f} pos_v_neg_u={pos_v_neg_u:.4f} "
                  f"neg_v_pos_u={neg_v_pos_u:.4f} neg_v_neg_u={neg_v_neg_u:.4f} " # vTu {vTu:.4f}
            )
            

    # ───── Apply the accumulated update (cached or regenerated) ──────────────
    if cache_gradients:
        grad_buffer = [torch.zeros_like(p.data) for p in model_params]

        for j, coeff in enumerate(grad_coeffs):
            seeds_v = [seeds_list[j] + idx for idx in range(len(model_params))]
            for p_idx, buf in enumerate(grad_buffer):
                for chunk_buf in torch.split(buf, CHUNK_SIZE, dim=0):
                    probe = generate_perturbation(chunk_buf, coeff, distribution,
                                                  seed=seeds_v[p_idx])
                    chunk_buf.add_(probe)

        for buf, p in zip(grad_buffer, model_params):
            p.data.add_(buf)

    else:
        for p_idx, p in enumerate(model_params):
            for j, coeff in enumerate(grad_coeffs):
                seed = seeds_list[j] + p_idx
                for chunk in torch.split(p.data, CHUNK_SIZE, dim=0):
                    probe = generate_perturbation(chunk, coeff, distribution, seed)
                    chunk.add_(probe)

    # ───── Final logging & metrics ───────────────────────────────────────────
    elapsed = time.time() - t0
    metrics['train_loss'].append(clean_loss)
    metrics['elapsed_time'].append(elapsed)

    if is_verbose:
        print(f"2SPSA: Completed in {elapsed:.2f}s ‑ clean loss {clean_loss:.6f}")

    return metrics







def train_step(model, args, batch_np, loss_closure, optimizer=None):
    """
    One optimisation step, agnostic to LSTM/DNC and to Zeroth-/First-order.
    `loss_closure()` must compute loss **using the CURRENT parameters**.
    """
    if args.solver=="BPTT":                   # ---------- first-order -----------
        if optimizer is None:
            raise ValueError("optimizer must be provided when using sovler==bptt")

        for p in model.parameters():
            p.requires_grad_(True)

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(args.macro_batch_size):
            loss = loss_closure()
            (loss / args.macro_batch_size).backward()
            total_loss += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        return {"train_loss": [total_loss], "grad_norm": [0.0]}

        
    elif args.solver=="1SPSA":
        # ------------------- zeroth-order (CD-RGE / FDRAS) --------------------
        with torch.no_grad():
            # can also replace this with cdrge_optimize, cdrge_no_chunking, or fdras_optimized depending on what you want to test
             
            return cdrge_optimize( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate_and_epsilon,
                epsilon           = args.learning_rate_and_epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = args.cache_gradients,
                args              = args,
            )
            
    elif args.solver=="1.5-SPSA":
        # ------------------- zeroth-order (CD-RGE / FDRAS) --------------------
        with torch.no_grad():
            return SPSA1_5( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate_and_epsilon,
                epsilon           = args.learning_rate_and_epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = args.cache_gradients,
                args              = args,
            )
            
             
            
    elif args.solver=="2SPSA":
        with torch.no_grad():
            return SPSA2( 
                model_params      = list(model.parameters()),
                loss_fn           = loss_closure,
                lr                = args.learning_rate_and_epsilon,
                epsilon           = args.learning_rate_and_epsilon,
                iterations        = 1,
                num_perturbations = args.num_perturbations,
                distribution      = args.distribution,
                antithetic        = args.antithetic_sampling,
                use_adaptive_step = False,
                clip_grad_norm    = args.grad_clip,
                cache_gradients   = args.cache_gradients,
                args              = args,
            )
    else:
        raise Exception(f"no solver implemented named {args.solver}")




def evaluate(model, tok, args, split="val", show_predictions=True, num_samples=3):
    with torch.no_grad():
        batch_np = get_examples_for_task(
            args.task, tok, args.micro_batch_size, args.seq_length,
            split=split, max_num=args.max_num
        )
        ids  = torch.as_tensor(batch_np, device=args.device, dtype=torch.long)
        xemb = model.embed(ids)
        logits, _, _ = model(xemb, require_gradients=False)

        loss     = compute_task_loss(logits, batch_np, tok, args.task)
        accuracy = compute_task_accuracy(logits, batch_np, tok, args.task)

        if show_predictions and args.task in ["copy","repeat_copy","sort","reverse","add"]:
            preds = torch.argmax(logits, dim=-1)

            if args.tokenizer == "hf_tiktoken":
                decode_fn = lambda ids: tok.decode(ids.tolist())
                decoded_preds = [decode_fn(seq) for seq in preds]
                decoded_inputs = [decode_fn(seq) for seq in ids]
                decoded_targets = [decode_fn(seq) for seq in torch.as_tensor(batch_np)]
            else:
                id_to_char = tok.id_to_char if hasattr(tok, "id_to_char") else tok.id_to_token
                decode_fn = lambda ids: "".join([id_to_char.get(i.item(), '') for i in ids])
                decoded_preds = [decode_fn(seq) for seq in preds]
                decoded_inputs = [decode_fn(seq) for seq in ids]
                decoded_targets = [decode_fn(seq) for seq in torch.as_tensor(batch_np)]

            print("=" * 40)
            print(f"Validation predictions ({split} split):")
            for i in range(min(len(decoded_preds), num_samples)):
                print(f"[Sample {i}] Input:     '{decoded_inputs[i]}'")
                print(f"[Sample {i}] Target:    '{decoded_targets[i]}'")
                print(f"[Sample {i}] Predicted: '{decoded_preds[i]}'")
            print("=" * 40)

        return {"loss": loss, "accuracy": accuracy}




def train(args):
    """Main training function"""
    # Set device
    device = torch.device(args.device)

    # Initialize wandb for experiment tracking (with defaults to avoid attribute errors)
    wandb_enabled = False

    # Check for existence of wandb attributes
    has_wandb_proj = hasattr(args, 'wandb_proj') and args.wandb_proj is not None
    has_api_key = hasattr(args, 'WANDB_API_KEY')
    has_run_name = hasattr(args, 'wandb_run_name')

    # Only enable wandb if all required attributes exist and conditions are met
    if WANDB_AVAILABLE and has_wandb_proj and args.wandb:
        # Set API key if available
        if has_api_key:
            os.environ["WANDB_API_KEY"] = args.WANDB_API_KEY

        # Determine run name
        if not has_run_name or args.wandb_run_name is None:
            run_name = f"{args.task}_{args.hidden_size}_{args.distribution}_{time.strftime('%Y%m%d_%H%M%S')}"
        else:
            run_name = args.wandb_run_name

        
        # Initialize wandb
        wandb.init(
            project=args.wandb_proj,
            name=run_name,
            config=vars(args),
            reinit=True
        )
        wandb_enabled = True
        
        print(f"Weights & Biases logging enabled. Project: {args.wandb_proj}, Run: {run_name}")
    else:
        # Don't show warnings for missing attributes in test mode
        print("No Weights & Biases logging enabled.")
    
    # Initialize tokenizer
    if args.tokenizer == "char_level" or not TIKTOKEN_AVAILABLE:
        if not TIKTOKEN_AVAILABLE and args.tokenizer == "hf_tiktoken":
            print("Warning: tiktoken is not available. Falling back to char_level tokenizer.")

        # Use CharTokenizer for sort/copy/reverse (letter-based tasks)
        # And NumericTokenizer only for add (number-based task)
        if args.task == "add":
            tok = NumericTokenizer(max_num=args.max_num)
        else:
            tok = CharTokenizer()
    else:  # hf_tiktoken
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            class TiktokenWrapper:
                def __init__(self, enc):
                    self.enc = enc
                    self.vocab_size = enc.n_vocab

                def encode(self, text):
                    return self.enc.encode(text)

                def decode(self, tokens):
                    return self.enc.decode(tokens)

            tok = TiktokenWrapper(enc)
        except Exception as e:
            print(f"Failed to load tiktoken: {str(e)}. Falling back to char_level tokenizer.")
            if args.task in ["add", "sort", "reverse"]:
                tok = NumericTokenizer(max_num=args.max_num)
            else:
                tok = CharTokenizer()
    
    dtype = torch.float32 if args.solver=="BPTT" else torch.float16 # bptt in fp16 is unstable... try it but be weary.. vanishing/exploding grads await you 

    # ───────── Model construction (LSTM or DNC) ─────────
    embed = nn.Embedding(tok.vocab_size, args.input_size,
                         device=device, dtype=dtype)
    
    if args.model_type.upper() == "LSTM":
        model = LSTM(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    elif args.model_type.upper() == "DNC":
        model = DNC(
            input_size   = args.input_size,
            output_size  = tok.vocab_size,
            hidden_size  = args.hidden_size,
            memory_size  = args.memory_size,
            head_size    = args.head_size,
            num_heads    = args.num_heads,
            embed        = embed,
            device       = device,
            dtype        = dtype,
        )
    else:
        raise ValueError(f"Unknown model_type {args.model_type}")
    
    model_params = list(model.parameters())          # <- used by zeroth-order
    param_count  = sum(p.numel() for p in model_params)
    bytes_per_param = 4 if args.solver=="BPTT" else 2
    param_memory_mb = param_count * bytes_per_param / (1024*1024)
    
    print(f"  Parameters: {param_count:,} ({param_memory_mb:.2f} MB)")

    
    
    
    # Determine bytes per parameter based on optimizer type
    if args.solver=="BPTT":
        bytes_per_param = 4  # float32 = 4 bytes
    else:
        bytes_per_param = 2  # bfloat16 = 2 bytes
        
    
    # Set up results dictionary
    results = {
        'args': vars(args),
        'train_metrics': {
            'loss': [],
            'accuracy': [],
            'iterations': [],
            'time': []
        },
        'val_metrics': {
            'loss': [],
            'accuracy': [],
            'iterations': []
        }
    }
    
    # Main training loop
    print(f"Starting training for task: {args.task}")
    print(f"Model config: hidden_size={args.hidden_size}, heads={args.num_heads}, dim_per_head={args.head_size}")
    print(f"Optimization: learning_rate_and_epsilon={args.learning_rate_and_epsilon}, perturbations={args.num_perturbations}, distribution={args.distribution}")
    print("="*50)
    print(args)
    print("="*50)

    total_start_time = time.time()

    has_overfit_flag = hasattr(args, 'overfit_to_one_batch_flag') and args.overfit_to_one_batch_flag

    # Generate first batch to compute initial loss
    
    init_batch = get_examples_for_task(
        args.task, tok, args.micro_batch_size, args.seq_length,
        split='train', max_num=args.max_num
    )
    
    if has_overfit_flag:
        print("="*50)
        print("OVERFITTING TO A SINGLE BATCH!")
        

        train_batch = init_batch 
        # sep_token = tokenizer.token_to_id.get(" ", 0) 

    # ───────── Initial loss / accuracy BEFORE training ─────────
    with torch.no_grad():
        ids_init   = torch.as_tensor(init_batch, device=device, dtype=torch.long)
        x_emb_init = model.embed(ids_init)               # shared embed layer
        init_logits, _, _ = model(x_emb_init, require_gradients=False)            # works for LSTM & DNC
    
        initial_loss     = compute_task_loss(init_logits, init_batch, tok, args.task)
        initial_accuracy = compute_task_accuracy(init_logits, init_batch, tok, args.task)


    print(f"Initial loss (before training): {initial_loss:.4f}, Accuracy: {initial_accuracy:.4f}")

    # Log initial metrics to wandb
    if wandb_enabled:
        wandb.log({
            "initial_loss": initial_loss,
            "initial_accuracy": initial_accuracy
        }, step=0)

    # Store initial metrics in results
    results['train_metrics']['loss'].append(initial_loss)
    results['train_metrics']['accuracy'].append(initial_accuracy)
    results['train_metrics']['iterations'].append(0)
    results['train_metrics']['time'].append(0)
    status = "training" 

    require_gradients = False
    if args.solver == "BPTT":
        require_gradients = True
        if args.use_adam:
            # Set up optimizer with memory-efficient settings
            # Use AdamW with lower memory footprint than standard Adam
            print("Using AdamW")
            optimizer = torch.optim.AdamW(
                model_params, 
                lr=args.learning_rate_and_epsilon,
                betas=(0.99, 0.999),
                eps=1e-8,  # Slightly larger epsilon for stability
                weight_decay=0.0,  
                amsgrad=False  # Disable amsgrad to save memory
            )
        else:
            print("Using vanilla SGD")
            optimizer = torch.optim.SGD(
                model_params, 
                lr=args.learning_rate_and_epsilon,
                momentum=0.0, 
                weight_decay=0.0
            )
    else:
        optimizer = None

    if args.hidden_size<4096:
        args.cache_gradients = True # THIS WILL COST US VRAM, but speed us up at the end, if we have the space we should use it.
    else:
        args.cache_gradients = False # THIS WILL SAVE VRAM, but slow us down a bit at the end. 

    total_iterations = 0
    for iteration in range(args.max_iterations):
        # Get batch based on task (or reuse same batch if overfitting)
        start_time = time.time()
        if not has_overfit_flag:
            train_batch = get_examples_for_task(
                args.task, tok, args.micro_batch_size, args.seq_length,
                split='train', max_num=args.max_num
            )
                
        # Define loss function for current batch
        def compute_loss(return_acc=False):
            ids  = torch.as_tensor(train_batch, device=device, dtype=torch.long)
            xemb = model.embed(ids)                     # [B,T,E]
            logits, _, _ = model(xemb, require_gradients=require_gradients)                  # same for LSTM & DNC
        
            loss = compute_task_loss(logits, train_batch, tok, args.task)
            if return_acc:
                acc = compute_task_accuracy(logits, train_batch, tok, args.task)
                return loss, acc
            return loss
    
        
        step_metrics = train_step(model, args, train_batch, compute_loss, optimizer)
        train_accuracy = -1.0  # or calculate here if you want but loss is fine for me
        
        # Store metrics
        results['train_metrics']['loss'].append(step_metrics['train_loss'][-1])
        results['train_metrics']['accuracy'].append(train_accuracy) # placeholder for now... 
        results['train_metrics']['iterations'].append(iteration + 1)
        results['train_metrics']['time'].append(time.time() - start_time)
        
        # Calculate iteration time and other metrics
        iteration_time = time.time() - start_time
        total_elapsed = time.time() - total_start_time


        # Get actual GPU memory usage if available
        max_gpu_memory_mb = 0
        if torch.cuda.is_available():
            max_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            # print(f"Max GPU memory used: {max_gpu_memory_mb:.1f} MB")

        print(f"Iteration {iteration+1}/{args.max_iterations} | "
              f"Train Loss: {step_metrics['train_loss'][-1]:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | "
              f"iteration time: {iteration_time:.4f}")

        print(f"  Parameters: {param_count:,} ({param_memory_mb:.2f} MB)")

        if has_overfit_flag:
            if step_metrics['train_loss'][-1] < 0.1:
                print(f"SUCCESS FINISHED! Overfit in iterations = {iteration}")
                print(args)
                total_iterations = iteration
                status = "success"
                
                # time.sleep(100000)
                # return
                break
            elif step_metrics['train_loss'][-1] > 50. or math.isnan(step_metrics['train_loss'][-1]):
                print(f"FAILED DIVERGING!")
                print(args)
                total_iterations = iteration
                status = "diverged"
                # time.sleep(100000)
                # return
                break
        # Logging
        if (iteration + 1) % args.log_interval == 0 and not has_overfit_flag:
            with torch.no_grad():
                # Validation (every 10% of iterations or every 10 iterations for short runs)
                # eval_interval = max(1, min(10, args.max_iterations // 10))
                # During training, don't show predictions to save space
                val_metrics = evaluate(model, tok, args, split='val', show_predictions=True)


                # val_metrics = evaluate(model_params, tok, args, split='val', show_predictions=True)
                results['val_metrics']['loss'].append(val_metrics['loss'])
                results['val_metrics']['accuracy'].append(val_metrics['accuracy'])
                results['val_metrics']['iterations'].append(iteration + 1)

            print(f"Validation at iter {iteration+1}: "
                  f"Loss: {val_metrics['loss']:.4f} | "
                  f"Accuracy: {val_metrics['accuracy']:.4f}")

            print(f"  Parameters: {param_count:,} ({param_memory_mb:.2f} MB)")

            # Log metrics to wandb
            if wandb_enabled:
                # Basic metrics
                wandb_metrics = {
                    "train_loss": step_metrics['train_loss'][-1],
                    "train_acc": train_accuracy,
                    "lr": args.learning_rate_and_epsilon,
                    "iter_time_s": iteration_time,
                    "total_time_hours": total_elapsed / 3600.0,
                    "grad_norm": step_metrics.get('grad_norm', [0])[-1] if step_metrics.get('grad_norm') else 0,
                    "val_loss": val_metrics['loss'],
                    "val_accuracy": val_metrics['accuracy'],
                }
                wandb.log(wandb_metrics, step=iteration + 1)



    # final cleanup after training
    with torch.no_grad():
        ids_final   = torch.as_tensor(init_batch, device=device, dtype=torch.long)
        x_emb_final = model.embed(ids_final)
        final_logits, _, _ = model(x_emb_final, require_gradients=False)
        final_loss     = compute_task_loss(final_logits, init_batch, tok, args.task)
        final_accuracy = compute_task_accuracy(final_logits, init_batch, tok, args.task)
        
    results['train_metrics']['loss'].append(final_loss)
    results['train_metrics']['accuracy'].append(final_accuracy)
    results['train_metrics']['iterations'].append(total_iterations)
    results['train_metrics']['time'].append(0)
    results["status"] = status

    print(f"Final loss (after training): {final_loss:.4f}, Accuracy: {final_accuracy:.4f}")
    
    # Log final results
    total_time = time.time() - total_start_time
    print("\nTraining completed!")
    print(f"Loss reduction: {results['train_metrics']['loss'][0] - results['train_metrics']['loss'][-1]:.4f}")
    print(f"Total training time: {total_time:.2f} seconds")

    print(" DONE! Now go get the actual vram")
    time.sleep(10)
    # --- drop strong Python refs --------------------------------------------
    if 'model_params' in locals():
        del model_params           # list of tensors
    if 'optimizer' in locals() and optimizer is not None:
        del optimizer
    
    # --- force Python to release any lingering tensors -----------------------
    import gc
    gc.collect()                   # releases refs held only by GC
    
    # --- tell the CUDA caching allocator to give memory back to the pool ----
    if torch.cuda.is_available():
        torch.cuda.empty_cache()   # frees cached blocks
        torch.cuda.ipc_collect()   # optional: reclaims inter-process buffers

    print("="*50)
    print("all results")
    print(results)
    print("="*50)
    
    return results
    

import argparse, time, math, traceback
import torch
import argparse, time, math, torch


def run_unittest(args):
    """Run a battery of quick sanity tests on `train()`."""
    
    batch_size = args.micro_batch_size
     

    # ---------------------------------------------------------------------- #
    #  All experiment configurations                                         #
    # ---------------------------------------------------------------------- #
    all_runs = {

            ### DNC TESTS
            # ========================= DNC BPTT ===================================== #
            # "tiny-bptt-default-dnc":   dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, memory_size=128, model_type="DNC",
            #                             solver="BPTT", use_adam=False, learning_rate_and_epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Tiny dnc with BPTT"),
            # "small-bptt-default-dnc":  dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, model_type="DNC",
            #                            solver="BPTT", use_adam=False, learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Small dnc with BPTT "),
            # "medium-bptt-default-dnc": dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, model_type="DNC",
            #                             solver="BPTT", use_adam=False, learning_rate_and_epsilon=0.0001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Medium dnc with BPTT "),
            # "medium-bptt-default-dnc": dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, model_type="DNC",
            #                                 solver="BPTT", use_adam=True, learning_rate_and_epsilon=0.0001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                 description="Medium dnc with BPTT with Adam"),
            
            # ========================= DNC cdrge-96 ================================= #
            "tiny-cdrge96-default-dnc":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, memory_size=128, model_type="DNC",
                                          learning_rate_and_epsilon=0.01,   #.01 for 1SPSA,  
                                              micro_batch_size=int(batch_size/1), macro_batch_size=1,
                                          num_perturbations=96,  antithetic=False,
                                          description="Tiny dnc with cdrge@96"),
            # "small-cdrge96-default-dnc": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, memory_size=128, model_type="DNC",
            #                               learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Small dnc with cdrge@96"),
            # "medium-cdrge96-default-dnc":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, memory_size=128, model_type="DNC",
            #                               learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Medium dnc with cdrge@96"),
            # "large-cdrge96-default-dnc": dict(model_size="large",  hidden_size=2**13, num_heads=1,  head_size=0, memory_size=2**13, model_type="DNC",
            #                               learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Large dnc with cdrge@96"),
            # 2.4B  with  hidden_size=2**14, num_heads=2**8, head_size=0,
            # "xlarge-cdrge96-default-dnc":dict(model_size="xlarge", hidden_size=2**14, num_heads=2**8, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="XLarge dnc with cdrge@96"),
            # 10B  with  hidden_size=2**15, num_heads=2**9, head_size=0,
            # "xxlarge-cdrge96-default-dnc":dict(model_size="xxlarge", hidden_size=2**15, num_heads=2**10, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate_and_epsilon=0.0001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="XXLarge dnc with cdrge@96"),
        
            # ========================= DNC cdrge-512 ================================ #
            "tiny-cdrge512-default-dnc":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, memory_size=128, model_type="DNC",
                                           learning_rate_and_epsilon=0.01,   #.1 for 1SPSA,  
                                               micro_batch_size=int(batch_size/1), macro_batch_size=1,
                                           num_perturbations=512, antithetic=False,
                                           description="Tiny dnc with cdrge@512"),
            # "small-cdrge512-default-dnc": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, memory_size=128, model_type="DNC",
            #                                learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Small dnc with cdrge@512"),
            # "medium-cdrge512-default-dnc":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150, memory_size=128, model_type="DNC",
            #                                learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Medium dnc with cdrge@512"),
            # "large-cdrge512-default-dnc": dict(model_size="large",  hidden_size=66000, num_heads=220,  head_size=300, memory_size=128, model_type="DNC",
            #                                learning_rate_and_epsilon=0.001, micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Large dnc with cdrge@512"),
            # 2.4B  with  hidden_size=2**14, num_heads=2**8, head_size=0,
            # "xlarge-cdrge512-default-dnc":dict(model_size="xlarge", hidden_size=2**14, num_heads=2**8, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=512,  antithetic=False,
            #                               description="XLarge dnc with cdrge@96"),
            # 10B  with  hidden_size=2**15, num_heads=2**9, head_size=0,
            # "xxlarge-cdrge96-default-dnc":dict(model_size="xxlarge", hidden_size=2**15, num_heads=2**10, head_size=0, memory_size=128, model_type="DNC",
            #                               learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=512,  antithetic=False,
            #                               description="XXLarge dnc with cdrge@512"),

            ### LSTM TESTS
            # ========================= LSTM BPTT ===================================== #
            
            "tiny-bptt-default-lstm":   dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20, model_type="LSTM",
                                        solver="BPTT", use_adam=True, learning_rate_and_epsilon=0.01,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
                                        description="Tiny lstm with BPTT"),
            # "small-bptt-default-lstm":  dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50, model_type="LSTM",
            #                             solver="BPTT", use_adam=True, learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                             description="Small lstm with BPTT"),
            # "medium-bptt-default-lstm": dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150,model_type="LSTM",
            #                             solver="BPTT", use_adam=True, learning_rate_and_epsilon=0.001,  micro_batch_size=int(batch_size/128), macro_batch_size=128,
            #                             description="Medium lstm with BPTT - optimized for GPU memory constraints"),
            
            # ========================= LSTM cdrge-96 ================================= #
            "tiny-cdrge96-default-lstm":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20,model_type="LSTM",
                                          learning_rate_and_epsilon=0.1,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
                                          num_perturbations=96,  antithetic=False,
                                          description="Tiny lstm with cdrge@96"),
            # "small-cdrge96-default-lstm": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50,model_type="LSTM",
            #                               learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Small lstm with cdrge@96"),
            # "medium-cdrge96-default-lstm":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150,model_type="LSTM",
            #                               learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Medium lstm with cdrge@96"),
            # "large-cdrge96-default-lstm": dict(model_size="large",  hidden_size=66000, num_heads=220,  head_size=300,model_type="LSTM",
            #                               learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="Large lstm with cdrge@96"),
            # "xlarge-cdrge96-default-lstm":dict(model_size="xlarge", hidden_size=297500, num_heads=350, head_size=850,model_type="LSTM",
            #                               learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                               num_perturbations=96,  antithetic=False,
            #                               description="XLarge lstm with cdrge@96"),
        
            # ========================= LSTM cdrge-512 ================================ #
            "tiny-cdrge512-default-lstm":  dict(model_size="tiny",   hidden_size=240,   num_heads=12,  head_size=20,model_type="LSTM",
                                           learning_rate_and_epsilon=0.1,   micro_batch_size=int(batch_size/1), macro_batch_size=1,
                                           num_perturbations=512, antithetic=False,
                                           description="Tiny lstm with cdrge@512"),
            # "small-cdrge512-default-lstm": dict(model_size="small",  hidden_size=1600,  num_heads=32,  head_size=50,model_type="LSTM",
            #                                learning_rate_and_epsilon=0.1,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Small lstm with cdrge@512"),
            # "medium-cdrge512-default-lstm":dict(model_size="medium", hidden_size=9600,  num_heads=64,  head_size=150,model_type="LSTM",
            #                                learning_rate_and_epsilon=0.01,  micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Medium lstm with cdrge@512"),
            # "large-cdrge512-default-lstm": dict(model_size="large",  hidden_size=66000, num_heads=220,  head_size=300,model_type="LSTM",
            #                                learning_rate_and_epsilon=0.01, micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="Large lstm with cdrge@512"),
            # "xlarge-cdrge512-default-lstm":dict(model_size="xlarge", hidden_size=297500, num_heads=350, head_size=850,model_type="LSTM",
            #                                learning_rate_and_epsilon=0.01, micro_batch_size=int(batch_size/1), macro_batch_size=1,
            #                                num_perturbations=512, antithetic=False,
            #                                description="XLarge lstm with cdrge@512"),


    }
    


    # ---------------------------------------------------------------------- #
    #  Main loop                                                             #
    # ---------------------------------------------------------------------- #
    failures = []
    for run_name, run_cfg in all_runs.items():
        print("="*50)
        print("="*50)
        print("="*50)
        print(f"\n=== {run_name}: {run_cfg['description']} ===")
    
        # Get a fresh copy of default CLI args
        run_args = vars(get_argument_parser())

        run_args["overfit_to_one_batch_flag"] = True
        
        # Override defaults with specific run config
        run_args.update(run_cfg)
    
        # Create final args object
        args = argparse.Namespace(**run_args)
        print(args)

        try:
            t0 = time.time()
            results = train(args)                     
            total_time = time.time() - t0
            avg_time   = total_time / args.max_iterations

            # ---------------- aggregate metrics -------------------------- #
            train_losses = results['train_metrics']['loss']
            val_losses   = results['val_metrics']['loss']

            train_delta = train_losses[0] - train_losses[-1]
            val_delta   = (val_losses[0] - val_losses[-1]) if len(val_losses) >= 2 else float('nan')

            print(f"Total time: {total_time:.2f}s | Avg/iter: {avg_time:.4f}s | "
                  # f"≈VRAM: {est_vram_gb:.2f} GB | "
                  f"ΔTrain-loss: {train_delta:+.4f} | ΔVal-loss: {val_delta:+.4f}")

            # ---------------- basic assertions --------------------------- #
            if train_delta <= 0:
                raise AssertionError("training loss did not improve")
            if not math.isnan(val_delta) and val_delta <= 0:
                raise AssertionError("validation loss did not improve")


            print(f"{run_name} ✅ passed!")
        except RuntimeError as e:
            print("="*50)
            print("="*50)
            print("="*50)
            failures.append((run_name, str(e)))
            print(f"❌  {run_name} FAILED: {e}")
            print("="*50)
            print("="*50)
            print("="*50)
            traceback.print_exc()
            # PyTorch 2.1+ raises a dedicated subclass
            oom = isinstance(e, torch.cuda.OutOfMemoryError) \
                  or "out of memory" in str(e).lower()
        
            if oom:
                raise RuntimeError(
                    "\nCUDA OOM detected.\n"
                    "  • First, try setting `args.cache_gradients = True` "
                    "or lowering `CHUNK_SIZE`.\n"
                    "  • If that’s not enough, reduce `micro_batch_size` and raise "
                    "`macro_batch_size` to keep the same total tokens.\n"
                    "  • As a last resort, choose a smaller model (hidden_size / heads).\n"
                ) from e
            else:
                # not an OOM → re-raise unchanged
                raise
            
            # return
            # Continue running the remaining tests instead of aborting immediately

    # ---------------------------------------------------------------------- #
    #  Summary                                                               #
    # ---------------------------------------------------------------------- #
    if failures:
        print("\n================ FAILED TESTS ================")
        for name, err in failures:
            print(f"{name}: {err}")
        raise RuntimeError(f"{len(failures)} test(s) failed.")
    else:
        print("\n✅  All unit tests passed!")





# ───────── Main ─────────
def get_argument_parser():
    """Create and return the argument parser"""
    parser = argparse.ArgumentParser(description="Train neural models on basic functions")

    # Task and model configuration
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="penn_tree_bank",
                        choices=["copy", "repeat_copy", "sort", "reverse", "add", "penn_tree_bank"])
    parser.add_argument("--model_type", type=str, default="LSTM",
                        choices=["LSTM", "DNC"])
    parser.add_argument("--seq_length", type=int, default=10,
                        help="For generation.")
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--macro_batch_size", type=int, default=1)
    parser.add_argument("--max_iterations", type=int, default=100)
    parser.add_argument("--max_num", type=int, default=9,
                        help="This is the max number in the domain to use in training for arithmetic tasks.")

    # Model architecture
    parser.add_argument("--input_size", type=int, default=100)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--memory_size", type=int, default=128)
    parser.add_argument("--head_size", type=int, default=128)

    # Optimization settings
    parser.add_argument("--solver", type=str, default="1SPSA",
                        choices=["BPTT", "1SPSA", "1.5-SPSA", "2SPSA"] )
    
    parser.add_argument("--distribution", type=str, default="rad",
                        choices=["rad", "normal", "uniform"])
    parser.add_argument("--learning_rate_and_epsilon", type=float, default=0.01)
    parser.add_argument("--saturating_alpha", type=float, default=1.0)
    parser.add_argument("--num_perturbations", type=int, default=20)
    parser.add_argument("--antithetic_sampling", action="store_true")
    parser.add_argument("--use_adaptive_step", action="store_true",
                        help="Use adaptive step sizes for FDRAS optimization")
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--beta1", type=float, default=0.0, help="for momentum")
    parser.add_argument("--beta2", type=float, default=0.0, help="for RMSProp-style variance")
    parser.add_argument("--use_adam", action="store_true", default=False,
                        help="Use Adam optimizer vs. vanilla SGD")
    parser.add_argument("--overfit_to_one_batch_flag", action="store_true", default=False,
                        help="Use the same batch for all training iterations")

    # Curriculum learning
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--curriculum_steps", type=int, default=5)


    # Experiment tracking
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_proj", type=str, default="neural-functions")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--verbose", action="store_true", help="Print detailed debugging information")

    # Tokenization
    parser.add_argument("--tokenizer", type=str, default="char_level",
                       choices=["char_level", "numeric", "hf_tiktoken"])
    # Unit Test
    parser.add_argument("--unit_test", action="store_true")

    # Logging
    parser.add_argument("--output_dir", type=str, default="./results_copy/")
    parser.add_argument("--oom_backoff_sec", type=int, default=600) # 10 min
    parser.add_argument("--oom_max_retries", type=int, default=1000)


     # Parse the arguments
    args = parser.parse_args()
    
    return args


# OLD, SIMPLE VERSION OF MAIN
# def main():
#     """Parse arguments and run training"""
#     args = get_argument_parser()

#     # Set random seed
#     if args.seed == -1:
#         args.seed = np.random.randint(1, 1001) 
#         print(f"Setting seed to {args.seed}")

#     set_seed(args.seed)

#     if args.unit_test:
#         success = run_unittest(args)
#     else:
#         # Run training
#         results = train(args)

#         return results


import argparse, hashlib          
import json, time, traceback, pathlib, torch, numpy as np
import argparse, hashlib, json, time, traceback, pathlib
import torch, numpy as np

# ───────── Main ─────────
def main() -> None:
    """Train with OOM-retry loop and write results + full args to JSON."""
    args = get_argument_parser()

    # seed --------------------------------------------------------------
    if args.seed == -1:
        args.seed = np.random.randint(1, 1001)
        print(f"[INFO] Setting seed to {args.seed}")
    set_seed(args.seed)

    # unit test ---------------------------------------------------------
    if args.unit_test:
        run_unittest(args)
        return

    # output dir --------------------------------------------------------
    out_root = pathlib.Path(args.output_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    seed_hash = hashlib.md5(str(args.seed).encode()).hexdigest()[:6]
    run_prefix = args.wandb_run_name or "run"
    base_name = f"{run_prefix}_{seed_hash}"          # everything else added later

    backoff_sec = args.oom_backoff_sec
    max_retries = args.oom_max_retries
    attempt = 0

    while True:
        try:
            results = train(args)

            status = results["status"]
            iters  = int(results["train_metrics"]["iterations"][-1])
            filename = f"{status}_{base_name}_{iters}.json"
            payload = {
                "status": status,
                "seed": int(args.seed),
                "iters": iters,
                "final_loss": float(results["train_metrics"]["loss"][-1]),
                "final_acc": float(results["train_metrics"]
                                   .get("accuracy", [0.0])[-1]),
                "wall_time_sec": float(results["train_metrics"]["time"][-1]),
                "wandb_run_name": run_prefix,
                "args": vars(args),
            }
            json.dump(payload, open(out_root / filename, "w"), indent=2)
            print(f"[INFO] Success → {filename} status {status} iters {iters}")
            return

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise
            attempt += 1
            if attempt > max_retries:
                filename = f"{base_name}_oom.json"
                json.dump({"status": "fail", "error": "OOM loop",
                           "args": vars(args)}, open(out_root / filename, "w"), indent=2)
                print(f"[FATAL] OOM loop → {filename}")
                raise
            traceback.print_exc()
            print(f"[OOM] attempt {attempt} — sleeping {backoff_sec//60} min")
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            time.sleep(backoff_sec)

        except Exception as e:
            filename = f"{base_name}_fail.json"
            json.dump({"status": "fail", "error": str(e),
                       "args": vars(args)}, open(out_root / filename, "w"), indent=2)
            print(f"[ERROR] Exception → {filename}")
            raise


if __name__ == "__main__":
    main()

