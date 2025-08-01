"""
Neural architectures used by the project.

* `LSTM`  – Flash-RNN accelerated where available (falls back to PyTorch).
* `DNC`   – Differentiable Neural Computer with memory operations.
"""
import math, warnings, torch
import torch.nn as nn
from typing import Tuple, Optional
import contextlib

# Flash-RNN is optional
try:
    from flashrnn.flashrnn import flashrnn     # pip install flash-rnn
    FLASH_OK = True
except ModuleNotFoundError:
    FLASH_OK = False
    warnings.warn("flashrnn not found – falling back to nn.LSTM", RuntimeWarning)



# --------------------------------
# FlashRNN LSTM
# --------------------------------
# • Constructor signature and attribute names identical to DNC.
# • All learnable tensors are registered as nn.Parameter so you can
#   iterate with `self.param_list = list(model.parameters())`.
# • `forward()` returns  (logits, memory, hidden)  where
#   `memory` is always None to keep the tuple structure.
class LSTM(nn.Module):
    # --------------------------------------------------------------- #
    # Constructor matches the DNC signature                           #
    # --------------------------------------------------------------- #
    def __init__(
        self,
        input_size:  int,
        output_size: int,
        hidden_size: int,
        memory_size: int,
        head_size:   int,
        num_heads:   int,
        embed:       nn.Embedding,
        device:      torch.device | None = None,
        dtype:       torch.dtype = torch.bfloat16
    ):
        super().__init__()

        # ------------- public attributes (keep DNC names) ---------- #
        self.device       = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size   = input_size
        self.output_size  = output_size
        self.hidden_size  = hidden_size # must be num_heads * head_size
        self.memory_size  = memory_size     # unused but preserved
        # self.head_size    = head_size #(THIS IS D TO MATCH FLASH PAPER)
        # self.num_heads    = num_heads       # (THIS IS N TO MATCH Flash paper)
        self.embed        = embed.to(self.device)

        # ------------- Flash-RNN gate sizes ------------------------ #
        self.G = 4                           # LSTM has 4 gates
        self.N = num_heads              # reuse same symbol as Flash docs
        self.D = head_size if head_size > 0 else hidden_size // self.N
        
        self.dtype = dtype

        if self.hidden_size % self.N != 0:
            raise ValueError(f"hidden_size={self.hidden_size} not cleanly divisible by num_heads={self.num_heads}, or just set head_size==0 and we will interpret what head_size is based on num_reads {self.num_heads} based on, i.e. { self.hidden_size // self.num_heads}.")
            

        # ------------- parameters expected by flashrnn ------------- #
        # Input projection (equivalent to W_emb in the reference snippet)
        self.W_in = nn.Parameter(
            torch.randn(input_size, self.G * self.N * self.D, dtype=dtype, device=self.device)
            / math.sqrt(input_size)
        )

        # Recurrent weights  R  [G, N, D, D]
        self.R = nn.Parameter(
            torch.randn(self.G, self.N, self.D, self.D, dtype=dtype, device=self.device)
            / math.sqrt(self.D)
        )

        # Biases  b  [G, N, D]
        self.b = nn.Parameter(torch.zeros(self.G, self.N, self.D, dtype=dtype, device=self.device))

        # Initial states  [S=2, B=1, 1, N, D]   («1» batch placeholder, expanded on forward)
        S = 2
        self.states0 = nn.Parameter(
            torch.zeros(S, 1, 1, self.N, self.D, dtype=dtype, device=self.device),
            requires_grad=False,   # treat initial state as non-trainable (same as snippet)
        )

        # Output projection  W_proj  &  bias
        self.W_proj = nn.Parameter(
            torch.randn(output_size, self.N * self.D, dtype=dtype, device=self.device)
            / math.sqrt(self.D)
        )
        self.b_proj = nn.Parameter(torch.zeros(output_size, dtype=dtype, device=self.device))

        # When FlashRNN is unavailable fall back to a tiny nn.LSTM + Linear
        if not FLASH_OK:
            self.fallback_lstm = nn.LSTM(
                input_size, hidden_size, batch_first=True, device=self.device, dtype=self.dtype
            )
            self.fallback_proj = nn.Linear(hidden_size, output_size, device=self.device, dtype=self.dtype)

    # --------------------------------------------------------------- #
    # Forward (same signature & tuple structure as DNC)               #
    # --------------------------------------------------------------- #
    def forward(
        self,
        x_emb:  torch.Tensor,                                 # [B, T, E]
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
        memory: torch.Tensor | None = None,                   # placeholder
        require_gradients: bool = False
    ):
        """
        Parameters
        ----------
        x_emb   : [B, T, input_size]  pre-embedded sequence
        hidden  : (h0, c0) with shapes  [1, B, H]
        memory  : ignored (kept to mirror DNC)
        require_gradients : bit flag to store gradients during the fpass or not

        Returns
        -------
        logits  : [B, T, output_size]
        memory  : None     (to keep call-site tuple)
        hidden  : (hN, cN)
        """
        

        with (torch.no_grad() if not require_gradients else contextlib.nullcontext()):
            B, T, E = x_emb.shape
            dtype   = x_emb.dtype
            dev     = x_emb.device
    
            # ---------------- FlashRNN fast path --------------------- #
            if FLASH_OK and dev.type == "cuda":
                # ---- 1. project input to gate dimensions (Wx) ---- #
                Wx = torch.einsum("bte,eg->btg", x_emb.to(self.dtype), self.W_in)   # [B,T,G*N*D]
                Wx = Wx.view(B, T, self.G, self.N, self.D).contiguous()
    
                # ---- 2. initial states ---- #
                if hidden is None:
                    # expand states0 to current batch
                    states0 = self.states0.expand(-1, B, -1, -1, -1).contiguous()
                else:
                    h0, c0 = hidden
                    states0 = torch.stack([
                        h0.transpose(0,1).reshape(B, 1, self.N, self.D),   # [B,1,N,D]
                        c0.transpose(0,1).reshape(B, 1, self.N, self.D),
                    ], dim=0)
    
                # ---- 3. run flashrnn ---- #
                states, _ = flashrnn(                 
                    Wx,                               # [B,T,G,N,D] contiguous
                    self.R,
                    self.b,
                    states=states0,                   # shape  [S,B,1,N,D]
                    function="lstm",
                    backend="cuda",  # or "cuda" or "cuda_fused" etc., cuda_fused is very fussy.. 
                )                                     # states[0] is h
                
                h_flat = states[0].reshape(B, T, self.N * self.D)
    
                # ---- 4. projection to vocab ---- #
                logits = torch.einsum("btn,vn->btv", h_flat, self.W_proj) + self.b_proj
    
                # reshape last hidden state to PyTorch format
                hN = h_flat[:, -1, :].unsqueeze(0)          # [1,B,H]
                cN = states[1].reshape(B, T, self.N * self.D)[:, -1, :].unsqueeze(0)
    
                return logits, None, (hN.to(dtype), cN.to(dtype))
    
            # ---------------- fallback path (CPU or missing flashrnn)------ #
            else:
                if hidden is None:
                    h0 = x_emb.new_zeros(1, B, self.hidden_size, dtype=dtype)
                    c0 = x_emb.new_zeros(1, B, self.hidden_size, dtype=dtype)
                else:
                    h0, c0 = hidden
    
                y, (hN, cN) = self.fallback_lstm(x_emb, (h0, c0))
                logits = self.fallback_proj(y)
                return logits, None, (hN, cN)




# --------------------------------------------------------------------------- #
#  Differentiable Neural Computer                                             #
# --------------------------------------------------------------------------- #
class DNC(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, memory_size, head_size, num_heads, embed, device=None, dtype=torch.float):
            super(DNC, self).__init__()
       
            # with torch.inference_mode():
            # Set the device for initialization
            self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Move the model to the specified device immediately
            self.to(self.device)
            
            self.input_size = input_size
            self.output_size = output_size  # This should be vocab_size
            self.hidden_size = hidden_size
            self.memory_size = memory_size
            self.head_size = head_size
            self.num_reads = num_heads
            self.embed = embed

            self.head_size = head_size if head_size > 0 else hidden_size // self.num_reads

            if self.hidden_size % self.num_reads != 0:
                raise ValueError(f"hidden_size={self.hidden_size} not cleanly divisible by num_heads={self.num_reads}, or just set head_size==0 and we will interpret what head_size is based on num_reads {self.num_reads} based on, i.e. { self.hidden_size // self.num_reads}.")

            controller_input_size = input_size + self.num_reads * self.head_size

            # ── Temporarily disable cuDNN if either dimension exceeds its limit ──
            if controller_input_size > 8192 or hidden_size > 8192:
                torch.backends.cudnn.enabled = False
    
            # Input normalisation
            self.input_norm = nn.LayerNorm(controller_input_size, device=self.device, dtype=dtype)
    
            # Controller LSTM (this is the op cuDNN would choke on)
            self.controller = nn.LSTM(
                controller_input_size,
                hidden_size,
                batch_first=True,
                device=self.device,
                dtype=dtype
            )
    
            self.controller_norm = nn.LayerNorm(hidden_size, device=self.device, dtype=dtype)
        
            # Memory operation layers with normalization
            self.fc_read_keys = nn.Linear(hidden_size, self.num_reads * self.head_size, device=self.device, dtype=dtype)
            self.fc_write_keys = nn.Linear(hidden_size, self.head_size, device=self.device, dtype=dtype)
            self.fc_write_strength = nn.Linear(hidden_size, 1, device=self.device, dtype=dtype)
            self.fc_erase_vector = nn.Linear(hidden_size, self.head_size, device=self.device, dtype=dtype)
            self.fc_add_vector = nn.Linear(hidden_size, self.head_size, device=self.device, dtype=dtype)
    
            self.read_keys_norm = nn.LayerNorm( self.head_size, device=self.device, dtype=dtype)
            self.write_keys_norm = nn.LayerNorm( self.head_size, device=self.device, dtype=dtype)
            self.memory_norm = nn.LayerNorm( self.head_size, device=self.device, dtype=dtype)
    
            # Output projection with normalization - project directly to vocab size
            total_output_size = hidden_size + self.num_reads * self.head_size
            self.pre_output_norm = nn.LayerNorm(total_output_size, device=self.device, dtype=dtype)
            self.fc_proj = nn.Linear(total_output_size, output_size, device=self.device, dtype=dtype) # Project directly to vocab size
    
            # Initialize parameters on GPU
            self.reset_parameters()
    
    
    
    
    def reset_parameters(self):
        """Initialize parameters with appropriate distributions"""
        # Initialize LSTM params
        for name, p in self.controller.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.constant_(p, 0)
    
        # Initialize memory operation layers (TIGHTER INIT)
        for name, p in self.named_parameters():
            if 'fc_' in name and 'weight' in name:
                # nn.init.uniform_(p, -0.1, 0.1)  # Original, potentially too wide
                nn.init.xavier_uniform_(p)  # Or use Xavier initialization
            elif 'fc_' in name and 'bias' in name:
                nn.init.constant_(p, 0) # GOOD: Keep bias at 0
                
    def _read_memory(self, memory, read_keys):
        """Read from memory using normalized attention."""
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        read_keys = self.read_keys_norm(read_keys.view(-1, self.head_size)).view(-1, self.num_reads, self.head_size)

        # Compute attention weights (no scaling, let LayerNorm handle it)
        read_weights = torch.softmax(
            torch.einsum('bnh,bmh->bnm', read_keys, memory_normalized),
            dim=2
        )
        read_vectors = torch.einsum('bnm,bmh->bnh', read_weights, memory)
        return read_vectors

    def _write_memory(self, memory, write_keys, write_str, erase_vec, write_vec, require_gradients=False):
        """Write to memory using normalized attention."""
        # Normalize memory and keys
        memory_normalized = self.memory_norm(memory)
        write_keys = self.write_keys_norm(write_keys)

        # Compute write weights
        write_weights = torch.softmax(
            torch.einsum('bh,bmh->bm', write_keys, memory_normalized),
            dim=1
        ).unsqueeze(1)  # [B, 1, memory_size]

        # Scale by write strength
        write_weights = write_weights * write_str.unsqueeze(1)

        # Erase and write operations
        erase = torch.einsum('bnm,bh->bmh', write_weights, erase_vec)
        write = torch.einsum('bnm,bh->bmh', write_weights, write_vec)
        
        # Update memory
        if require_gradients:
            memory = memory * (1 - erase) + write
        else:
            # do ops in place to conserve memory
            erase.neg_().add_(1)      # erase = 1 - erase, in-place
            memory.mul_(erase)        # memory *= (1 - erase)
            memory.add_(write)        # memory += write
            
        return memory

    def forward(self, x_emb, hidden=None, memory=None, require_gradients=False):
        
        with (torch.no_grad() if not require_gradients else contextlib.nullcontext()):
             
            B, L, E = x_emb.size()
            device = x_emb.device
    
            # Initialize states if needed
            if hidden is None:
                h0 = x_emb.new_zeros(1, B, self.hidden_size)
                c0 = x_emb.new_zeros(1, B, self.hidden_size)
                hidden = (h0, c0)
    
            if memory is None:
                memory = x_emb.new_zeros(B, self.memory_size, self.head_size)
    
            read_vec = x_emb.new_zeros(B, self.num_reads * self.head_size)
            outputs = []
    
            for t in range(L):
                # Normalize and combine input with read vector
                controller_input = torch.cat([x_emb[:, t, :], read_vec], dim=-1)
                controller_input = self.input_norm(controller_input)
                
                # Controller
                out_ctrl, hidden = self.controller(controller_input.unsqueeze(1), hidden)
                h = self.controller_norm(out_ctrl.squeeze(1))
    
                # Memory parameters
                read_keys = self.fc_read_keys(h).view(B, self.num_reads, self.head_size)
                write_keys = self.fc_write_keys(h)
                write_str = torch.sigmoid(self.fc_write_strength(h))
                erase_vec = torch.sigmoid(self.fc_erase_vector(h))
                write_vec = torch.tanh(self.fc_add_vector(h))
    
                # Memory operations
                memory = self._write_memory(memory, write_keys, write_str, erase_vec, write_vec, require_gradients=require_gradients)
                read_vectors = self._read_memory(memory, read_keys)
                read_vec = read_vectors.reshape(B, -1)
    
                # Output projection with normalization - project directly to logits
                output = torch.cat([h, read_vec], dim=-1)
                output = self.pre_output_norm(output)
                logits = self.fc_proj(output)  # Direct projection to vocab size
                outputs.append(logits.unsqueeze(1))
    
            outputs = torch.cat(outputs, dim=1)
            return outputs, memory, hidden








####################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
# import math, contextlib, torch
# import torch.nn as nn
# from typing import Optional, Tuple, List

# # Import off-the-shelf implementations
# try:
#     from mamba_ssm import Mamba as MambaBlock
#     # If available, import Mamba's optimized RMSNorm (falls back to None if not present)
#     try:
#         from mamba_ssm.ops.triton.layernorm import RMSNorm as MambaRMSNorm
#     except ImportError:
#         MambaRMSNorm = None
# except ImportError as e:
#     raise ImportError("Mamba-SSM library is not installed. Please install mamba-ssm to use the Mamba class.") from e

# try:
#     from s5 import S5 as SSMCore
# except ImportError as e:
#     raise ImportError("S5 library is not installed. Please install s5-pytorch to use the SSM class.") from e

# # ---------------------------------------------------------------------------
# # 1) Transformer (Encoder)
# # ---------------------------------------------------------------------------
# class Transformer(nn.Module):
#     """Transformer encoder using PyTorch's nn.TransformerEncoder."""
#     def __init__(
#         self,
#         input_size:  int,
#         output_size: int,
#         hidden_size: int,
#         memory_size: int,   # not used in Transformer
#         head_size:   int,
#         num_heads:   int,
#         embed:       nn.Embedding,
#         device:      torch.device | None = None,
#         dtype:       torch.dtype = torch.bfloat16,
#         n_layers:    int = 1
#     ):
#         super().__init__()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.dtype  = dtype

#         # Store interface attributes
#         self.input_size  = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.memory_size = memory_size  # not used
#         self.head_size   = head_size if head_size > 0 else hidden_size // num_heads
#         self.num_heads   = num_heads
#         self.embed       = embed.to(self.device)

#         if hidden_size % num_heads != 0:
#             raise ValueError("hidden_size must be divisible by num_heads")

#         # Input projection (if embedding size E != hidden size H)
#         self.in_proj = nn.Linear(input_size, hidden_size, device=self.device, dtype=dtype) \
#                        if input_size != hidden_size else nn.Identity()

#         # Fixed sinusoidal positional encoding (precomputed for max length)
#         max_len = 8192
#         pos = torch.arange(max_len, dtype=torch.float32, device=self.device).unsqueeze(1)
#         div = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32, device=self.device)
#                         * (-math.log(10000.0) / hidden_size))
#         pe = torch.zeros(max_len, hidden_size, dtype=dtype, device=self.device)
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer("pos_enc", pe, persistent=False)

#         # Transformer encoder layers (using PyTorch's built-in implementation)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_size, nhead=num_heads, dim_feedforward=4 * hidden_size,
#             dropout=0.0, batch_first=True, device=self.device, dtype=dtype
#         )
#         self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.out_proj = nn.Linear(hidden_size, output_size, device=self.device, dtype=dtype)

#     def forward(
#         self,
#         x_emb: torch.Tensor,                # [B, T, E] input embeddings
#         hidden: Optional[Tuple] = None,     # (unused for Transformer)
#         memory: Optional[torch.Tensor] = None,  # (unused for Transformer)
#         require_gradients: bool = False
#     ) -> Tuple[torch.Tensor, None, None]:
#         # The Transformer encoder doesn't use hidden or memory; we include them for interface compatibility.
#         with (torch.no_grad() if not require_gradients else contextlib.nullcontext()):
#             # Project to hidden size and add positional encoding
#             x = self.in_proj(x_emb.to(self.dtype))
#             x = x + self.pos_enc[: x.size(1)]
#             # Pass through Transformer encoder stack
#             y = self.encoder(x)
#             # Project to output vocabulary or feature size
#             logits = self.out_proj(y)
#             return logits, None, None

# # ---------------------------------------------------------------------------
# # 2) Mamba (State-Space Model with gating and convolution)
# # ---------------------------------------------------------------------------
# class Mamba(nn.Module):
#     """
#     Mamba sequence model using the official mamba-ssm implementation.
#     This wraps one or more Mamba blocks to fit the (input_size,..., embed, device, dtype) interface.
#     """
#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         hidden_size: int,
#         memory_size: int,  # not used (for interface consistency)
#         head_size: int,    # not used
#         num_heads: int,    # not used
#         embed: nn.Embedding,
#         device: torch.device | None = None,
#         dtype: torch.dtype = torch.bfloat16,
#         n_layers: int = 1,
#         state_size: int = 16,
#         conv_size: int = 4,
#         expand_factor: int = 2
#     ):
#         super().__init__()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.input_size  = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.memory_size = memory_size
#         self.head_size   = head_size
#         self.num_heads   = num_heads
#         self.embed       = embed.to(self.device)
#         self.dtype       = dtype

#         # Input projection if input embedding size != hidden size
#         self.input_proj = nn.Linear(input_size, hidden_size, device=self.device, dtype=dtype) \
#                           if input_size != hidden_size else nn.Identity()

#         # Stack of Mamba blocks from the library
#         self.layers = nn.ModuleList([
#             MambaBlock(
#                 d_model=hidden_size,
#                 d_state=state_size,
#                 d_conv=conv_size,
#                 expand=expand_factor
#             ).to(self.device) for _ in range(n_layers)
#         ])

#         # Use Mamba's RMSNorm if available, otherwise default to LayerNorm
#         if 'MambaRMSNorm' in globals() and MambaRMSNorm is not None:
#             self.norm = MambaRMSNorm(hidden_size, eps=1e-5, device=self.device, dtype=dtype)
#         else:
#             self.norm = nn.LayerNorm(hidden_size, eps=1e-5, device=self.device, dtype=dtype)

#         self.output_proj = nn.Linear(hidden_size, output_size, device=self.device, dtype=dtype)

#     def forward(
#         self,
#         x_emb: torch.Tensor, 
#         hidden: Optional[List] = None,   # list of per-layer states (not explicitly used in this wrapper)
#         memory: Optional[torch.Tensor] = None,  # not used
#         require_gradients: bool = False
#     ) -> Tuple[torch.Tensor, None, Optional[List]]:
#         """
#         x_emb: [B, T, input_size] input sequence embeddings.
#         hidden: (optional) list of hidden states for each layer (each state is typically a tuple for conv/SSM states).
#                 Not required for full-sequence processing; the Mamba blocks handle state internally.
#         """
#         with (torch.no_grad() if not require_gradients else contextlib.nullcontext()):
#             x = self.input_proj(x_emb.to(self.dtype))
#             for layer in self.layers:
#                 # Each Mamba block processes the sequence; we are not explicitly passing or retrieving state here.
#                 x = layer(x)
#             x = self.norm(x)
#             logits = self.output_proj(x)
#             # We do not expose Mamba internal states in this simple wrapper (hidden returns as None).
#             return logits, None, None

# # ---------------------------------------------------------------------------
# # 3) SSM (Linear State-Space Model)
# # ---------------------------------------------------------------------------
# class SSM(nn.Module):
#     """Linear State-Space Model using the S5 implementation."""
#     def __init__(
#         self,
#         input_size: int,
#         output_size: int,
#         hidden_size: int,
#         memory_size: int,  # not used
#         head_size: int,    # not used
#         num_heads: int,    # not used
#         embed: nn.Embedding,
#         device: torch.device | None = None,
#         dtype: torch.dtype = torch.bfloat16,
#         state_size: int = 16    # (SSM internal state size, not directly used by S5 in this wrapper)
#     ):
#         super().__init__()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.input_size  = input_size
#         self.output_size = output_size
#         self.hidden_size = hidden_size
#         self.memory_size = memory_size
#         self.head_size   = head_size
#         self.num_heads   = num_heads
#         self.embed       = embed.to(self.device)
#         self.dtype       = dtype

#         # Linear layers for input and output
#         self.input_proj = nn.Linear(input_size, hidden_size, device=self.device, dtype=dtype) \
#                           if input_size != hidden_size else nn.Identity()
#         self.output_proj = nn.Linear(hidden_size, output_size, device=self.device, dtype=dtype)

#         # Normalization before feeding into SSM (using LayerNorm with small epsilon)
#         self.norm = nn.LayerNorm(input_size, eps=1e-6, device=self.device, dtype=dtype)

#         # SSM core (S5 layer)
#         # Note: The S5 constructor internally decides the state dimension. The state_size parameter here is kept for interface,
#         # but we do not explicitly pass it to S5 (one could modify S5 initialization if needed).
#         self.ssm = SSMCore(hidden_size, hidden_size).to(self.device)

#     def forward(
#         self,
#         x_emb: torch.Tensor,
#         hidden: Optional[torch.Tensor] = None,   # optional initial state for SSM (shape depends on S5 internals)
#         memory: Optional[torch.Tensor] = None,   # not used
#         require_gradients: bool = False
#     ) -> Tuple[torch.Tensor, None, Optional[torch.Tensor]]:
#         """
#         x_emb: [B, T, input_size] input sequence.
#         hidden: optional tensor representing the previous state of the SSM (for continuing sequences).
#         Returns:
#             logits: [B, T, output_size] output sequence after the SSM and projection.
#             memory: None (not used in this model).
#             new_hidden: the new state tensor after processing this sequence (can be fed as `hidden` next call).
#         """
#         with (torch.no_grad() if not require_gradients else contextlib.nullcontext()):
#             # Normalize and project input to hidden size
#             x = self.norm(x_emb.to(self.dtype))
#             x = self.input_proj(x)
#             # Pass through the SSM layer (with initial state if provided)
#             if hidden is not None:
#                 result = self.ssm(x, hidden)
#             else:
#                 result = self.ssm(x)
#             # The S5 layer may return (output, final_state) or just output
#             if isinstance(result, tuple):
#                 y, new_hidden = result
#             else:
#                 y, new_hidden = result, None
#             logits = self.output_proj(y)
#             return logits, None, new_hidden
