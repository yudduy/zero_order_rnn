# README FOR NEURIPS SUBMISSION  
**Scaling Recurrent Neural Networks to a Billion Parameters with Zero-Order Optimization**  

This README explains **exactly** how to reproduce every experiment reported in the paper.  
All experiments were run on **A40 GPUs** (46 GB) obtained from [RunPod](https://www.runpod.io/) (up to 10 × A40 per node, ≈ \$4 / hr).  

---

## 0. Quick-start checklist ⚡️
1. **Provision** an 8-10× A40 instance on RunPod (or equivalent).  
2. **SSH** into the node and follow the **Setup** section below.  
3. Run the **Smoke Test** to confirm everything works.  
4. Follow the instructions below to reproduce Tables 1–2 and Figure 1.  

---

## 1. Setup 

### 1.1. System packages
```bash
sudo apt update -y
sudo apt install -y screen vim micro nano unzip
```

### 1.2. Python environment
```bash
# (Optionally) create a fresh conda or venv before this step
pip install --upgrade pip
pip install transformers datasets timeout_decorator wandb matplotlib pynvml neptune
pip install --upgrade "filelock>=3.12"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 1.3. Clone and install **FlashRNN**
```bash
git clone https://github.com/NX-AI/flashrnn.git
cd flashrnn
pip install -e .
cd ..
```

### 1.4. Unpack the NeurIPS submission
```bash
unzip submission.zip
cd cdrge_experiments      # all project scripts live here
chmod +x *.sh      # ensure helper scripts are executable
```

---

## 2. Smoke Test 🔥

To ensure everything is working, lets train an 100m parameter LSTM across 8 GPUs to verify NCCL and Torch-distributed are working properly:

```bash
# Port 29500 is the PyTorch default; change if it is already in use.
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py
# or, explicitly:
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 distributed_rge.py
```

**If it hangs on systems *without* InfiniBand:**
```bash
export NCCL_P2P_DISABLE=1   # disables peer-to-peer paths
# then re-run the smoke test
```

The first run will require download of OWT from HF. Then you should expect to see:
### Expected Smoke-Test Output
```text
==================================================
[Train] Iteration 0, train_loss = 7.537032127380371, loss_ema_fast = 7.537032127380371, loss_ema_slow = 7.537032127380371, lr = 0.01, val_loss = 10000.0, current_max_seq_len = 10 time per step (TOTAL) = 0:00:17.729556
==================================================
[Train] Iteration 1, train_loss = 3.640536069869995, loss_ema_fast = 7.537032127380371, loss_ema_slow = 7.537032127380371, lr = 0.01, val_loss = 10000.0, current_max_seq_len = 10 time per step (TOTAL) = 0:00:00.443671
==================================================
[Train] Iteration 2, train_loss = 1.754884958267212, loss_ema_fast = 7.537032127380371, loss_ema_slow = 7.537032127380371, lr = 0.01, val_loss = 10000.0, current_max_seq_len = 10 time per step (TOTAL) = 0:00:00.440503
==================================================
[Train] Iteration 3, train_loss = 0.3342941999435425, loss_ema_fast = 7.537032127380371, loss_ema_slow = 7.537032127380371, lr = 0.01, val_loss = 10000.0, current_max_seq_len = 10 time per step (TOTAL) = 0:00:00.439098
==================================================
[Train] Iteration 4, train_loss = 0.09542964398860931, loss_ema_fast = 7.537032127380371, loss_ema_slow = 7.537032127380371, lr = 0.01, val_loss = 10000.0, current_max_seq_len = 10 time per step (TOTAL) = 0:00:00.438517
==================================================
==================================================
==================================================
[Rank 0] 2025-05-22 18:26:12 - FINISHED TRAINING: in 4 iterations acheived 0.09542964398860931 loss in 0:00:00.438517 seconds per iter.
[Init] Model has 112606100 parameters across 1 layers.
==================================================
==================================================
==================================================
```
---

## 3. Enabling Weights & Biases or Neptune (optional)
Remote logging is **disabled by default** to simplify setup for reproducibility.  
To enable, search for each `wandb.` (or similar setup for neptune) call in the code and un-comment the block, or run with, for example:
```bash
WANDB_API_KEY=<your-key> python <script>.py --wandb_proj <project> ...
```

---

## 4. Reproducing Experimental Results 📊

> All commands assume **8 × A40** unless otherwise stated.  
> The `--verbose false` flag suppresses per-step logging for cleaner output but you can enable to get full detail:

### 4.1. Table 1 — Distributed speed / VRAM benchmarks  

#### Row 4 · **CD-RGE @ 96** (8 × 12 meta-perturbations)
```bash
# tiny (~1.0 × 10⁵ params, ≈ 0.05 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 12 --model_scale 1 \
    --hidden_size 240     --input_size 100 --num_heads 12  \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.1 --verbose false

# small (~1.0 × 10⁶ params, ≈ 0.06 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 12 --model_scale 1 \
    --hidden_size 1600    --input_size 100 --num_heads 32  \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.1 --verbose false

# medium (~1.0 × 10⁷ params, ≈ 0.07 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 12 --model_scale 1 \
    --hidden_size 9600    --input_size 100 --num_heads 64  \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.01 --verbose false

# large (~1.0 × 10⁸ params, ≈ 0.4 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 12 --model_scale 1 \
    --hidden_size 66000   --input_size 100 --num_heads 220 \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.01 --verbose false

# xlarge (~1.1 × 10⁹ params, ≈ 4.4 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 12 --model_scale 1 \
    --hidden_size 297500  --input_size 100 --num_heads 350 \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.005 --verbose false

# xxlarge (~4.5 × 10⁹ params, ≈ 17 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 12 --model_scale 1 \
    --hidden_size 1000000 --input_size 100 --num_heads 1000 \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.0001 --verbose false
```

#### Row 5 · **CD-RGE @ 512** (8 × 64 meta-perturbations)
```bash
# tiny (~1.0 × 10⁵ params, ≈ 0.24 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 64 --model_scale 1 \
    --hidden_size 240     --input_size 100 --num_heads 12  \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.5 --verbose false

# small (~1.0 × 10⁶ params, ≈ 0.25 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 64 --model_scale 1 \
    --hidden_size 1600    --input_size 100 --num_heads 32  \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.1 --verbose false

# medium (~1.0 × 10⁷ params, ≈ 0.30 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 64 --model_scale 1 \
    --hidden_size 9600    --input_size 100 --num_heads 64  \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.1 --verbose false

# large (~1.0 × 10⁸ params, ≈ 0.30 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 64 --model_scale 1 \
    --hidden_size 66000   --input_size 100 --num_heads 220 \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.1 --verbose false

# xlarge (~1.1 × 10⁹ params, ≈ 1.9 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 64 --model_scale 1 \
    --hidden_size 297500  --input_size 100 --num_heads 350 \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.01 --verbose false

# xxlarge (~4.5 × 10⁹ params, ≈ 17 s/step)
python -m torch.distributed.launch --nproc_per_node=8 distributed_rge.py \
    --batch_size 1024 --meta_perturbations 64 --model_scale 1 \
    --hidden_size 1000000 --input_size 100 --num_heads 1000 \
    --model_type LSTM --min_seq_len 10 --learning_rate 0.001 --verbose false
```

### 4.2. Table 1 — **Series** (single-GPU) benchmarks  
All series time and VRAM results are produced by a unit-test harness:
```bash
python lstm_experiments.py --unittest
```

This will run one at a time all model sizes and optimizers (BPTT and CDRGE). Please observe the times per step and you can measure the actual VRAM used via nvidia-smi when prompted with "DONE! Now go get the actual vram"

### 4.3. Figure 1 — DNC over-fit comparison (BPTT vs. CD-RGE ≈ FDRAS)

```bash
# you can run these with a HPP sweep over GPUs via:
./run_dnc_experiments.sh
```

This is setup to overfit a 7m DNC model (scale=8) with 512 perturbations per step, which if everything is running correctly, you should see train_loss drop quickly like so:
```text
Iter=0, train_loss=5.299,  val_loss=10000.000,LR=0.010000, dor 0.0, eps=0.010000, vram_inferred=0.132011 GB iter_time=82.08s, total_time=1.39m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0
Grad norm: 538539.8125
Model has 6553893 parameters across 14 layers.
Iter=1, train_loss=4.759,  val_loss=0.000,LR=0.010000, dor 0.0, eps=0.010000, vram_inferred=0.132987 GB iter_time=80.05s, total_time=2.73m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0 
Grad norm: 531739.125
Model has 6553893 parameters across 14 layers.
Iter=2, train_loss=4.348,  val_loss=0.000,LR=0.010000, dor 0.0, eps=0.010000, vram_inferred=0.132987 GB iter_time=81.25s, total_time=4.08m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0 
Grad norm: 617567.25
Model has 6553893 parameters across 14 layers.
Iter=3, train_loss=3.950,  val_loss=0.000,LR=0.010000, dor 0.0, eps=0.010000, vram_inferred=0.132987 GB iter_time=81.80s, total_time=5.45m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0
```
Vs. BPTT (which is optimizer SGD) which will drop much slower, like so:
```text
Iter=0, train_loss=5.484,  val_loss=10000.000,LR=0.001000, dor 0.0, eps=0.001000, vram_inferred=0.833
376 GB iter_time=0.70s, total_time=0.04m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sampl
e=0
Model has 6553893 parameters across 14 layers.
Iter=1, train_loss=5.416,  val_loss=0.000,LR=0.001000, dor 0.0, eps=0.001000, vram_inferred=0.841311 
GB iter_time=0.51s, total_time=0.05m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0
Model has 6553893 parameters across 14 layers.
Iter=2, train_loss=5.349,  val_loss=0.000,LR=0.001000, dor 0.0, eps=0.001000, vram_inferred=0.841311 
GB iter_time=0.49s, total_time=0.06m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0
Model has 6553893 parameters across 14 layers.
Iter=3, train_loss=5.282,  val_loss=0.000,LR=0.001000, dor 0.0, eps=0.001000, vram_inferred=0.841311 
GB iter_time=0.48s, total_time=0.07m, context_len=100, max_num=15, gen_eff_token=0, gen_eff_sample=0
```

*Guidelines*  
- If you observe **NaNs**, lower the learning rate.  
- If training is **too slow**, raise the learning rate.  
- Optimal LR for SGD ≈ *(CD-RGE LR) / 10–100*.

*Notes*
The code for all of these runs is in dnc_experiments.py where we compare many zero order methods (far beyond what is reported in the paper). For all run configs, you should run an lr sweep from 1e-5 to 0.1. The smaller the model the more perturbations, the bigger your lr should be to get the same results as us, and visa versa. Typically, sgd's optimal lr is 10x or 100x smaller than the CDRGE code. For example, to train with sgd on model_scale 8, you need to set lr = 0.0001. To train with CDRGE 512 on model_scale 8, you need to set lr = 0.01. Note, all Figure 1 results were all run with the in series code as the distributed code was developed after the fact but gives similar results. There may be discrepency at later stages due to fp16 vs. fp32, so just increase the precision if you see discrepencies with n_pert > 96. 

### 4.4. Table 2 — LSTM (FlashRNN) training sweeps
```bash
# similarly, you can run these with a HPP sweep over GPUs via:
./run_lstm_experiments.sh
```
This is setup to train LSTMs of varying model sizes on varying tasks at 96 perturbations per step just to show how the script works.

You can kill all screen session via:
```bash
screen -ls | grep '\.' | awk '{print $1}' | xargs -I{} screen -S {} -X quit 2>/dev/null
```

---

## 5. Notes on Table 1 results
Table 1 was incorrectly copied from a stale table so these values in incorrect. You will observe far **faster** distributed throughput. Here are the udpated times per step results you should expect. The table below aggregates the latest step time measured on PyTorch 2.4.0+cu121 and the distributed are on 8 GPUs vs. 10 as incorrectly specified in the paper. 

| # Params | 100 k | 1 M | 10 M | 100 M | 1.1 B |
|----------|------:|----:|-----:|------:|------:|
| **BPTT (series)**          | 0.10 | 0.17 | 33.7 | —   | —   |
| **96·CD-RGE (series)**     | 5.7  | 6.0  | 7.8  | 24.3 | 145 |
| **512·CD-RGE (series)**    | 30.0 | 31.1 | 40.7 | 132  | 777 |
| **96·CD-RGE (dist × 8)**   | 0.05 | 0.06 | 0.07 | 0.40 | 4.4 |
| **512·CD-RGE (dist × 8)**  | 0.25 | 0.29 | 0.29 | 1.87 | 19.3 |

> **Units:** seconds per training step (forward + backward + parameter update)  
> Dashes (—) indicate experiments not run due to memory limits on a single A40.
