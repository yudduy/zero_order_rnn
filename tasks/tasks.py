"""
Dataset generation  •  task-specific loss & accuracy utilities
"""
import os, random, string, urllib.request, numpy as np, torch
from typing import Tuple
from simpletokenizers.simpletokenizers import CharTokenizer, NumericTokenizer, get_tiktoken

# # --------------------------------------------------------------------------- #
# #  Synthetic string & arithmetic tasks                                        #
# # --------------------------------------------------------------------------- #
# def _pad_array(shape, pad_id):
#     return np.full(shape, pad_id, dtype=np.int32)

# def make_string_batch(task: str, tok, B: int, min_len=2, max_len=10):
#     pad_id = tok.token_to_id.get(" ", 0) if hasattr(tok, "token_to_id") else tok.char_to_id.get(" ", 0)
#     T = max_len*2+1
#     arr    = _pad_array((B, T), pad_id)

#     for i in range(B):
#         L       = random.randint(min_len, max_len)
#         letters = [random.choice(string.ascii_lowercase) for _ in range(L)]
#         out     = dict(copy=letters,
#                        sort=sorted(letters),
#                        reverse=list(reversed(letters)))[task]
#         full    = "".join(letters) + " " + "".join(out)
#         ids     = tok.encode(full)[:arr.shape[1]]
#         arr[i,:T] = ids[:T]
#         # arr[i,:len(ids)] = ids
#     return arr

# def make_add_batch(tok, B, min_len=2, max_len=10, max_num=110):
#     pad_id = tok.token_to_id.get(" ", 0)
#     T = max_len*2+5
#     arr    = _pad_array((B, T), pad_id)
#     for i in range(B):
#         L   = random.randint(min_len, max_len)
#         nums= [random.randint(0,9) for _ in range(L)]
#         txt = "+".join(map(str, nums)) + "=" + str(sum(nums))
#         ids = tok.encode(txt)[:arr.shape[1]]
#         arr[i,:T] = ids[:T]
#         # arr[i,:len(ids)] = ids
#     return arr

# # --------------------------------------------------------------------------- #
# #  Penn Treebank                                                               #
# # --------------------------------------------------------------------------- #
# _PTB_URLS = {
#     'train':"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
#     'valid':"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
#     'test' :"https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
# }
# def _download_ptb(split:str, root='data'):
#     os.makedirs(root, exist_ok=True)
#     fp = os.path.join(root, f"ptb.{split}.txt")
#     if not os.path.exists(fp):
#         urllib.request.urlretrieve(_PTB_URLS[split], fp)
#     with open(fp,'r',encoding='utf-8') as f: return f.read()

# def make_ptb_batch(tok, B, T, split='train'):
#     txt   = _download_ptb(split)
#     words = txt.split()
#     pad_id= tok.char_to_id.get(" ", 0)
#     arr   = _pad_array((B,T), pad_id)
#     for i in range(B):
#         s      = random.randint(0, max(0,len(words)-T-1))
#         sample = " ".join(words[s:s+T])
#         ids    = tok.encode(sample)
#         arr[i,:T] = ids[:T]
#     return arr

# # --------------------------------------------------------------------------- #
# #  Public API                                                                  #
# # --------------------------------------------------------------------------- #
# def get_examples_for_task(task:str, tok, B:int, T:int, split='train', **kw):
#     if task in ("copy","sort","reverse"):   return make_string_batch(task,tok,B,2,T)
#     if task == "repeat_copy":               return make_string_batch("copy",tok,B,T,T)  # simplified
#     if task == "add":                       return make_add_batch(tok,B,2,T,kw.get("max_num",110))
#     if task == "penn_tree_bank":            return make_ptb_batch(tok,B,T,split)
#     raise ValueError(f"unknown task {task}")

# # --------------------------------------------------------------------------- #
# #  Loss & accuracy (task specific)                                             #
# # --------------------------------------------------------------------------- #
# def _shift_logits_ids_for_lm(logits, ids_np):
#     return logits[:,:-1,:], torch.as_tensor(ids_np[:,1:], device=logits.device)

# def compute_task_loss(logits, ids_np, tok, task):
#     if task=="penn_tree_bank":                          # next-token LM
#         shifted, targets = _shift_logits_ids_for_lm(logits, ids_np)
#         return torch.nn.functional.cross_entropy(
#             shifted.reshape(-1, shifted.size(-1)),
#             targets.reshape(-1))
#     # all other tasks – generic cross-entropy on RHS of ‘ ’ or ‘=’
#     sep = tok.char_to_id.get(" ", 0) if task!="add" else tok.token_to_id.get("=",0)
#     losses=[]
#     for b in range(ids_np.shape[0]):
#         seq   = ids_np[b]
#         if sep not in seq: continue
#         p     = list(seq).index(sep)
#         target_ids = torch.as_tensor(seq[p+1:], device=logits.device, dtype=torch.long)
#         pred   = logits[b,p:p+len(target_ids)]
#         if target_ids.numel()==0: continue
#         losses.append(torch.nn.functional.cross_entropy(pred, target_ids))
#     return torch.stack(losses).mean() if losses else logits.sum()*0

# def compute_task_accuracy(logits, ids_np, tok, task):
#     if task=="penn_tree_bank":
#         shifted, targets = _shift_logits_ids_for_lm(logits, ids_np)
#         preds = shifted.argmax(-1)
#         return (preds==targets).float().mean().item()
#     sep = tok.char_to_id.get(" ",0) if task!="add" else tok.token_to_id.get("=",0)
#     correct, total = 0,0
#     for b in range(ids_np.shape[0]):
#         seq=ids_np[b]
#         if sep not in seq: continue
#         p   = list(seq).index(sep)
#         tgt = seq[p+1:]
#         pred= logits[b,p+1:p+1+len(tgt)].argmax(-1).cpu().numpy()
#         N   = len(tgt)
#         correct += (pred[:N]==tgt[:N]).sum()
#         total   += N
#     return correct/total if total else 0.0

def make_string_batch(task, tok, batch_size, min_len=2, max_len=10):
    """Create a batch for the string manip task (input: random letters, output: same letters but manip'd)"""
    # Pre-allocate array with space token for padding (not zeros)
    # For NumericTokenizer, get the space character ID (should be in symbols)
    if hasattr(tok, 'token_to_id'):
        pad_token_id = tok.token_to_id.get(" ", 0)  # Space token, default to 0 if not found
    else:
        pad_token_id = tok.char_to_id.get(" ", 0)  # For character tokenizers

    batch = np.full((batch_size, max_len*2+1), pad_token_id, dtype=np.int32)

    for i in range(batch_size):
        # Always generate a valid example that will fit
        # while True:
        # Pick a random length for this sequence
       
       
        length = random.randint(min_len, max_len)

        # Generate random lowercase letters for simplicity
        letters = [random.choice(string.ascii_lowercase) for _ in range(length)]

        # Format input and output - concatenate without commas
        input_str = "".join(letters)
        if task=="copy":
            output_str = "".join(letters)
        elif task=="sort":
            output_str = "".join(sorted(letters))
        elif task=="reverse":
            output_str = "".join(reversed(letters))
        else:
            raise Exception(f"No task named {task}")
        
        full_str = input_str + " " + output_str

        # Encode
        ids = tok.encode(full_str)

        length = min(len(ids), batch.shape[1])
        batch[i, :length] = ids[:length]
        
    return batch
    

def make_add_batch(tok, batch_size, min_len=2, max_len=10, max_num=110):
    """Create a batch for the addition task"""
    # Pre-allocate array with space token for padding (not zeros)
    # For NumericTokenizer, get the space character ID (should be in symbols)
    if hasattr(tok, 'token_to_id'):
        pad_token_id = tok.token_to_id.get(" ", 0)  # Space token, default to 0 if not found
    else:
        pad_token_id = tok.char_to_id.get(" ", 0)  # For character tokenizers

    batch = np.full((batch_size, max_len*2+5), pad_token_id, dtype=np.int32)

    for i in range(batch_size):
        # Pick a random length for this sequence
        
        length = random.randint(min_len, max_len)

        # Generate random single-digit numbers
        numbers = [random.randint(0, 9) for _ in range(length)]

        # Calculate sum (ensuring it doesn't exceed max_num)
        sum_val = sum(numbers)
        
        # Format the example
        input_str = "+".join(str(n) for n in numbers)
        output_str = str(sum_val)
        full_str = input_str + "=" + output_str

        # Encode
        ids = tok.encode(full_str)

        length = min(len(ids), batch.shape[1])
        batch[i, :length] = ids[:length]
        
    return batch

# PTB dataset loading functions
def load_penn_tree_bank(data_dir='data', split='train'):
    """Load Penn Treebank data from file or download if not available"""
    # Define file paths
    os.makedirs(data_dir, exist_ok=True)

    # URLs for PTB datasets
    urls = {
        'train': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
        'valid': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt',
        'test': 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt'
    }

    file_path = os.path.join(data_dir, f'ptb.{split}.txt')

    # Download if file doesn't exist
    if not os.path.exists(file_path):
        try:
            import urllib.request
            print(f"Downloading Penn Treebank {split} set...")
            urllib.request.urlretrieve(urls[split], file_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading Penn Treebank data: {str(e)}")
            # Provide a small sample of PTB data as fallback
            if split == 'train':
                text = ("the cat sat on the mat . a dog barked loudly . "
                        "he ran quickly through the park . she smiled at him warmly .")
            else:
                text = "the girl walked home . they played games all day ."

            # Write the sample data to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text)

            return text

    # Read the data
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text

def make_penn_tree_bank_batch(tok, batch_size, seq_length, split='train'):
    """Create a batch for the Penn Tree Bank language modeling task"""
    # Pre-allocate array with space token for padding (not zeros)
    # For NumericTokenizer, get the space character ID (should be in symbols)
    if hasattr(tok, 'token_to_id'):
        pad_token_id = tok.token_to_id.get(" ", 0)  # Space token, default to 0 if not found
    else:
        pad_token_id = tok.char_to_id.get(" ", 0)  # For character tokenizers

    batch = np.full((batch_size, seq_length), pad_token_id, dtype=np.int32)

    # Load PTB data or use synthetic data for testing
    try:
        text = load_penn_tree_bank(split=split)
        words = text.split()

        # Ensure we have enough words
        if len(words) < seq_length:
            # Repeat words if needed
            words = words * (seq_length // len(words) + 1)

        # Create indices for efficient sampling
        max_start_idx = len(words) - seq_length

        # Generate batch_size samples
        for i in range(batch_size):
            if max_start_idx > 0:
                # Pick a random starting point
                start_idx = random.randint(0, max_start_idx)
                seq = ' '.join(words[start_idx:start_idx+seq_length])
            else:
                # If text is too short, generate a simple sequence
                common_words = ["the", "of", "and", "to", "in", "a", "is", "that", "for", "it"]
                seq = ' '.join(random.choice(common_words) for _ in range(seq_length // 5))

            # Encode and add to batch
            ids = tok.encode(seq)
            # Copy to batch (truncate if needed)
            length = min(len(ids), seq_length)
            batch[i, :length] = ids[:length]

    except Exception as e:
        # Fallback to synthetic data if PTB loading fails
        raise Exception("CANT LOAD PTB!")
 
    return batch

def get_examples_for_task(task, tok, batch_size, seq_length, split='train', max_num=110):
    """Generate appropriate data batch based on the specified task and split"""

    if seq_length<2:
        print("cant have a seq_length<2")
        seq_length = 2 
        
    # Set length ranges based on train/val split
    if split == 'train':
        min_len, max_len = 2, seq_length
    else:  # validation
        min_len, max_len = seq_length + 1, seq_length + 50 

    # Safety check: ensure min_len <= max_len
    if min_len > max_len:
        min_len, max_len = max_len, min_len  # Swap if min > max

    if task in ["copy","sort","reverse"]:
        # Generate random texts for copy task (lowercase letters only)
        chars = string.ascii_lowercase  # Only lowercase letters, no digits
        
        # Validate min_len and max_len
        if min_len > max_len:
            min_len, max_len = max_len, min_len  # Swap if min > max
        
        return make_string_batch(task, tok, batch_size, min_len=min_len, max_len=max_len)

    elif task == "repeat_copy":
        # seq_length = max_len
        # Generate random texts for repeat copy task (lowercase letters only)
        chars = string.ascii_lowercase  # Only lowercase letters, no digits
        
        # Get space token ID for padding
        if hasattr(tok, 'token_to_id'):
            pad_token_id = tok.token_to_id.get(" ", 0)  # Space token, default to 0 if not found
        else:
            pad_token_id = tok.char_to_id.get(" ", 0)  # For character tokenizers

        max_num_copies = 4
        # Create a batch with varying text lengths and repetition counts
        batch = np.full((batch_size, max_len*(max_num_copies+1)), pad_token_id, dtype=np.int32)
        
        for i in range(batch_size):
            # Generate random text with appropriate length, ensuring min_value < max_value
            length = random.randint(min_len, max_len)
            txt = ''.join(random.choice(chars) for _ in range(length))
            
            # Random repetition count (1-9) - different for each example
            repeats = random.randint(1, max_num_copies)
                            
            # Format with repetition indicator
            repeats_indicator = str(repeats)
            output = txt * repeats
            
            # Encode
            ids = tok.encode(txt + repeats_indicator + " " + output)
            
            # Copy into batch (truncate if needed)
            length = min(len(ids), batch.shape[1])
            batch[i, :length] = ids[:length]
            
        return batch

    elif task == "add":
        # Generate addition examples
        return make_add_batch(tok, batch_size, min_len, max_len, max_num)

    elif task == "penn_tree_bank":
        # Use Penn Tree Bank corpus
        return make_penn_tree_bank_batch(tok, batch_size, max_len, split)

    else:
        raise ValueError(f"Unknown task: {task}")



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


# ------------------------------------------------------------------
#  unified data loader
# ------------------------------------------------------------------
def generate_openwebtext_task(
        num_samples: int,
        ds,                                 # the HF dataset / list of docs
        tokenizer,                          # HF tokenizer *or* CharTokenizer
        min_tokens: int  = 10,
        max_tokens: int  = 2048,
        return_strings: bool = False,
        device: str = "cuda",
        verbose: bool = False,
):
    """
    Return a batch of (x_ids, y_ids) ready for language-model training.
    Works with both HF and the simple CharTokenizer.

    x = <bos> + first token
    y = remaining tokens + <eos>

    Shapes: (num_samples, seq_len)  – padded on the right with pad-token-id.
    """
    # --- tiny shim so HF and CharTokenizer share the same API -----------
    is_hf = hasattr(tokenizer, "bos_token_id")  # crude but works

    bos = tokenizer.bos_token_id if is_hf else tokenizer.char_to_id["<"]
    eos = tokenizer.eos_token_id if is_hf else tokenizer.char_to_id[">"]
    pad = tokenizer.pad_token_id if is_hf else 0

    encode = tokenizer.encode if is_hf else (lambda s: tokenizer.encode(s))
    decode = tokenizer.decode if is_hf else (lambda ids: tokenizer.decode(ids))

    # -------------------------------------------------------------------
    x_batch, y_batch   = [], []
    x_strings, y_strings = [], []

    ds_len = len(ds)

    while len(x_batch) < num_samples:
        # 1) pick a random doc
        text = ds[random.randint(0, ds_len - 1)].get("text", "")

        # 2) tokenise (HF) or char-split
        tokens = encode(text)
        if len(tokens) < min_tokens:
            continue

        # 3) maybe trim
        if len(tokens) > max_tokens:
            sub_len = random.randint(min_tokens, max_tokens)
            start   = random.randint(0, len(tokens) - sub_len)
            tokens  = tokens[start : start + sub_len]

        # 4) split into x / y
        split       = 1
        x_tokens    = [bos] + tokens[:split]
        y_tokens    = tokens[split:] + [eos]

        # 5) save
        x_batch.append(torch.tensor(x_tokens, device=device))
        y_batch.append(torch.tensor(y_tokens, device=device))

        if return_strings:
            x_strings.append(decode(x_tokens))
            y_strings.append(decode(y_tokens))

    # 6) pad to max-len in batch
    x_ids = torch.nn.utils.rnn.pad_sequence(x_batch, batch_first=True, padding_value=pad)
    y_ids = torch.nn.utils.rnn.pad_sequence(y_batch, batch_first=True, padding_value=pad)

    if return_strings:
        return x_ids, y_ids, x_strings, y_strings
    return x_ids, y_ids


# def generate_openwebtext_task_unified(
#     num_samples: int,
#     ds,
#     tokenizer,
#     min_tokens: int = 10,
#     max_tokens: int = 2048,
#     char_tokenizer: bool = False,
#     char_to_id=None,
#     str_to_tensor=None,
#     return_strings: bool = False,
#     device="cuda",
#     verbose: bool = False,
# ):
#     input_token_list = []
#     target_token_list = []
#     input_str_list = []
#     target_str_list = []
 
#     size_of_ds = len(ds) #300 #len(ds) # REMOVE AFTER TESTING FINISHED! WANT TO TEST IF WE CAN OVERFIT

#     while len(input_token_list) < num_samples:
#         doc_idx = random.randint(0, size_of_ds - 1)
#         text = ds[doc_idx].get("text", "") or ""

#         if char_tokenizer:
#             total_len = len(text)
#         else:
#             tokens = tokenizer.encode(text, add_special_tokens=True)
#             total_len = len(tokens)
            

#         if total_len < min_tokens:
#             continue

#         if total_len > max_tokens:
#             # Uniform sample a substring in range [min_tokens, max_tokens]
#             sub_len = random.randint(min_tokens, max_tokens)
#             start = random.randint(0, total_len - sub_len)
#             if char_tokenizer:
#                 text = text[start:start + sub_len]
#             else:
#                 tokens = tokens[start:start + sub_len]

#         # If char tokenizer, split on characters
#         if char_tokenizer:
#             split_point = 1
#             x_str = "<bos>" + text[:split_point]
#             y_str = text[split_point:] + "<eos>"

#             x_ids = str_to_tensor([x_str], char_to_id).to(device)
#             y = str_to_tensor([y_str], char_to_id).to(device)

#             input_token_list.append(x_ids[0])
#             target_token_list.append(y[0])

#             if return_strings:
#                 input_str_list.append(x_str)
#                 target_str_list.append(y_str)

#         else:
#             # For HF tokenizer: work directly with tokens
#             split_point = 1

#             input_tokens = [tokenizer.bos_token_id] + tokens[:split_point]
#             target_tokens = tokens[split_point:] + [tokenizer.eos_token_id]

#             # input_tokens = tokens[:split_point]
#             # target_tokens = tokens[split_point:]

#             input_token_list.append(torch.tensor(input_tokens, device=device))
#             target_token_list.append(torch.tensor(target_tokens, device=device))

#             if return_strings:
#                 input_str_list.append(tokenizer.decode(input_tokens, skip_special_tokens=False))
#                 target_str_list.append(tokenizer.decode(target_tokens, skip_special_tokens=False))

#             # Pad and stack
#             x_ids = torch.nn.utils.rnn.pad_sequence(input_token_list, batch_first=True, padding_value=tokenizer.pad_token_id)
#             y = torch.nn.utils.rnn.pad_sequence(target_token_list, batch_first=True, padding_value=tokenizer.pad_token_id)

#     # print("="*50)
#     # print(f" x_ids: {x_ids}")
#     # print(f" y: {y}")
#     # print(f" input_str_list: {input_str_list}")
#     # print(f" target_str_list: {target_str_list}")
#     # print("="*50)
#     # import pdb
#     # pdb.set_trace()
#     if return_strings:
#         return x_ids, y, input_str_list, target_str_list
#     return x_ids, y
