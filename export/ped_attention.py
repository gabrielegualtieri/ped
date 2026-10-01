"""Banded attention for the ONNX export (`export_onnx.py --attention banded`): transformers' "eager"
results with a graph whose cost grows with L x 192 instead of L x L in the sliding-window layers.

ModernBERT / mmBERT alternate global layers (every token attends to every token) with sliding-window
layers (a token attends to the tokens at most `sliding_window - 1` = 64 positions away). Exported through
`scaled_dot_product_attention`, every layer computes the full L x L score matrix plus an L x L mask and
NaN guards around it, so a 2,000-token request spends most of its time on attention.

Registered as attention implementation "banded":

- the mask is only the key padding, [B, 1, 1, L], never an L x L matrix;
- global layers: scores + padding bias, softmax, weighted sum (the "eager" path, without the NaN guards);
- sliding layers: the sequence is cut into blocks of 64 and each block attends to its own and its two
  neighbouring blocks (192 keys) with a fixed band mask, so the cost grows with L x 192 instead of L x L.

The decision model never reads padded positions (markers sit on real tokens), and every real query sees
at least itself, so no softmax row is empty.

Measured on 4 CPU cores against the "sdpa" graph: 2.1x faster at 2,048 tokens and 2.2x at 4,096 with a
third less memory, but slower below ~1,000 tokens (every block is padded to 3 x 64 keys and runs as many
small matmuls), so the published bundles use "eager", and long documents are best read in windows
(`systemOneLong`), whose cost grows linearly. Use this for single-pass reading of very long inputs.
"""
import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface

NEG = torch.finfo(torch.float32).min


def padding_mask(batch_size, cache_position=None, kv_length=None, kv_offset=0, mask_function=None, attention_mask=None, **kwargs):
    """Key padding only: [B, 1, 1, L] booleans (True = real token), or None without padding information."""
    return None if attention_mask is None else attention_mask.bool()[:, None, None, :]


def _keep(attention_mask, q):
    if attention_mask is None:
        return torch.ones(q.shape[0], q.shape[2], dtype=torch.bool, device=q.device)
    return attention_mask.reshape(attention_mask.shape[0], -1)


def _full(query, key, value, allowed):
    scores = torch.matmul(query, key.transpose(2, 3))
    scores = scores.masked_fill(~allowed, NEG)
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(probs, value).transpose(1, 2).contiguous()


def _sliding_banded(query, key, value, keep, w):
    b = w  # block size: a block and its two neighbours cover every key within w
    B, H, L, D = query.shape
    pad = (b - L % b) % b
    q = F.pad(query, (0, 0, 0, pad))
    nb = q.shape[2] // b
    q = q.reshape(B, H, nb, b, D)
    # keys/values padded by one block on each side, then the (previous, own, next) block window per block
    k = F.pad(key, (0, 0, b, b + pad)).reshape(B, H, nb + 2, b, D)
    v = F.pad(value, (0, 0, b, b + pad)).reshape(B, H, nb + 2, b, D)
    k_win = torch.cat([k[:, :, :-2], k[:, :, 1:-1], k[:, :, 2:]], dim=3)  # [B, H, nb, 3b, D]
    v_win = torch.cat([v[:, :, :-2], v[:, :, 1:-1], v[:, :, 2:]], dim=3)
    kp = F.pad(keep, (b, b + pad), value=False).reshape(B, nb + 2, b)
    keep_win = torch.cat([kp[:, :-2], kp[:, 1:-1], kp[:, 2:]], dim=2)  # [B, nb, 3b]
    # query r of a block, key c of its window: positions n*b + r and (n-1)*b + c are |b + r - c| apart
    r = torch.arange(b, device=query.device)[:, None]
    c = torch.arange(3 * b, device=query.device)[None, :]
    band = (b + r - c).abs() <= w  # [b, 3b]
    allowed = band[None, None, None] & keep_win[:, None, :, None, :]  # [B, 1, nb, b, 3b]
    scores = torch.matmul(q, k_win.transpose(-1, -2)).masked_fill(~allowed, NEG)  # [B, H, nb, b, 3b]
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    out = torch.matmul(probs, v_win).reshape(B, H, nb * b, D)[:, :, :L]
    return out.transpose(1, 2).contiguous()


def banded_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, sliding_window=None, **kwargs):
    # query / key / value: [B, H, L, D]; scaling on the queries (1/8 for 64-dim heads: exact in floating point)
    keep = _keep(attention_mask, query)  # [B, L]
    query = query * scaling
    if sliding_window is None:
        return _full(query, key, value, keep[:, None, None, :]), None
    return _sliding_banded(query, key, value, keep, sliding_window - 1), None


AttentionInterface.register("banded", banded_attention)
AttentionMaskInterface.register("banded", padding_mask)
