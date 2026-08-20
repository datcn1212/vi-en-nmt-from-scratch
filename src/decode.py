"""Shared decoding: only calls model.encode() and model.decode_step(), so
every architecture is decoded the same way. Beam search added later.
"""
import torch


def greedy_decode(model, src, src_pad_mask, sos_id=1, eos_id=2, max_len=50):
    with torch.no_grad():
        memory = model.encode(src, src_pad_mask)
        tgt = [sos_id]
        result = []
        for _ in range(max_len):
            logits = model.decode_step(memory, src_pad_mask, torch.tensor([tgt]))
            next_id = logits.argmax(dim=-1).item()
            result.append(next_id)
            tgt.append(next_id)
            if next_id == eos_id:
                break
    return result
