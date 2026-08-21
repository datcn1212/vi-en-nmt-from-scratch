"""Cheap checks on a batch of decoded hypotheses
"""

def degeneracy_rate(hyps):
    degenerate = sum(1 for h in hyps if len(h) <= 2)
    return degenerate / len(hyps)

def truncation_rate(model, srcs, src_pad_masks, sos_id, eos_id, max_len, decode_fn):
    truncated = 0
    for src, mask in zip(srcs, src_pad_masks):
        hyp = decode_fn(model, src, mask, sos_id=sos_id, eos_id=eos_id, max_len=max_len)
        if len(hyp) == 0 or hyp[-1] != eos_id:
            truncated += 1
    return truncated / len(srcs)
