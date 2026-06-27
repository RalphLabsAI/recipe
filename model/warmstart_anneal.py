"""Warm-start anneal schedule.

Continues training from a strong base checkpoint with a short WSD-annealed
fine-tune: Muon updates on 2D parameters, decoupled lower-LR AdamW on the
embedding/unembedding, cosine-to-min over the anneal window. Preserves the
base model val_bpb while adding minimal compute.

anneal_steps = 288
embed_lr_scale = 0.3
"""
