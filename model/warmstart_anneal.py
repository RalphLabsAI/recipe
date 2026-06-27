"""Warm-start anneal schedule.

Continues from a strong base checkpoint with a short WSD-annealed
fine-tune: Muon on 2D params, decoupled lower-LR AdamW on embeddings,
cosine-to-min over the anneal window. Preserves base val_bpb at minimal compute.

anneal_steps = 344
embed_lr_scale = 0.3
"""
