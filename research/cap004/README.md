# CAP004: PSN2 query-specific readout and explicit masked objective

The task readout preserves each input representation through a learned residual node-bank read. Node keys remain differentiable. Batch-average coalition commitments no longer replace distinct D1 input contexts. `forward_batch` accepts a finite nonnegative `loss_mask` and returns differentiable `logits`; detached `pred` remains available for evaluation. Existing supervised defaults retain their full-grid loss when no explicit loss_mask is supplied.

The new node read is normalize(shape+sigmoid(g)*attention(shape,nu)@nu), with normalized query/key similarities. This retains an input path while allowing the node bank to earn a residual contribution. Masked CE divides by the number/weight of requested cells, not all observed cells.

This port also carries the already-tested CAP003 spatial decoder and stable token-identity dependencies, which were not present on the current default branch. Old decoder/token checkpoints cannot silently load into the changed graph. The neighborhood-attention gradient fixture now guarantees two distinct relevant edges; random single-edge neighborhoods can legitimately have zero softmax-selection gradient.

Three seeds, 480 updates per arm: see PSN2_LEARNING.json. The new center-mask digit task remains essentially tied with the cheaper decoder-only control; no pulse or memory advantage is demonstrated. These previously exposed digit holdouts are exploratory, and all-16-pixel exact restoration remains poor. The complete developmental curriculum is not certified by these results.

One focused CAP004 case and 87 available native tests passed locally, including the 5,000-step mixed-batch validation. Two additional native modules could not collect because hypothesis is absent. Run `python -m pytest research/cap004/test_regressions.py -q` and the project's native suite in its declared dependency environment.
