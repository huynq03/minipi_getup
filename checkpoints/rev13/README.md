# rev13 checkpoint: model_1100

| | |
|---|---|
| File | `model_1100.pt`, 16 MB, SHA256 `2a4451ae9a5293a7…` |
| Source run | `20261013_123752_rev6a` (working name of revision 13), iteration 1100 |
| Trained with | commit `14861b3` |
| Asset | `cl_pai_host_torso_host_legs_hip120_legcoll.xml`, SHA256 `ea7d909d9a05…` (the tracked one) |
| Curriculum state | pull force 0 N, action scale 0.466, penalty success EMA 0.50 |

## Behaviour

Measured without the pull force, at the training action scale:

- Fallen-start success 0.95.
- Two-foot rising 84% (training metric).
- Standing stance error about 2 cm, no standing jitter.

The feet still slide into place after standing; it does not step yet.

This is the last checkpoint before that run collapsed. Afterwards the drag termination ended 40–70% of
episodes, and the action scale fell to 0.24. Revision 13 now disables the drag termination and floors
the action scale at 0.30.

## Resume

```bash
tools/train_host_staged.sh -r checkpoints/rev13/model_1100.pt -i 8000 -N rev13_from1100
```

This loads the policy, the optimizer and the curriculum state (pull force, action scale, success EMA).
With a different `-n`, the per-env curriculum values are replaced by their mean.
