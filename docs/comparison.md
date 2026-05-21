# Comparison to related software

MES is designed to **compose** established EEG tools into a rehabilitation-focused
pipeline. It does not replace them.

## When to use MES

- You record **16-channel OpenBCI** (or mapped 10–20) EEG during motor imagery or movement.
- You need a **single 0–100 engagement score** per session for longitudinal plots.
- You want **reproducible ONNX models** and a validation harness in one repository.

## When to use other tools

| Need | Tool |
|------|------|
| General preprocessing, source imaging, group stats | [MNE-Python](https://mne.tools/) |
| Benchmark many ML pipelines on public datasets | [MOABB](https://moabb.neurotechx.com/) |
| Riemannian geometry decoding research | [PyRiemann](https://pyriemann.github.io/) |
| Deep learning architecture experiments | [Braindecode](https://braindecode.org/) |
| Real-time neurofeedback decoding | py_neuromodulation, OpenViBE, etc. |

## Build vs. contribute

Contributing upstream (e.g. a scoring module in MOABB) would fragment the
**OpenBCI montage contract**, **MES formula**, **PDF/longitudinal app**, and
**HF artifact publishing** that are co-developed here. MES therefore ships as an
integrated library with clear extension points (`mes_core.pipeline`, `fit_mes_weights`).
