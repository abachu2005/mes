# Pipeline overview

```
EDF / BDF / OpenBCI .txt
        │
        ▼
   Load (MNE)
        │
        ▼
   Bandpass 0.5–40 Hz · Notch 50/60 Hz
        │
        ▼
   Common average reference
        │
        ▼
   ICA + ICLabel artifact removal      (skipped if <32 source channels)
        │
        ▼
   Spatial map → 16-ch OpenBCI montage (spherical-spline interpolation)
        │
        ▼
   Resample to 125 Hz
        │
        ▼
   Cue-locked epoching: [-2 s, +4 s]   baseline (-1.5, -0.5)
        │
        ▼
   Feature extraction
        │   • mu/beta band power (Welch)
        │   • ERD% (task vs baseline)
        │   • Movement-Related Cortical Potential
        │   • Lateralization index
        │   • Riemannian covariance + tangent-space projection
        │
        ▼
   Classifier (ONNX, CPU)
        │   • Riemannian + Logistic Regression (baseline)
        │   • EEGNet v4 (production)
        │
        ▼
   MES combination
        │
        ▼
   Per-trial MES + session summary + PDF report
```

See [The MES formula](mes-formula.md) for the final combination step.
