# MES — Motor Engagement Signal

> Open-source EEG pipeline that produces a single, calibrated score (0–100)
> reflecting the strength of motor-cortical engagement during a movement or
> motor-imagery task. Designed for tracking rehabilitation progress in stroke
> and SCI patients.

!!! warning "Research use only"
    MES is not FDA / CE cleared. Do not enter PHI. Use pseudonymous research
    codes only.

## Quick links

- **Live demo:** [https://huggingface.co/spaces/abachu2005/mes](https://huggingface.co/spaces/abachu2005/mes)
- **Source:** [github.com/abachu2005/mes](https://github.com/abachu2005/mes)
- **Model repo:** [huggingface.co/abachu2005/mes-models](https://huggingface.co/abachu2005/mes-models)
- **Processed datasets:** [huggingface.co/datasets/abachu2005/mes-eeg-processed](https://huggingface.co/datasets/abachu2005/mes-eeg-processed)

## What problem does this solve?

Movement rehabilitation needs an objective, repeatable marker for whether a
patient's motor cortex is engaging during a therapeutic task. Behavioral
measures (e.g. limb movement) miss attempted but failed motor commands; FMRI
is too slow and expensive for routine use. EEG is cheap, fast, and sensitive
to motor cortical drive — but raw EEG features (mu/beta ERD, MRCP, etc.) are
hard to read off a single trace.

MES collapses these signals into one number per trial. A score around 70+ is
consistent with strong, lateralized motor engagement; below 30 indicates
minimal engagement. Track this over weeks of therapy to see whether neural
drive is recovering.

## Read next

- [Pipeline overview](methods/pipeline.md)
- [The MES formula](methods/mes-formula.md)
- [Validation](methods/validation.md)
- [Clinical interpretation](clinical.md)
- [Hardware SOP](hardware.md)
- [Limitations](limitations.md)
