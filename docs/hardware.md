# Hardware SOP — OpenBCI Cyton + Daisy

## Target setup

- **Board:** OpenBCI Cyton + Daisy (16 channels)
- **Sample rate:** 125 Hz
- **Channel order in the production model:**
  Fpz, Fz, FC3, FCz, FC4, C3, C1, Cz, C2, C4, CP3, CPz, CP4, T7, T8, Pz
- **Reference:** A1 (left earlobe). Bias: A2 (right earlobe).

## Electrode placement (10-20)

Use a snug, properly sized cap. Gel/saline electrodes should reach
impedance < 20 kΩ; dry electrodes < 200 kΩ.

```
              Fpz
               
       Fz
   FC3 FCz FC4
 T7 C3 C1 Cz C2 C4 T8
   CP3 CPz CP4
        Pz
```

## Recording protocol

1. Cap participant. Wait 5 minutes for gel to settle.
2. Check impedances with the OpenBCI GUI. Re-gel any channel > 50 kΩ.
3. Have the participant sit relaxed, eyes open, fixating on a cross.
4. **Rest block** — 60 s of quiet rest. Save as `participantcode_S###_rest.txt`.
5. **Task block** — alternate 5 s cue / 5 s task / 5 s rest, for 30+ trials.
   Save as `participantcode_S###_task.txt`.
6. Stop recording and immediately upload to MES.

## File export

- In the OpenBCI GUI, use **File → Save as → .txt** (or .csv).
- Include the standard header (Sample Rate, Number of channels, Board).
- Do not modify the file before upload.

## Common pitfalls

- **Jaw / face muscle artifact** in T7/T8 → instruct participant to relax jaw.
- **Eye blinks** in Fpz/Fz → ICA handles this automatically on the source montage.
- **Channel pops** from cable movement → re-gel and re-record the affected segment.
