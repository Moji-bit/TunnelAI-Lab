# Label Quality Specification

This document defines event labels and horizon labels used in TunnelAI-Lab.

## Event Labels
Primary binary event tag:
- `Z3.EVT.Incident.Active`
  - `1` while incident is active
  - `0` otherwise

Transition tags (derived from Active):
- `Z3.EVT.Incident.Onset`
  - `1` exactly at first timestep when Active switches `0 -> 1`
- `Z3.EVT.Incident.Offset`
  - `1` exactly at first timestep when Active switches `1 -> 0`

Additional event metadata:
- `Z3.EVT.Incident.Type` (enum code)
- `Z3.EVT.Incident.Severity` (0..1)
- `Z3.EVT.Weather.Active` (binary)
- `Z3.EVT.Weather.Type` (enum code)

## Prediction Horizon Label (`H`)
For a window ending at time `t` with horizon `H` seconds:
- target `y_event(t, H) = 1` if **any** `Z3.EVT.Incident.Active == 1` in `(t, t+H]`
- else `y_event(t, H) = 0`

This is implemented in `dataset/dataset_builder.py` by checking `future = ev_np[end_x:end_y]` and labeling positive if any future value is active.

## Class Imbalance Reporting
Run the report script:

```bash
python scripts/report_label_quality.py --csv data/raw/all_runs.csv --h 60
```

Report outputs:
- per-scenario active ratio
- onset/offset counts
- horizon-positive ratio
- global active ratio and transition counts
