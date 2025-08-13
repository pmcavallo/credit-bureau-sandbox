# Model Card – Credit Bureau Segmentation (Draft)

## Overview
- **Intended Use:** Pre-decision risk segmentation and monitoring
- **Users:** Risk Strategy, Credit Ops, Data Science
- **Data:** See `data_contracts/bureau_schema.yaml`
- **Label:** `defaulted` (0/1)

## Performance Summary (to be auto-filled)
- AUC (train/valid):
- KS (train/valid):
- PSI (baseline->current):
- Calibration plot:

## Explainability
- Global: SHAP summary plot
- Local: Top features per applicant
- Candidate adverse action reasons mapped via `governance/policy_map.yaml`

## Fairness & Compliance (placeholders)
- Sensitive attributes excluded; proxy checks documented
- Drift monitoring with PSI & distributional tests
- Human-in-the-loop override policy

## Risks & Limitations
- Sandbox data may not reflect production distributions
- Reason codes are *candidates* pending Legal/Compliance review
