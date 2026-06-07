---
name: kaggle-yolo-submission
description: Use this skill when working on the local Kaggle object-detection submission workflow that uses read-only YOLO weight files, notebooks, sample_submission.csv, and an images folder. Trigger for requests involving 01_best.pt with v3_01_WBF_outputcsv.ipynb, 08_best.pt with v5_08_NMS_outputcsv.ipynb, producing or checking Kaggle submission CSV files, adapting WBF/NMS inference notebooks, or explaining how to handle these artifacts without modifying originals.
---

# Kaggle YOLO Submission

This is a project-specific skill for the current workspace. Use the file names and assumptions below only inside this project.

## Core Rule

Treat all user-provided competition artifacts as read-only source files:

- `01_best.pt`
- `v3_01_WBF_outputcsv.ipynb`
- `08_best.pt`
- `v5_08_NMS_outputcsv.ipynb`
- `sample_submission.csv`
- `images/`

Do not overwrite, rename, move, delete, reformat, or checkpoint these files. If changes are needed, copy notebooks or create new output files with explicit names such as `submission_01_wbf.csv`, `submission_08_nms.csv`, or `submission_ensemble.csv`.

## Artifact Map

Use these pairs as the intended implementations:

- Implementation 01: `01_best.pt` + `v3_01_WBF_outputcsv.ipynb`
- Implementation 08: `08_best.pt` + `v5_08_NMS_outputcsv.ipynb`
- Kaggle dataset input: `sample_submission.csv` + `images/`

Assume `sample_submission.csv` defines the required submission columns and image ids. Use it as the schema source before generating any final CSV.

## Workflow

1. Inspect the working directory and confirm required artifacts exist before running inference.
2. Read notebooks as JSON or through notebook-aware tooling; avoid editing the original `.ipynb` files directly.
3. Preserve the original model-to-notebook pairing unless the user explicitly asks for cross-testing or ensembling.
4. Run inference against images from `images/` using the corresponding `.pt` file.
5. Apply the notebook's intended post-processing:
   - Implementation 01 uses WBF from `v3_01_WBF_outputcsv.ipynb`.
   - Implementation 08 uses NMS from `v5_08_NMS_outputcsv.ipynb`.
6. Emit a new Kaggle submission CSV that matches `sample_submission.csv` column names, row count expectations, and prediction-string format.
7. Validate the output CSV before reporting completion.

## Validation Checklist

Before calling the result done, verify:

- The original six artifacts listed in Core Rule are unchanged.
- The output CSV has the same required columns as `sample_submission.csv`.
- Every image id required by `sample_submission.csv` has exactly one output row.
- Missing detections use the competition's expected empty prediction representation.
- Bounding boxes, class ids, and confidence scores use the format expected by the notebook and competition.
- The output filename clearly identifies which implementation or ensemble produced it.

## Editing Guidance

When notebook changes are necessary, create a copy first, for example:

```powershell
Copy-Item -LiteralPath .\v3_01_WBF_outputcsv.ipynb -Destination .\v3_01_WBF_outputcsv_working.ipynb
```

Modify only the copied notebook or generate a separate script from it. Keep any generated caches, labels, predictions, and CSVs separate from the original Kaggle input files.

## User Communication

When explaining this workflow to the user, state plainly that the listed `.pt`, `.ipynb`, `sample_submission.csv`, and `images/` artifacts are read-only inputs. Mention the exact output file created or proposed, and summarize validation results rather than dumping notebook internals.
