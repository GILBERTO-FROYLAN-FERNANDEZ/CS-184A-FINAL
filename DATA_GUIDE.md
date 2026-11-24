# Data Guide

This is a map of the downloaded data: what each folder and file holds, and which parts the project uses.
The per-file details (row, column and subject counts) are in [`data_inventory.csv`](data_inventory.csv).

> **Do not commit or share the data folders.** They are covered by the COINS data use agreement.
> This guide and the inventory hold only counts, no subject-level data.

## 1. Overview

| Folder | What it is | Size |
|---|---|---|
| `COBRE data/` | Clinical and test CSVs for the COBRE study (prefix `1139_`) | ~1 MB |
| `BrainGluSchi data/` | Clinical and test CSVs for the BrainGluSchi study (prefix `2079_`) | ~0.6 MB |
| `COBRE MR/` | 22 zips of raw MRI (DICOM) for COBRE | ~127 GB |
| `BrainGluSchi MR/` | 12 zips of raw MRI (DICOM) for BrainGluSchi | ~58 GB |

- Both studies come from the Mind Research Network in New Mexico and were scanned on the same scanner model (Siemens 3T Trio).
- Every file links by the **`Anonymized ID`** column, e.g. `A00000243`. In the MRI zips, the same ID is the subject folder name.
- **Usable subjects:** 365 have a label and at least one data type (189 controls, 176 schizophrenia). Of those, 309 have all three data types the project uses (166 controls, 143 schizophrenia).

## 2. The data grouped by purpose

### A. Labels: who is a patient
The label comes from the `Subject Type` column in `*_Demographics_*.csv`.

| Subject Type | Study | Project label |
|---|---|---|
| Control | COBRE | control |
| Old Control, Young Control | BrainGluSchi | control |
| Schizophrenia | COBRE | schizophrenia |
| Chronic Sz, Early Sz | BrainGluSchi | schizophrenia |
| Bipolar, Schizoaffective | both | **dropped** (21 subjects) |

`COBRE data/enrollment_summary.csv` lists which studies each person enrolled in. **104 people appear in both studies' clinical files.** Count them once, and keep each one on a single side of any train/test split.

### B. Cognitive tests: tabular model input
| File | Subjects | Columns |
|---|---|---|
| `COBRE data/1139_Cobre_Neuropsych_V2_20261123.csv` | 189 | 111 |
| `BrainGluSchi data/2079_Neurospsych_20261123.csv` | 175 | 61 |

All 61 BrainGluSchi columns also appear in the COBRE file. Use those 61 shared columns when combining the two studies.

### C. Demographics: confounds, not inputs
`*_Demographics_*.csv` holds age, sex (`Gender`), education, ethnicity and race. Use these to check confounds, not as model features.
- **Sex is unbalanced:** 80–90% of patients have Gender = 1, compared with about 70% of controls.
- **Age varies by group:** the mean age of each subgroup ranges from about 24 to 46.

### D. Patient-only forms: never use as inputs
Only patients filled out these forms, so whether a file exists for a person already reveals their label. Using them as inputs would leak the answer.
- `PANSS` (symptoms), `CGI` (illness severity), `Calgary` (depression), `QOL` (quality of life), `UPSA` (COBRE only)
- `Medication_Log`: may be used **only** for the confound check of medication dose vs. prediction
- `SCID_II_Summary`: the diagnostic interview, which encodes the diagnosis itself

### E. Other forms: not used
- **Medication side-effect scales:** `AIMS`, `BAS`, `SAS`
- **Medical and neurological:** `Medical_History`, `Neurological_Exam` (COBRE only)
- **Mood scales, few subjects:** `Young_Mania_Scale`, `Schizo-Bipolar_Scale`
- **Optional confounds:** `Fagerstrom` (smoking) and `Pre-Scan_Form` (caffeine, sleep and nicotine before each scan)
- **Not used:** `Debriefing_Form` (post-task questionnaire)

### F. MRI scans
Inside the zips, the files are laid out like this:
```
COBRE:        human/dicom/triotim/vcalhoun/cobre_07325/<ID>/Session_N/<scan>_NNNN/*.dcm
BrainGluSchi: human/dicom/triotim/jbustillo/braingluschi_20542/<ID>/<num>_visit1/<scan>/*.dcm
```

| Scan | Folder name | COBRE subjects | BrainGluSchi subjects | Used? |
|---|---|---|---|---|
| T1 structural | `mprage_5e` (COBRE also has `mprage_5e_RMS`) | 193 | 184 | **yes**, for brain region volumes and thickness |
| Resting fMRI | `RSTpre_V01_R01` / `rest_V01_R01` | 190 | 163 | **yes**, for connectivity between brain regions |
| Diffusion (DTI) | `dti_35dir` | 162 | present (not counted) | no (out of scope) |
| T2 | `t2_tse_tra_192` | 139 | present (not counted) | no |
| Task fMRI | GATE task | ~123 | — | no |

- Each subject has one folder per scan holding the DICOM slices.
- The resting-scan folders also contain `+CM.nii` and `+CP.nii` files (4D, 64×64×33×~150). What they are is unconfirmed, so convert from the DICOMs instead.
- No subject appears in both studies' MRI.

## 3. Gotchas
1. **Each clinical CSV has two header rows.** Row 1 has readable names and row 2 has codes such as `FIPAN_1`. `enrollment_summary.csv` has only one.
2. **Column names can repeat** (e.g. in QOL and Calgary), which breaks PowerShell `Import-Csv`. pandas handles repeats by adding `.1`.
3. **There can be several rows per person**, one per visit. Use the first visit unless there is a reason not to.
4. **COBRE files have three extra columns** (`Sub Study Label`, `Days since sub study enrollment`, `Days since first enrollment`). BrainGluSchi files have just `Days since enrollment`.
5. `COBRE MR/assessment_data.zip` is a copy of `COBRE data/`, so ignore it.
6. Leave the MRI zipped. Extract only the T1 and resting-scan folders you need, into a separate working folder.

## 4. How to load a clinical CSV
```python
import pandas as pd

def load(path):
    # Row 1 = readable header, row 2 = coded names (skip it).
    return pd.read_csv(path, header=0, skiprows=[1])

demo = load("COBRE data/1139_Demographics_20261123.csv")
cog  = load("COBRE data/1139_Cobre_Neuropsych_V2_20261123.csv")
df = cog.merge(demo[["Anonymized ID", "Subject Type", "Current Age", "Gender"]]
               .drop_duplicates("Anonymized ID"), on="Anonymized ID")
```
