"""
Preprocess structural MRI (sMRI) data from COBRE and BrainGluSchi.

Convert DICOM scans to NIfTI using dcm2niix. Use COBRE's combined T1
scans and combine BrainGluSchi's five echoes where necessary.

Extract regional brain volumes and cortical thickness measurements
using FastSurfer. Apply ComBat harmonization to reduce study-related
differences while preserving age, sex, and diagnosis effects.

Return a subject-level feature table containing approximately
100 regional structural measurements.
"""