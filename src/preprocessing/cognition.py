"""
Dataset	File	Subjects	Columns
COBRE	data/cobre_clinical/1139_Cobre_Neuropsych_V2_20261123.csv	189	111
BrainGluSchi	data/braingluschi_clinical/2079_Neurospsych_20261123.csv	175	61
--
Preprocess cognitive assessment data from COBRE and BrainGluSchi.

Merge both datasets using their shared neuropsychological features,
retain each subject's first visit, and prepare the resulting feature
table for model training.

Missing-value imputation and feature standardization should be fitted
within each training fold to prevent data leakage.
"""