"""
Preprocess resting-state functional MRI (fMRI) data from COBRE and
BrainGluSchi.

Perform motion correction and signal cleaning, then extract regional
time series using the 200-region Schaefer atlas.

Calculate pairwise correlations between regional time series to
generate a 200 x 200 functional connectivity matrix for each subject.

Exclude subjects with mean framewise displacement above 0.5 mm.
"""