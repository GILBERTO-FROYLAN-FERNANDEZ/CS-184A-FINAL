def cognition(raw_data):
    """
    Preprocess cognitive assessment data from COBRE and BrainGluSchi.

    Merge both datasets using their shared neuropsychological features,
    retain each subject's first visit, and prepare the resulting feature
    table for model training.

    Missing-value imputation and feature standardization should be fitted
    within each training fold to prevent data leakage.
    """
    pass

def structural_mri(raw_data):
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
    pass

def functional_mri(raw_data):
    """
    Preprocess resting-state functional MRI (fMRI) data from COBRE and
    BrainGluSchi.

    Perform motion correction and signal cleaning, then extract regional
    time series using the 200-region Schaefer atlas.

    Calculate pairwise correlations between regional time series to
    generate a 200 x 200 functional connectivity matrix for each subject.

    Exclude subjects with mean framewise displacement above 0.5 mm.
    """
    pass