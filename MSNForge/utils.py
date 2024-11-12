#bin/python3
import os
import re


# The standard os.path.splitext doesn't work well with multiple extensions i.e '.nii.gz'
def split_full_extension(filepath):
    # Split the file path into base and extension parts
    base, ext = os.path.splitext(filepath)
    # If it's a multi-part extension like .nii.gz, keep extracting until base has no further extension
    while os.path.splitext(base)[1]:  
        base, additional_ext = os.path.splitext(base)
        ext = additional_ext + ext
    return base, ext

def appendToBasename(filepath, addedStr, onlyBasename=False, newext=None):
    if addedStr is None:
        raise ValueError("Invalid argument: 'added_str' cannot be None.")
    if onlyBasename:
        base, ext = split_full_extension(os.path.basename(filepath))
    else:
        base, ext = split_full_extension(filepath)

    if newext is None:
        return base + addedStr + ext
    else:
        return base + addedStr + newext


# very specific to diffusion maps outputted by qsiprep
def extract_modality_suffix(filename):
    # Regular expression to capture either "dti_fa" or "ad" (before _gqiscalar)
    pattern = r"sub-\d+_ses-\d+_space-[\w]+_desc-[\w]+_desc-([\w]+(?:_[\w]+)?)_gqiscalar"
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    else:
        return None