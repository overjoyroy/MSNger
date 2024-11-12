#bin/python3
import argparse
import nibabel as nib
import nipy as nipy
import nipype.interfaces.io as nio 
import nipype.interfaces.fsl as fsl 
import nipype.interfaces.ants as ants 
import nipype.interfaces.utility as util 
import nipype.pipeline.engine as pe
import numpy as np
import os, sys
import pandas as pd
import re
import shutil
from pathlib import Path
import time

DATATYPE_SUBJECT_DIR = 'anat'
DATATYPE_FILE_SUFFIX = 'T1w'

def makeParser():
    parser = argparse.ArgumentParser(
                        prog='T1_Preproc', 
                        usage='This program preprocesses T1 MRIs for later use with an MSN development pipeline.',
                        epilog='BUG REPORTING: Report bugs to pirc@chp.edu or more directly to Joy Roy at the Childrens Hospital of Pittsburgh.'
        )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-p','--parentDir', nargs=1, required=True,
                        help='Path to the parent data directory. BIDS compatible datasets are encouraged.')
    parser.add_argument('-sid','--subject_id', nargs=1, required=False,
                        help='Subject ID used to indicate which patient to preprocess')
    # parser.add_argument('-spath','--subject_t1_path', nargs=1, required=False,
    #                     help='Path to a subjects T1 scan. This is not necessary if subject ID is provided as the T1 will be automatically found using the T1w.nii.gz extension')
    parser.add_argument('-ses_id','--session_id', nargs=1, required=False,
                        help='Session ID used to indicate which session to look for the patient to preprocess')
    parser.add_argument('-tem','--template', nargs=1, required=False,
                        help='Template to be used to register into patient space. Default is MNI152lin_T1_2mm_brain.nii.gz')
    parser.add_argument('-seg','--segment', nargs=1, required=False,
                        help='Atlas to be used to identify brain regions in patient space. This is used in conjunction with the template. Please ensure that the atlas is in the same space as the template. Default is the AALv2 template.')
    parser.add_argument('-o','--outDir', nargs=1, required=True,
                        help='Path to the \'derivatives\' folder or chosen out folder.')
    parser.add_argument('--testmode', required=False, action='store_true',
                        help='Activates TEST_MODE to make pipeline finish faster for quicker debugging')

    return parser 

def vet_inputs(args):
    """
    Takes the parsed arguments from the user, checks them, and ensures that the paths are absolute.
    
    Parameters:
    args (argparse.Namespace): Parsed arguments from argparse.
    
    Returns:
    argparse.Namespace: The updated arguments with absolute paths where applicable.
    """
    # Convert parent directory to absolute path if it's relative
    args.parentDir = [os.path.abspath(os.path.expanduser(args.parentDir[0]))]
    
    # Convert output directory to absolute path if it's relative
    args.outDir = [os.path.abspath(os.path.expanduser(args.outDir[0]))]
    if not os.path.exists(args.outDir[0]):
        print("Output directory does not currently exist. Making it now.")
        os.makedirs(args.outDir[0], exist_ok=True)
    
    # Convert template path to absolute if provided and relative
    if args.template:
        args.template = [os.path.abspath(os.path.expanduser(args.template[0]))]
    else:
        args.template = ['/app/Template/MNI152lin_T1_2mm_brain.nii.gz']
    
    # Convert segmentation atlas path to absolute if provided and relative
    if args.segment:
        args.segment = [os.path.abspath(os.path.expanduser(args.segment[0]))]
    else:
        args.segment = ['/app/Template/aal2.nii.gz']
    
    # Validate subject ID
    if not args.subject_id or not isinstance(args.subject_id[0], str):
        raise ValueError("Subject ID is missing or invalid. It should be in the form 'sub-#+' ")
    
    # Validate session ID if provided
    if args.session_id and not isinstance(args.session_id[0], str):
        raise ValueError("Session ID is invalid. It should be in the form 'ses-#+' ")
    elif not args.session_id :
        for i in os.listdir(os.path.join(args.parentDir[0], args.subject_id[0])):
            if 'ses-' in i:
                ValueError("Session ID is invalid. Your data seems to be organized by sessions but one was not provided.")
    
    return args

# This was developed instead of using the default parameter in the argparser
# bc argparser only returns a list or None and you can't do None[0]. 
# Not all variables need a default but need to be inspected whether they are None
def vetArgNone(variable, default):
    if variable==None:
        return default
    else:
        return os.path.abspath(os.path.expanduser(variable[0]))

def makeOutDir(outDirName, args):

    outDirPath_parents = args.outDir[0]
    if '~' in outDirPath_parents:
        outDirPath_parents = os.path.expanduser(outDirPath_parents)

    outDirPath_parents = os.path.abspath(outDirPath_parents)

    segmentName =  os.path.basename(args.segment[0]).split('.')[0] #/path/to/aal.nii.gz
    print('Atlas name appears to be: {}. This will be the name of the output subdirectory'.format(segmentName))


    outDir = os.path.join(outDirPath_parents, outDirName, args.subject_id[0], segmentName)

    if not os.path.exists(outDir):
        os.makedirs(outDir, exist_ok=True)

    print("Outputting results to path: {}".format(outDir))

    return outDir


def copy_modality_outputs(modality_path, out_dir, keywords=None):
    if keywords is None:
        raise ValueError("Keywords list cannot be None. Please provide keywords to match in filenames.\n")
    
    if not os.path.exists(modality_path):
        print(f'WARNING!: This path {modality_path} does not exist! Copy cannot complete.\n')
        return -1

    print(f"Copying files from {modality_path} to the destination directory.")
    for root, _, filenames in os.walk(modality_path):
        for file in filenames:
            if any(keyword in file for keyword in keywords):
                source = os.path.join(root, file)
                destination = os.path.join(out_dir, file)
                shutil.copy2(source, destination)
                print(f"Copied {file} to {destination}")

    print("Copy operation complete.\n")
    return 0

def copyModalityOutputsToForgery(args, outDir):
    outDirPath = Path(outDir)
    basepath = outDirPath.parents[2]

    print('Copying T1 outputs...')
    t1Path = os.path.join(basepath, 'RadT1cal_Features', args.subject_id[0])
    copy_modality_outputs(t1Path, outDir, ['_trans_radiomicsFeatures.csv', '_trans_volumes.csv'])

    print('Copying BOLD outputs...')
    boldPath = os.path.join(basepath, 'Sim_Funky_Pipeline', args.subject_id[0])
    copy_modality_outputs(boldPath, outDir, ['average_arr.csv', 'sim_matrix.csv'])

    print('Copying DTI outputs...')
    dtiPath = os.path.join(basepath, 'qsirecon', args.subject_id[0])
    copy_modality_outputs(dtiPath, outDir, ['_gqiscalar.nii.gz'])


def registerAtlasToMap(args, outDir, diffusionMapPath):

    tempDir = os.path.join(outDir, 'registration_intermediates')
    if not os.path.exists(tempDir):
        os.makedirs(tempDir, exist_ok=True)

    # FSL affine is better than ANTs affine imo
    flt = fsl.FLIRT()
    flt.inputs.in_file = args.template[0]
    flt.inputs.reference =  diffusionMapPath
    flt.inputs.out_file = os.path.join(tempDir, appendToBasename(args.template[0], '_flirt', True))
    flt.inputs.out_matrix_file = os.path.join(tempDir, appendToBasename(args.template[0], '_flirt', True, '.mat'))
    fltout = flt.run()

    # apply same transformation to segment
    applyflt = fsl.FLIRT()
    applyflt.inputs.in_file = args.segment[0]
    applyflt.inputs.reference =  diffusionMapPath
    applyflt.inputs.apply_xfm = True
    applyflt.inputs.in_matrix_file = fltout.outputs.out_matrix_file
    applyflt.inputs.interp = 'nearestneighbour'
    applyflt.inputs.out_file = os.path.join(tempDir, appendToBasename(args.segment[0], '_flirt', True))
    applyflt.inputs.out_matrix_file = os.path.join(tempDir, appendToBasename(args.segment[0], '_flirt', True, '.mat'))
    applyfltout = applyflt.run()

    
    # register MNI using ants affine + nonlinear
    antsReg = ants.Registration()
    antsReg.inputs.transforms = ['Affine', 'SyN']
    antsReg.inputs.transform_parameters = [(2.0,), (0.25, 3.0, 0.0)]
    antsReg.inputs.number_of_iterations = [[1500, 200], [100, 50, 30]]
    if args.testmode==True:
        antsReg.inputs.number_of_iterations = [[5, 5], [5, 5, 5]]
    antsReg.inputs.dimension = 3
    antsReg.inputs.write_composite_transform = False
    antsReg.inputs.collapse_output_transforms = False
    antsReg.inputs.initialize_transforms_per_stage = False
    antsReg.inputs.metric = ['Mattes']*2
    antsReg.inputs.metric_weight = [1]*2 # Default (value ignored currently by ANTs)
    antsReg.inputs.radius_or_number_of_bins = [32]*2
    antsReg.inputs.sampling_strategy = ['Random', None]
    antsReg.inputs.sampling_percentage = [0.05, None]
    antsReg.inputs.convergence_threshold = [1.e-8, 1.e-9]
    antsReg.inputs.convergence_window_size = [20]*2
    antsReg.inputs.smoothing_sigmas = [[1,0], [2,1,0]]
    antsReg.inputs.sigma_units = ['vox'] * 2
    antsReg.inputs.shrink_factors = [[2,1], [3,2,1]]
    antsReg.inputs.use_histogram_matching = [True, True] # This is the default
    antsReg.inputs.output_warped_image = os.path.join(tempDir, appendToBasename(fltout.outputs.out_file, '_warped', True))
    antsReg.inputs.output_transform_prefix = os.path.join(tempDir, 'transform_')

    antsReg.inputs.moving_image = fltout.outputs.out_file
    antsReg.inputs.fixed_image  = diffusionMapPath
    antsReg_out = antsReg.run()
    
    # apply transform to atlas
    antsAppTrfm = ants.ApplyTransforms()
    antsAppTrfm.inputs.dimension = 3
    antsAppTrfm.inputs.interpolation = 'NearestNeighbor'
    antsAppTrfm.inputs.default_value = 0

    antsAppTrfm.inputs.input_image = applyfltout.outputs.out_file
    antsAppTrfm.inputs.reference_image = diffusionMapPath
    antsAppTrfm.inputs.transforms = antsReg_out.outputs.reverse_forward_transforms
    antsAppTrfm.inputs.invert_transform_flags = antsReg_out.outputs.reverse_forward_invert_flags
    antsAppTrfm.inputs.output_image = os.path.join(outDir, appendToBasename(args.segment[0], '_registeredToPatient', True, '.nii.gz'))

    applyout = antsAppTrfm.run()
    regsiteredAtlas = applyout.outputs.output_image
    print('The registered atlas was generated ans saved here:{}'.format(regsiteredAtlas))

    print('Cleaning up intermediate products to save storage...')
    shutil.rmtree(tempDir)

    return regsiteredAtlas


def make_average_arr(diffusionMapPath, atlas_path):
    diffusionMap = nib.load(diffusionMapPath)
    atlas = nib.load(atlas_path)
    diffusionMap_array = diffusionMap.get_fdata()
    atlas_array = atlas.get_fdata() 

    uniq_structure_indices = np.unique(atlas_array)
    maxSegVal = max(uniq_structure_indices)

    # Prepare an empty list to hold the results
    results = []

    # Iterate through all unique structure indices (starting from 1 since 0 is null)
    for s in range(1, int(maxSegVal) + 1):
        if s not in uniq_structure_indices:
            results.append([s, 0.0])  # For missing structures, store 0.0
            continue

        atlas_indices = atlas_array == s
        matrix = diffusionMap_array[atlas_indices]
        avg_value = np.average(matrix)
        results.append([s, avg_value])

    results_array = np.array(results)

    return results_array

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

# save diffusion maps in tabular format
def collectDiffusionMapPerROI(args, outDir):
    fa_suffix = 'dti_fa_gqiscalar.nii.gz'
    fa_path = None
    for file in os.listdir(outDir):
        if fa_suffix in file:
            fa_path = os.path.join(outDir, file)

    if fa_path == None:
        raise ValueError("DTI FA map is not within the users outDirectory. Did \
            you run preprocess yor DWI? If so, check the file path name and \
            ensure it ends with {}".format(fa_suffix))

    # Register the atlas to the FA map
    regsiteredAtlas = registerAtlasToMap(args, outDir, fa_path)

    # Recenter all maps based on FA center, and reuse the FA registered atlas
    # NOTE: This logic only works assuming all diffusion maps are in the same space at the start
    maps = [os.path.join(outDir, image) for image in os.listdir(outDir) if '_gqiscalar.nii.gz' in image]
    averages = []
    for image in maps:
        avg_array = make_average_arr(image, regsiteredAtlas)
        modality = extract_modality_suffix(image)
        output_csv_path = os.path.join(outDir, f"{modality}_averages.csv")
        print(f'Saving per ROI averages for {modality} map to {output_csv_path}')
        np.savetxt(output_csv_path, avg_array, delimiter=',', header=f'ROI,{modality}', comments='', fmt='%d,%f')
        averages.append(output_csv_path)

    return averages

def consolidateFeatures(args, outDir, csv_files):

    combined_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)

        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.merge(combined_df, df, on='ROI', how='outer')

    out_file = 'consolidatedFeatures.csv'
    out_path = os.path.join(outDir, out_file)
    combined_df.to_csv(out_path, index=False)
    print(f"Combined CSV of MSN features saved to {out_path}")

    return out_path


def main():

    ################################################################################
    ### PREPWORK
    ################################################################################
    parser = makeParser()
    args   = parser.parse_args()
    args   = vet_inputs(args)
    data_dir      = os.path.abspath(os.path.expanduser(args.parentDir[0]))
    outDir        = ''
    outDirName    = 'MSNForge'
    session       = vetArgNone(args.session_id, None)
    template_path = vetArgNone(args.template, '/app/Template/MNI152lin_T1_2mm_brain.nii.gz') #path in docker container
    segment_path  = vetArgNone(args.segment, '/app/Template/aal2.nii.gz') #path in docker container
    enforceBIDS   = True

    #patient specific outDir
    outDir = makeOutDir(outDirName, args)

    if args.testmode:
        print("!!YOU ARE USING TEST MODE!!")


    ################################################################################
    ### WORK WORK
    ################################################################################

    # Move all stuff from T1/BOLD/DWI to new dir
    copyModalityOutputsToForgery(args, outDir)

    # register atlas to diffusion maps and get ROI averages
    dti_averages = collectDiffusionMapPerROI(args, outDir)

    # TO DO: if BOLD exists, calculate and output network stats

    # get struct features
    target_files = ['_trans_radiomicsFeatures.csv', '_trans_volumes.csv']
    struct_files = []
    for i in os.listdir(outDir):
        if any(target_file in i for target_file in target_files):
            struct_files.append(os.path.join(outDir,i))

    # combine all csvs into one table
    features = struct_files + dti_averages
    consolidateFeatures(args, outDir, features)

    # # take in fields user wants through a file

    # # make a default file optinos (set 'custom' if you want to give your own file)

    # # set similarity to pearson/euclidean/cosine

    # # output name features_similarit.csv
    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)