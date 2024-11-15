#bin/python3
import argparse
import bct
import json
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
from scipy.stats import zscore

from registration import registerAtlasToMap
from utils import split_full_extension
from utils import appendToBasename
from utils import extract_modality_suffix

# Define similarity functions
def pearson_similarity(a, b):
    """Compute Pearson correlation coefficient between two vectors."""
    return np.corrcoef(a, b)[0, 1]

def cosine_similarity(a, b):
    """Compute Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def inverse_euclidean_similarity(a, b):
    """Compute Inverse Euclidean distance between two vectors."""
    return 1 / (1 + np.linalg.norm(a - b))

similarity_functions = {
        'pearsonr': pearson_similarity,
        'cosine': cosine_similarity,
        'inverse_euclidean': inverse_euclidean_similarity
    }

def makeParser():
    parser = argparse.ArgumentParser(
                        prog='MSNForge', 
                        usage='This program is a continuation of MSNger and helps to build the MSNs after preprocessing the scans',
                        epilog='BUG REPORTING: Report bugs to pirc@chp.edu or more directly to Joy Roy at the Childrens Hospital of Pittsburgh.'
        )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-p','--parentDir', nargs=1, required=True,
                        help='Path to the parent data directory. BIDS compatible datasets are encouraged.')
    parser.add_argument('-sid','--subject_id', nargs=1, required=False,
                        help='Subject ID used to indicate which patient to preprocess')
    parser.add_argument('-ses_id','--session_id', nargs=1, required=False,
                        help='Session ID used to indicate which session to look for the patient to preprocess')
    parser.add_argument('-tem','--template', nargs=1, required=False,
                        help='Template to be used to register into patient space. Default is MNI152lin_T1_2mm_brain.nii.gz')
    parser.add_argument('-seg','--segment', nargs=1, required=False,
                        help='Atlas to be used to identify brain regions in patient space. This is used in conjunction with the template. \
                        Please ensure that the atlas is in the same space as the template. Default is the AALv2 template.')
    parser.add_argument('-o','--outDir', nargs=1, required=True,
                        help='Path to the \'derivatives\' folder or chosen out folder.')
    parser.add_argument('--features', required=False, choices=['all','vol_radb_diff','custom'], default='all', 
                        help='Specify which set of features to include in the MSN: '
                        '"all" for all available features, or "custom" to provide your own list of features. '
                        'See the "features.json" file for a list of available features and an example of how to structure the custom file. '
                        'Use "custom" if you want to specify a custom set of features via the --customFeatureFile.')
    parser.add_argument('--featureFile', nargs=1, required=False, 
                        help='JSON file containing a custom list of features to include in the MSN. '
                        'This argument is required if --features is set to "custom".')
    parser.add_argument('--similarity_measure', required=False, choices=['pearsonr', 'cosine', 'inverse_euclidean'], default='pearsonr', 
                        help='Specify the similarity measure to use for computing similarity between ROIs. Options: "pearsonr" (default).')
    parser.add_argument('--testmode', required=False, action='store_true',
                        help='Activates TEST_MODE to make pipeline finish faster for quicker debugging')
    parser.add_argument('--supersizeme', required=False, action='store_true',
                        help='Get every combination of MSN from the default features list')
    parser.add_argument('--animalstyle', required=False, action='store_true',
                        help='Get every combination of MSN from the similarity measure, includes custom feature files')

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

    # configure the desired features for the MSN
    if args.features == 'custom':
        if args.featureFile is None or args.featureFile[0] is None:
            raise ValueError("Custom featureFile cannot be None if features is set to 'custom'.")
        else:
            if not os.path.exists(args.featureFile[0]): 
                raise ValueError("Path to custom featureFile does not exist.")
            else:
                args.featureFile = [os.path.abspath(os.path.expanduser(args.featureFile[0]))]
    else:
        if args.featureFile is not None:
            print("WARNING: featureFile was provided, but the feature flag is not set to 'custom'. ,Ignoring featureFile.")
        args.featureFile = ['/app/Template/features.json']

    
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


def copy_modality_outputs(modality_path, out_dir, keywords=None, prefix=None):
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
                if prefix is None:
                    destination = os.path.join(out_dir, file)
                else:
                    destination = os.path.join(out_dir, prefix+file)
                shutil.copy2(source, destination)
                print(f"Copied {file} to {destination}")

    print("Copy operation complete.\n")
    return 0

def copyModalityOutputsToForge(args, outDir):
    outDirPath = Path(outDir)
    basepath = outDirPath.parents[2]

    successes = {}

    print('Copying T1 outputs...')
    t1Path = os.path.join(basepath, 'RadT1cal_Features', args.subject_id[0])
    successes['T1'] = copy_modality_outputs(t1Path, outDir, ['_trans_radiomicsFeatures.csv', '_trans_volumes.csv'])

    print('Copying BOLD outputs...')
    boldPath = os.path.join(basepath, 'Sim_Funky_Pipeline', args.subject_id[0])
    successes['BOLD'] = copy_modality_outputs(boldPath, outDir, ['sim_matrix.csv'], prefix='func_')

    print('Copying DTI outputs...')
    dtiPath = os.path.join(basepath, 'qsirecon', args.subject_id[0])
    successes['DTI'] = copy_modality_outputs(dtiPath, outDir, ['_gqiscalar.nii.gz'])

    return successes


def make_average_arr(diffusionMapPath, atlas_path):
    print("Calculating averages per roi in the diffusion maps...")
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

# save diffusion maps in tabular format
def collectDiffusionMapPerROI(args, outDir):
    print('\nCollecting diffusion map roi features...')
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

def saveNetworkProp(args, outDir, networkProp_name, networkProp_values, ROIs):
    df = pd.DataFrame({
        'ROI': ROIs,
        f'{networkProp_name}': networkProp_values
    })

    # Save DataFrame to CSV
    output_csv_path = os.path.join(outDir, f'{networkProp_name}.csv')
    df.to_csv(output_csv_path, index=False)
    return output_csv_path


def calculateFuncNetworkProperties(args, outDir):
    print('\nCalculating functional network properties...')
    sim_matrix_file = 'func_sim_matrix.csv'
    sim_matrix_path = os.path.join(outDir, sim_matrix_file)
    sim_matrix = np.loadtxt(sim_matrix_path, delimiter=",") 
    sim_matrix = np.abs(sim_matrix) # symmetric use of negative edge weights #maybe a config?

    atlas = nib.load(args.segment[0])
    atlas_array = atlas.get_fdata() 
    uniq_structure_indices = np.unique(atlas_array)
    maxSegVal = int(max(uniq_structure_indices))
    ROIs = list(range(1, maxSegVal+1))

    outfiles = []

    # Define properties and functions to calculate them
    properties = [
        ('clustering_coefficient', lambda sm: bct.clustering_coef_wu(sm)),
        ('node_betweeness', lambda sm: bct.betweenness_wei(sm)),
        ('eigenvector_centrality', lambda sm: bct.eigenvector_centrality_und(sm)),
        ('participation_coef', lambda sm: bct.participation_coef(sm, bct.community_louvain(sm, B='negative_sym')[0])),
        ('degree_at_abs50', lambda sm:bct.degrees_und(bct.threshold_absolute(np.abs(sm), 0.50))),
        ('degree_at_pro50', lambda sm:bct.degrees_und(bct.threshold_proportional(np.abs(sm), 0.50))),
        ('degree_at_abs25', lambda sm:bct.degrees_und(bct.threshold_absolute(np.abs(sm), 0.25))),
        ('degree_at_pro25', lambda sm:bct.degrees_und(bct.threshold_proportional(np.abs(sm), 0.25))),
    ]

    # Calculate each network property and save to CSV
    outfiles = []
    for prop_name, func in properties:
        networkProp_values = func(sim_matrix)
        outfile = saveNetworkProp(args, outDir, prop_name, networkProp_values, ROIs)
        print(f'Computing and saving {prop_name} of functional network to {outfile}')
        outfiles.append(outfile)

    return outfiles



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


def zscore_features(features_path, outDir):
    df = pd.read_csv(features_path)
    
    # Ensure 'ROI' column exists and separate it from columns to be z-scored
    if 'ROI' not in df.columns:
        raise ValueError("The input CSV must have an 'ROI' column.")
    
    # Apply z-score normalization to all columns except 'ROI'
    zscored_df = df.copy()
    zscored_df.loc[:, zscored_df.columns != 'ROI'] = df.loc[:, df.columns != 'ROI'].apply(zscore)
    
    outpath = os.path.join(outDir, appendToBasename(features_path, '_standardized', True))
    zscored_df.to_csv(outpath, index=False)
    print(f'Features in {features_path} were standardized and saved to {outpath}')
    return outpath

def getFeatureSets(config_file):
    # Load the JSON data
    with open(config_file, 'r') as file:
        data = json.load(file)

    available_feature_sets = list(data.keys())
    print(f"Available feature sets: {available_feature_sets}")
    return available_feature_sets

def load_features(config_file, feature_set=None):
    # Load the JSON data
    with open(config_file, 'r') as file:
        data = json.load(file)

    if feature_set in data:
        # Return the selected features list
        print(f"Desired features from '{feature_set}' to build MSN: {data[feature_set]}")
        return (feature_set, data[feature_set])
    else:
        raise ValueError(f"Feature set '{feature_set}' not found in the configuration file.")

def vet_desired_features(desired_features, available_features):
    available_columns = set(pd.read_csv(available_features).columns)
    desired_columns = set(desired_features)
    missing_columns = desired_columns - available_columns
    if len(missing_columns) == 0:
        return True
    else:
        raise ValueError(f"The following desired features are missing from available features: {missing_columns}")


def computeMSN(args, outDir, consolidated_standardized_path, desiredFeaturesTuple, similarity_function='pearsonr'):
    print("COMPUTING MSN")
    df = pd.read_csv(consolidated_standardized_path)
    
    # Extract, Z-score, and select desired features
    roi = df['ROI']
    features = df.drop(columns=['ROI'])
    desiredFeatures = features[desiredFeaturesTuple[1]]

    num_rois = desiredFeatures.shape[0]
    sim_matrix = np.zeros((num_rois, num_rois))

    # Choose similarity function from dictionary if specified in args
    similarity_func = similarity_functions.get(similarity_function, pearson_similarity)

    # Compute similarity between every pair of ROIs
    for i in range(num_rois):
        for j in range(i + 1, num_rois):  # Only compute upper triangular matrix
            sim = similarity_func(desiredFeatures.iloc[i].values, desiredFeatures.iloc[j].values)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim  # Matrix is symmetric

    np.fill_diagonal(sim_matrix, 0.)
    if args.features == 'custom':
        sim_matrix_path = os.path.join(outDir, f'MSN_feats-custom-{desiredFeaturesTuple[0]}_simfunc-{similarity_function}.csv')
    else:
        sim_matrix_path = os.path.join(outDir, f'MSN_feats-{desiredFeaturesTuple[0]}_simfunc-{similarity_function}.csv')

    np.savetxt(sim_matrix_path, sim_matrix, delimiter=",")
    print(f'Saving MSN to {sim_matrix_path}')
    

def miniMSNPipeline(args, outDir, consolidated_standardized_path, featureSet, similarity_function='pearsonr'):
    # Load and validate desired features
    desiredFeatures = load_features(args.featureFile[0], feature_set=featureSet)
    check = vet_desired_features(desiredFeatures[1], consolidated_standardized_path)
    # Compute MSN
    computeMSN(args, outDir, consolidated_standardized_path, desiredFeatures, similarity_function)
    print() #empty print intentional


def main():

    ################################################################################
    ### PREPWORK
    ################################################################################
    parser = makeParser()
    args   = parser.parse_args()
    args   = vet_inputs(args)
    outDir        = ''
    outDirName    = 'MSNForge'
    enforceBIDS   = True

    #patient specific outDir
    outDir = makeOutDir(outDirName, args)

    if args.testmode:
        print("!!YOU ARE USING TEST MODE!!")

    ################################################################################
    ### WORK WORK
    ################################################################################

    # # Move all stuff from T1/BOLD/DWI to new dir
    successes = copyModalityOutputsToForge(args, outDir)

    # get struct features
    struct_files = []
    if not successes['T1'] < 0:
        target_files = ['_trans_radiomicsFeatures.csv', '_trans_volumes.csv']
        for i in os.listdir(outDir):
            if any(target_file in i for target_file in target_files):
                struct_files.append(os.path.join(outDir,i))

    # # register atlas to diffusion maps and get ROI averages
    dti_averages = []
    if not successes['DTI'] < 0:
        dti_averages = collectDiffusionMapPerROI(args, outDir)


    # calculate and output network stats
    bold_networkprops = []
    if not successes['BOLD'] < 0:
        bold_networkprops = calculateFuncNetworkProperties(args, outDir)

    # combine all csvs into one table
    availableFeatures = struct_files + dti_averages + bold_networkprops
    consolidated_path = consolidateFeatures(args, outDir, availableFeatures)

    # standardize per column
    consolidated_standardized_path = zscore_features(consolidated_path, outDir)

    featureSet = getFeatureSets(args.featureFile[0])
    errors = []
    if args.animalstyle:
        for j in similarity_functions.keys():
            if args.supersizeme:
                for i in featureSet:
                    try:
                        miniMSNPipeline(args, outDir, consolidated_standardized_path, i, j)
                    except Exception as e:
                        errors.append(f"Error with feature '{i}' and similarity '{j}': {str(e)}")
            else:
                selected_feature = args.features if args.features != 'custom' else featureSet[0]
                try:
                    miniMSNPipeline(args, outDir, consolidated_standardized_path, selected_feature, j)
                except Exception as e:
                    errors.append(f"Error with feature '{selected_feature}' and similarity '{j}': {str(e)}")
    else:
        if args.supersizeme:
            for i in featureSet:
                try: 
                    miniMSNPipeline(args, outDir, consolidated_standardized_path, i, args.similarity_measure)
                except Exception as e:
                    errors.append(f"Error with feature '{i}' and similarity '{j}': {str(e)}")
        else:
            selected_feature = args.features if args.features != 'custom' else featureSet[0]
            try:
                miniMSNPipeline(args, outDir, consolidated_standardized_path, selected_feature, args.similarity_measure)
            except Exception as e:
                errors.append(f"Error with feature '{selected_feature}' and similarity '{args.similarity_measure}': {str(e)}")

    if errors:
        raise Exception("Errors encountered:\n" + "\n".join(errors))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)