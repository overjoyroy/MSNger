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
        args.template = ['/app/Template/aal2.nii.gz']
    
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

def makeOutDir(outDirName, args, ):

    outDirPath_parents = args.outDir[0]
    if '~' in outDirPath_parents:
        outDirPath_parents = os.path.expanduser(outDirPath_parents)

    outDirPath_parents = os.path.abspath(outDirPath_parents)

    templateName =  os.path.basename(args.template[0]).split('.')[0] #/path/to/aal.nii.gz
    print('Template name appears to be: {}. This will be the name of the output subdirectory'.format(templateName))


    outDir = os.path.join(outDirPath_parents, outDirName, args.subject_id[0], templateName)

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
    outdirPath = Path(outDir)
    basepath = outdirPath.parents[2]

    print('Copying T1 outputs...')
    t1Path = os.path.join(basepath, 'RadT1cal_Features', args.subject_id[0])
    print('Copying BOLD outputs...')
    boldPath = os.path.join(basepath, 'Sim_Funky_Pipeline', args.subject_id[0])
    print('Copying DTI outputs...')
    dtiPath = os.path.join(basepath, 'qsirecon', args.subject_id[0])

    copy_modality_outputs(t1Path, outDir, ['_trans_radiomicsFeatures.csv', '_trans_volumes.csv'])
    copy_modality_outputs(boldPath, outDir, ['average_arr.csv', 'sim_matrix.csv'])
    copy_modality_outputs(dtiPath, outDir, ['_gqiscalar.nii.gz'])



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

    #patient specific outdir
    outDir = makeOutDir(outDirName, args)

    if args.testmode:
        print("!!YOU ARE USING TEST MODE!!")


    # move all stuff from T1/BOLD/DWI to new dir
    copyModalityOutputsToForgery(args, outDir)

    # register aal2 to dwi (only 1)

    # get ROI avg values for aal2 for all dti maps, save for csv

    # combine all csvs into one table

    # take in fields user wants through a file

    # make a default file optinos (set 'custom' if you want to give your own file)

    # set similarity to pearson/spearman/cosine

    # 
    


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)