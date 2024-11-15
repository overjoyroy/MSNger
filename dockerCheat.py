import os
from python_on_whales import docker
import argparse
from tqdm import tqdm
import json

def makeParser():
    parser = argparse.ArgumentParser(
                        prog='MSNger', 
                        usage='This program preprocesses T1 MRI, rsfMRI, and DTIs in order to build an MSN.',
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
                        help='Atlas to be used to identify brain regions in patient space. This is used in conjunction with the template. Please ensure that the atlas is in the same space as the template. Default is the AALv2 template.')
    parser.add_argument('-o','--outDir', nargs=1, required=True,
                        help='Path to the \'derivatives\' folder or chosen out folder. All results will be submitted to outDir/out/str_preproc/subject_id/...')
    parser.add_argument('--fslicense', nargs=1, required=True,
                        help='Path to freesurfer license. Required for Qsiprep to work')
    parser.add_argument('--batch_whole_dataset', required=False, action='store_true',
                        help='Flag to process every patient and scan in the entire dataset.')
    # parser.add_argument('--do_not_preprocess', required=False, action='store_true',
    #                     help='Flag to skip preprocessing scans.')
    parser.add_argument( '--preprocess_only', required=False, choices=['all', 'none', 'dwi', 't1', 'bold'], default='all', 
                        help='Specify which modality to preprocess: "all", "none", or a particular modality (e.g., "dwi", "t1", "bold").')
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
    parser.add_argument('--skipforge', required=False, action='store_true',
                        help='Skips the MSN generation. Use only if you want to focus on preprocessing')
    parser.add_argument('--supersizeme', required=False, action='store_true',
                        help='Get every combination of MSN from the default features list')
    parser.add_argument('--animalstyle', required=False, action='store_true',
                        help='Get every combination of MSN from the similarity measure, includes custom feature files')
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
    
    # Convert Freesurfer license path to absolute if it's relative
    if args.fslicense:
        args.fslicense = [os.path.abspath(os.path.expanduser(args.fslicense[0]))]
    
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
    if not args.batch_whole_dataset:
        if not args.subject_id or not isinstance(args.subject_id[0], str):
            raise ValueError("Subject ID is missing or invalid. It should be in the form 'sub-#+' ")
    
        # this isn't needed if batch
        # Validate session ID if provided
        if args.session_id and not isinstance(args.session_id[0], str):
            raise ValueError("Session ID is invalid. It should be in the form 'ses-#+' ")
        else:
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

def preprocess_T1w(args):
    print(args)
    try:
        docker.run("jor115/t1proc",
                   interactive=True,
                   tty=True,
                   remove=True,
                   user="{}:{}".format(os.getuid(), os.getgid()),
                   platform="linux/amd64",
                   volumes=[(args.parentDir[0], "/data"), (args.outDir[0], "/out")],
                   command=["-p", "/data", "-o", "/out",
                            "-sid", args.subject_id[0], 
                            "--session_id", args.session_id[0]]
                   )
        return 0
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the container...")
        return 1
    except Exception as e: 
        print("Error, patient T1w couldn't be processed...")
        print(e)
        return 1


def preprocess_rsfMRI(args):
    try:
        docker.run("jor115/sfp",
                   interactive=True,
                   tty=True,
                   remove=True,
                   user="{}:{}".format(os.getuid(), os.getgid()),
                   platform="linux/amd64",
                   volumes=[(args.parentDir[0], "/data"), (args.outDir[0], "/out")],
                   command=["-p", "/data", "-o", "/out",
                            "-sid", args.subject_id[0], 
                            "--session_id", args.session_id[0]]
                   )
        return 0
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the container...")
        return 1
    except:
        print("Error, patient BOLD couldn't be processed...")
        return 1

def preprocess_DTI(args):
    try:
        docker.run("pennbbl/qsiprep:0.20.0", ## last version to package qsiprep and qsirecon together
                   # name="qsiprep_container",
                   interactive=True,
                   tty=True,
                   remove=True,
                   user="{}:{}".format(os.getuid(), os.getgid()),
                   platform="linux/amd64",
                   volumes=[(args.parentDir[0], "/data"), (args.outDir[0], "/out"), (args.fslicense[0], "/opt/freesurfer/license.txt"), ("./qgi_scalar_export.json", "/temp/qgi_scalar_export.json")],
                   command=["/data", "/out", "participant",
                            "--participant_label", args.subject_id[0],
                            "--skip_bids_validation", 
                            "--recon_input", "/out/qsiprep",
                            "--recon_spec", "/temp/qgi_scalar_export.json",
                            "--fs-license-file", "/opt/freesurfer/license.txt",
                            "--output-resolution", "2"]
                   )

        return 0
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the container...")
        return 1
    except:
        print("Error, patient DWI couldn't be processed...")
        return 1

def preprocess(args):
    if args.preprocess_only == 't1':
        preprocess_T1w(args)
    if args.preprocess_only == 'dwi':
        preprocess_DTI(args)
    if args.preprocess_only == 'bold':
        preprocess_rsfMRI(args)
    if args.preprocess_only == 'all':
        preprocess_T1w(args)
        preprocess_rsfMRI(args)
        preprocess_DTI(args)
    
def itsforgingtime(args):
    try:

        command = [
            "-p", "/data", 
            "-o", "/out",
            "-sid", args.subject_id[0], 
            "--session_id", args.session_id[0],
            "--features", args.features
        ]
        # Add featureFile if provided
        if args.featureFile:
            command.extend(["--featureFile", args.featureFile[0]])

        # Check and add the supersizeme flag if it is set
        if args.supersizeme:
            command.append("--supersizeme")

        # Check and add the animalstyle flag if it is set
        if args.animalstyle:
            command.append("--animalstyle")

        docker.run("jor115/msnforge",
                   interactive=True,
                   tty=True,
                   remove=True,
                   user="{}:{}".format(os.getuid(), os.getgid()),
                   platform="linux/amd64",
                   volumes=[(args.parentDir[0], "/data"), (args.outDir[0], "/out")],
                   command=command

                   )

        return 0
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the container...")
        return 1
    except Exception as e:
        print(f"Error, patient MSNger couldn't complete... {e}")
        return 1


def batch_process_whole_dataset(args):
    processing_error = []
    print('Batch preprocessing in place!')
    for subj in tqdm(os.listdir(args.parentDir[0])):
        if 'sub-' in subj[:4]:
            args.subject_id = [subj] # set subject
            subjdir = os.path.join(args.parentDir[0], subj)
            for sess in os.listdir(subjdir):
                if 'ses-' in sess:
                    args.session_id = [sess] # set session
                    res = 0

                    if not args.preprocess_only == 'none':
                        print('Preprocessing: {}-{}'.format(args.subject_id[0], args.session_id[0]))
                        res = preprocess(args)
                    else: 
                        print("Skipping all image preprocessing...")

                    if not args.skipforge:
                        res = itsforgingtime(args)
                    else:
                        print("Skipping forging...")

                    if res !=0:
                        processing_error.append(json.dumps(vars(args), indent=2))

                    fillerstring = '#'
                    print(fillerstring * 64)

    return processing_error

            

def main():
    parser = makeParser()
    args   = parser.parse_args()
    args   = vet_inputs(args)
    print("Arguments given: {}".format(args))

    res = 0
    if args.batch_whole_dataset:
        errors = batch_process_whole_dataset(args)
        print(f"Errors with the following args :")
        for index, err in enumerate(errors):
            print(f'\t{index}:{err}')

    else:
        if not args.preprocess_only == 'none':
            print('Preprocessing: {}-{}'.format(args.subject_id[0], args.session_id[0]))
            res = preprocess(args)
        else: 
            print("Skipping all image preprocessing...")

        if not args.skipforge:
            res = itsforgingtime(args)
        else:
            print("Skipping forging...")

    if res !=0:
        print(f"Errors with the following args {args}")
    print("Done!")





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)