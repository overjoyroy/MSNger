import os
from python_on_whales import docker
import argparse

def makeParser():
    parser = argparse.ArgumentParser(
                        prog='MSNger', 
                        usage='This program preprocesses T1 MRI, rsfMRI, and DTIs in order to build an MSN.',
                        epilog='BUG REPORTING: Report bugs to pirc@chp.edu or more directly to Joy Roy at the Childrens Hospital of Pittsburgh.'
        )
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    parser.add_argument('-p','--parentDir', nargs=1, required=True,
                        help='Path to the parent data directory. BIDS compatible datasets are encouraged.')
    parser.add_argument('-sid','--subject_id', nargs=1, required=True,
                        help='Subject ID used to indicate which patient to preprocess')
    parser.add_argument('-spath','--subject_t1_path', nargs=1, required=False,
                        help='Path to a subjects T1 scan. This is not necessary if subject ID is provided as the T1 will be automatically found using the T1w.nii.gz extension')
    parser.add_argument('-ses_id','--session_id', nargs=1, required=False,
                        help='Session ID used to indicate which session to look for the patient to preprocess')
    parser.add_argument('-tem','--template', nargs=1, required=False,
                        help='Template to be used to register into patient space. Default is MNI152lin_T1_2mm_brain.nii.gz')
    parser.add_argument('-seg','--segment', nargs=1, required=False,
                        help='Atlas to be used to identify brain regions in patient space. This is used in conjunction with the template. Please ensure that the atlas is in the same space as the template. Default is the AALv3 template.')
    parser.add_argument('-o','--outDir', nargs=1, required=True,
                        help='Path to the \'derivatives\' folder or chosen out folder. All results will be submitted to outDir/out/str_preproc/subject_id/...')
    parser.add_argument('--testmode', required=False, action='store_true',
                        help='Activates TEST_MODE to make pipeline finish faster for quicker debugging')

    return parser 

def preprocess(args):
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
                            "-sid", args.subject_id[0]]
                   )
    except KeyboardInterrupt:
        print("oops")
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received. Stopping the container...")
    #     docker.stop("sfp_container")

    # try:
    #     docker.run("jor115/t1proc",
    #                name="qsiprep_container",
    #                interactive=True,
    #                tty=True,
    #                remove=True,
    #                user="{}:{}".format(os.getuid(), os.getgid()),
    #                platform="linux/amd64",
    #                volumes=[("./test_svr/svr", "/data"), ("./test_svr/out", "/out"), ("./test_svr/workingdir", "/workingdir"), ("./test_svr/license.txt", "/opt/freesurfer/license.txt"), ("./test_svr/qgi_scalar_export.json", "/workingdir/qgi_scalar_export.json")],
    #                command=["/data", "/out", "participant",
    #                         "--participant_label", "sub-00000107",
    #                         "--recon_input", "/out/qsiprep",
    #                         "--recon_spec", "/workingdir/qgi_scalar_export.json",
    #                         "--fs-license-file", "/opt/freesurfer/license.txt",
    #                         "--output-resolution", "1"]
    #                )
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received. Stopping the container...")
    #     docker.stop("qsiprep_container")


    # try:
    #     docker.run("pennbbl/qsiprep:0.20.0",
    #                name="qsiprep_container",
    #                interactive=True,
    #                tty=True,
    #                remove=True,
    #                user="{}:{}".format(os.getuid(), os.getgid()),
    #                platform="linux/amd64",
    #                volumes=[("./test_svr/svr", "/data"), ("./test_svr/out", "/out"), ("./test_svr/workingdir", "/workingdir"), ("./test_svr/license.txt", "/opt/freesurfer/license.txt"), ("./test_svr/qgi_scalar_export.json", "/workingdir/qgi_scalar_export.json")],
    #                command=["/data", "/out", "participant",
    #                			"--participant_label", "sub-00000107",
    #                         "--recon_input", "/out/qsiprep",
    #                         "--recon_spec", "/workingdir/qgi_scalar_export.json",
    #                         "--fs-license-file", "/opt/freesurfer/license.txt",
    #                         "--output-resolution", "1"]
    #                )
    # except KeyboardInterrupt:
    #     print("Keyboard interrupt received. Stopping the container...")
    #     docker.stop("qsiprep_container")



def main():
    parser = makeParser()
    args   = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\tReceived Keyboard Interrupt, ending program.\n")
        sys.exit(2)