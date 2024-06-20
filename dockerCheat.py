import os
from python_on_whales import docker

try:
    docker.run("pennbbl/qsiprep:0.20.0",
               name="qsiprep_container",
               interactive=True,
               tty=True,
               remove=True,
               user="{}:{}".format(os.getuid(), os.getgid()),
               platform="linux/amd64",
               volumes=[("./test_svr/svr", "/data"), ("./test_svr/out", "/out"), ("./test_svr/workingdir", "/workingdir"), ("./test_svr/license.txt", "/opt/freesurfer/license.txt"), ("./test_svr/qgi_scalar_export.json", "/workingdir/qgi_scalar_export.json")],
               command=["/data", "/out", "participant",
               			"--participant_label", "sub-00000107",
                        "--recon_input", "/out/qsiprep",
                        "--recon_spec", "/workingdir/qgi_scalar_export.json",
                        "--fs-license-file", "/opt/freesurfer/license.txt",
                        "--output-resolution", "1"]
               )
except KeyboardInterrupt:
    print("Keyboard interrupt received. Stopping the container...")
    docker.stop("qsiprep_container")