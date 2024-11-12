#bin/python3
import argparse
import nipype.interfaces.fsl as fsl 
import nipype.interfaces.ants as ants 
import os
import shutil

from utils import split_full_extension
from utils import appendToBasename


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