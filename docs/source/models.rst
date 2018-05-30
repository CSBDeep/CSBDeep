Currently supported restoration models 
=====================================

The following list provides an overview of currently supported restoration models that are tailored to commonly used imaging scenarios: 



:obj:`csbdeep.models.CARE`
^^^^^^^^^^^^^^^^^^^^^^^^^^


  Description:
    * Basic model that learns a mapping from input (degraded) to output (restored) stacks
    * Input/output can be 2D or 3D stacks.
    * Expects spatially registered input/output pairs

  Typical use-case:
    * Denoising of live-cell images (e.g. acquired with reduced laser power/exposure).
    * Improving SNR of fast time-lapses of vesicle trafficking images. 

	
  Examples:
    * ``notebooks/denoising3D``
    * ``notebooks/denoising2D_probabilistic``

:obj:`csbdeep.models.UpsamplingCARE`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Description:
    * A model that will additionally increase sampling along a given (eg. axial) dimension by a given factor ``s``.
    * Input/output should be registered 3D stacks with the desired pixel size.
    * After training, model is applied to lower resolution data producing output stacks with a ``s`` fold increased number sample planes. 

  Typical use-case:
    * Improving the axial resolution of volumetric time-lapses, when only a limited number of focal planes can be acquired.

  Examples:
    * ``notebooks/upsampling3D``

:obj:`csbdeep.models.IsotropicCARE`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  Description:
    * A model that improves axial resolution of (axially) anisotropic stacks.
    * Takes anisotropic 3D stacks as input (Important: doesn't need corresponding output stacks).
    * The PSF of the microscope has to be known.
    * Assumes isotropic distribution of biological structures (i.e. don't use it highly anisotropic tissue like cortical tissue).

  Typical use-case:
    * Enhancing axial resolution of light-sheet microscopy time-lapses of developing embryos.

  Examples:
    * ``notebooks/isotropic_reconstruction``

