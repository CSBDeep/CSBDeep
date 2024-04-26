# Extra resources 


* Denoising/Upsampling 2D example colab notebook [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/csbdeep/csbdeep/blob/main/extras/care_example_denoising_upsampling_2D_colab.ipynb)

## Conda environment

If you use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/) and have a CUDA-compatible GPU, you can install an environment with `csbdeep` via the provided environment file(s).

- Python 3.7 and TensorFlow 2.3:

  ```console
  $ conda env create -f https://raw.githubusercontent.com/CSBDeep/CSBDeep/main/extras/environment-gpu-py3.7-tf2.3.yml
  ```

- Python 3.8 and TensorFlow 2.4:

  ```console
  $ conda env create -f https://raw.githubusercontent.com/CSBDeep/CSBDeep/main/extras/environment-gpu-py3.8-tf2.4.yml
  ```
