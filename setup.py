from __future__ import absolute_import
from setuptools import setup, find_packages

exec (open('csbdeep/version.py').read())

setup(name='csbdeep',
      version=__version__,
      description='CSBDeep - a toolbox for Content-aware Image Restoration (CARE)',
      url='https://github.com/csbdeep/csbdeep',
      author='Uwe Schmidt, Martin Weigert',
      author_email='uschmidt@mpi-cbg.de, mweigert@mpi-cbg.de',
      license='BSD 3-Clause License',
      packages=find_packages(),

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
      ],

      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "six",
          "keras>=2.0.7",
          "tifffile",
          "tqdm",
          "pathlib2;python_version<'3'",
          "backports.tempfile;python_version<'3.4'",
      ],

      entry_points={
          'console_scripts': [
              'care_predict = csbdeep.scripts.care_predict:main'
          ]
      }
      )
