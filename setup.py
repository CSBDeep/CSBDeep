from __future__ import absolute_import
from setuptools import setup, find_packages


setup(name='csbdeep',
      version="0.1.0",
      author='CSBDeep Team',
      author_email='...',
      license='BSD 3-Clause License',
      packages=find_packages(),

      install_requires=["keras>=2.0.7",
                        "tqdm",
                        "tifffile",
                        "six",
                        "pathlib2;python_version<'3'",
                        "numpy",
                        "matplotlib",
                        # "tensorflow-gpu>=1.2.0",
                        # "pandas>=0.20.1",
                        ],
      entry_points={}
      )