"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path

from setuptools import setup


setup(name='example-wise-f1-maximizer',
      version='0.1.0',
      description='A scikit-learn meta-estimator for multi-label classification that aims to maximize the example-wise '
                  'F1 measure',
      long_description=(Path(__file__).resolve().parent.parent / 'README.md').read_text(),
      long_description_content_type='text/markdown',
      author='Michael Rapp',
      author_email='michael.rapp.ml@gmail.com',
      url='https://github.com/mrapp-ke/ExampleWiseF1Maximizer',
      download_url='https://github.com/mrapp-ke/ExampleWiseF1Maximizer/releases',
      project_urls={
          'Issue Tracker': 'https://github.com/mrapp-ke/ExampleWiseF1Maximizer/issues',
      },
      license='MIT',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      keywords=[
          'machine learning',
          'scikit-learn',
          'multi-label classification'
      ],
      platforms=['any'],
      python_requires='>=3.7',
      install_requires=[
          'numpy >= 1.24, < 1.25',
          'scipy >= 1.10, < 1.11',
          'scikit-learn >=1.2, < 1.3'
      ],
      packages=['example_wise_f1_maximizer'],
      package_dir={'': 'src'},
      zip_safe=True)
