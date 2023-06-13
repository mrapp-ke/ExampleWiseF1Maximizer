"""
Author: Michael Rapp (michael.rapp.ml@gmail.com)
"""
from pathlib import Path

from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def find_dependencies(requirements_file, dependency_names):
    requirements = {
        requirement.key: requirement
        for requirement in parse_requirements(requirements_file.read_text().split('\n'))
    }
    dependencies = []

    for dependency_name in dependency_names:
        match = requirements.get(dependency_name)

        if match is None:
            raise RuntimeError(
                'Failed to determine required version of dependency "' + dependency_name + '"')

        dependencies.append(str(match))

    return dependencies


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
          *find_dependencies(requirements_file=Path(__file__).resolve().parent / 'requirements.txt',
                             dependency_names=['numpy', 'scipy', 'scikit-learn']),
      ],
      packages=find_packages(),
      zip_safe=True)
