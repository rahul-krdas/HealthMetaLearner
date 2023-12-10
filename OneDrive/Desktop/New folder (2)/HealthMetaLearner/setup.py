# setup.py

from setuptools import setup, find_packages

setup(
    name='HealthMetaLearner',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'torch',
        # Add any additional dependencies here
    ],
    entry_points={
        'console_scripts': [
            # If your package includes command-line scripts, define them here
        ],
    },
)
