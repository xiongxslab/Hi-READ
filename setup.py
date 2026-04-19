#!/usr/bin/env python3
"""Setup script for Hi-READ package."""

from setuptools import setup, find_packages
import os

def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

INSTALL_REQUIRES = [
    'numpy>=1.19.0',
    'torch>=1.12.0',
    'lightning>=2.0.0',
    'pytorch-lightning>=1.6.0',
    'torchvision>=0.13.0',
    'scipy>=1.7.0',
    'pandas>=1.3.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'scikit-learn>=1.0.0',
    'pyBigWig>=0.3.18',
    'pyfaidx>=0.7.0',
    'cooler>=0.8.11',
    'h5py>=3.1.0',
    'tqdm>=4.62.0',
    'pyyaml>=5.4.0',
    'scikit-image>=0.19.0',
    'einops>=0.4.0',
    'timm>=0.6.0',
    'biopython>=1.79',
    'denoising-diffusion-pytorch>=2.0.0',
]

EXTRAS_REQUIRE = {
    'workflows': [
        'scikit-learn>=0.24.0',
        'umap-learn>=0.5.0',
        'hdbscan>=0.8.27',
    ],
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'black>=21.6b0',
        'flake8>=3.9.0',
        'mypy>=0.910',
    ],
}

setup(
    name='hiread',
    version='1.0.0',
    description='Hi-READ: High-resolution Regulatory Element Analysis and Design',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Hi-READ Team',
    author_email='',
    url='https://github.com/your-org/Hi-READ',
    packages=find_packages(include=['hiread', 'hiread.*', 'hiread_diffusion', 'hiread_diffusion.*']),
    python_requires='>=3.8',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='genomics deep-learning hi-c chromatin epigenetics',
    project_urls={
        'Bug Reports': 'https://github.com/your-org/Hi-READ/issues',
        'Source': 'https://github.com/your-org/Hi-READ',
    },
)
