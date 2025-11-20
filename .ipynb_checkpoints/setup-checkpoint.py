from setuptools import setup, find_packages

setup(
    name='exseq_kit',
    version='0.1',
    description='Toolkit for ExSeq-based spatial transcriptomics analysis',
    author='Moshe Shenhav',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'scikit-learn'
    ],
    include_package_data=True,
    zip_safe=False
)
