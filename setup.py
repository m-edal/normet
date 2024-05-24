from setuptools import setup, find_packages

classifiers = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Atmospheric Science"
    ]

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name='normet',
    version="0.0.0",
    author="Dr. Congbo Song and other MEDAL group members",
    url='https://github.com/m-edal/normet',
    description='Normet for automated air quality intervention studies',
    long_description=long_description,
    license='MIT',
    packages=find_packages(),
    package_data={'normet': ['datasets/*/*']},
    classifiers=classifiers,
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'joblib',
        'flaml',
        'matplotlib',
        'seaborn',
        'ruptures',
        'scikit-learn',
        'statsmodels',
        'cdsapi',
        'pyreadr',
        'wget',
        'xarray'
    ],
    zip_safe=False)
