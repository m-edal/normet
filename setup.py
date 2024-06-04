import subprocess
import sys

def install_with_pip(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "pandas",
    "numpy",
    "scipy",
    "joblib",
    "flaml",
    "matplotlib",
    "seaborn",
    "ruptures",
    "scikit-learn",
    "statsmodels",
    "cdsapi",
    "pyreadr",
    "wget",
    "xarray"
]

for package in required_packages:
    install_with_pip(package)



from setuptools import setup, find_packages

setup(
    name="normet",
    version="0.1.0",
    description="Normet for automated air quality intervention studies",
    long_description=open("README.rst","r",encoding="utf-8").read(),
    long_description_content_type="text/x-rst",
    author="Dr. Congbo Song and other MEDAL group members",
    license="MIT",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Atmospheric Science"
    ],
    keywords=[
        "Atmospheric Science", "Air Quality", "Machine Learning", "Causal Analysis"
    ],
    install_requires=[
        "pandas",
        "numpy",
        "scipy",
        "joblib",
        "flaml",
        "matplotlib",
        "seaborn",
        "ruptures",
        "scikit-learn",
        "statsmodels",
        "cdsapi",
        "pyreadr",
        "wget",
        "xarray"
    ],
    packages=find_packages(),
    package_data={
        "normet": ["datasets/*/*"]
    },
    zip_safe=False,
    project_urls={
        "homepage": "https://github.com/dsncas/NORmet"
    },
)
