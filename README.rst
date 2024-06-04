normet
======

**normet** is a Python package to conduct automated data curation, automated machine learning-based meteorology/weather normalisation and causal analysis on air quality interventions for atmospheric science, air pollution and policy analysis. The main aim of this package is to provide a Swiss army knife enabling rapid automated-air quality intervention studies, and contributing to cross-disciplinary studies with public health, economics, policy, etc. The framework below shows the modules included in the package and how different modules are linked to each other.

.. image:: docs/figs/Framework.jpg
   :alt: Image
   :width: 800

Installation
============

Install from source:

.. code-block:: bash

   git clone https://github.com/dsncas/normet
   cd normet
   python setup.py install

Main Features
=============

Here are a few of the functions that normet implemented:

  - Automated data curation. Download air quality data and re-analysis data at any time in any area.
  - Automated machine learning. Help to select the 'best' ML model for the dataset and model training.
  - Partial dependency. Look at the drivers (both interactive and noninteractive) of changes in air pollutant concentrations and feature importance.
  - Weather normalisation. Decoupling emission-related air pollutant concentrations from meteorological effects.
  - Change point detection. Detect the change points caused by policy interventions.
  - Causal inference for air quality interventions. Attribution of changes in air pollutant concentrations to air quality policy interventions.

Repository structure
====================

.. code-block:: none

      .                                  # Root folder of our repository
      ├── normet                         # Contains datasets and function modules
      |------__init__.py
      |------ getdata.py                 # Functions of downloading AQ and ERA5 datasets
      |------ autodew.py                 # Functions of automl-based weather normalisation
      |------ pdp.py                     # Functions of partial dependency
      |------ cpd.py                     # Functions of change-point detection
      |------ intervention.py            # Functions of causal inference
      |---------- ...
      |── datasets                       # Datasets used for demonstration
      ├── docs                           # Documentation of the package
      |------ figs                       # Figures for the demonstration
      |---------- NORmet_Framework.webp  # framework for the package
      ├------ tutorials                  # Contains demos and tutorials
      |---------- Case1_autodeweather    # Automl-based weather normalisation
      |---------- Case2_changepoint      # Change-point detection
      |---------- Case3_getdata function # Download AQ and ERA5 data
      |---------- Case4_Intervention     # Causal analysis of air quality interventions
      |------ ...
      ├── setup.py
      ├── pyproject.toml
      ├── LICENSE
      └── README.md

Documentation
=============

You can find Demo and tutorials of the functions `here <https://github.com/m-edal/NORmet/tree/main/tutorials>`_.
