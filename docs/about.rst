About
======================================================================================

**normet** is a Python package to conduct automated data curation, automated machine learning-based meteorology/weather normalisation and causal analysis on air quality interventions for atmospheric science, air pollution and policy analysis. The main aim of this package is to provide a Swiss army knife enabling rapid automated-air quality intervention studies, and contributing to cross-disciplinary studies with public health, economics, policy, etc. The framework below shows the modules included in the package and how different modules are linked to each other.

.. image:: figs/Framework.jpg
   :alt: Image
   :width: 800

Here are a few of the functions that normet implemented:

  - Automated data curation. Download air quality data and re-analysis data at any time in any area.
  - Automated machine learning. Help to select the 'best' ML model for the dataset and model training.
  - Partial dependency. Look at the drivers (both interactive and noninteractive) of changes in air pollutant concentrations and feature importance.
  - Weather normalisation. Decoupling emission-related air pollutant concentrations from meteorological effects.
  - Change point detection. Detect the change points caused by policy interventions.
  - Causal inference for air quality interventions. Attribution of changes in air pollutant concentrations to air quality policy interventions.
