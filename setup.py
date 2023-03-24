from setuptools import setup

setup(name='normet',
      packages=['normet'],
      package_data={'normet': ['datasets/*/*']},
      description='Weather normalisation using automated machine learning',
      author='M-edal',
      url='https://github.com/m-edal/normet',
      license='MIT',
      classifiers = [],
      install_requires=[
          'pandas',
          'numpy',
          'multiprocessing',
          'scipy',
          'joblib',
          'flaml',
          'matplotlib',
          'pylab',
          'seaborn',
          'os',
          'random',
          'pdpbox'
      ],
      zip_safe=False)
