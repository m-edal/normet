Installation
======================================================================================
Create a Python3.9 environment:

.. code-block:: bash

   conda create -n normet python=3.9
   conda activate normet

This package depends on AutoML from flaml. Install FLAML first:

.. code-block:: bash

   conda install flaml -c conda-forge

Install normet from source:

.. code-block:: bash

   git clone https://github.com/dsncas/normet.git
   cd normet
   python setup.py install
