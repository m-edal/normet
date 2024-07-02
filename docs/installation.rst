Installation
======================================================================================

. code-block:: bash

   conda create -n normet python=3.9 jupyter
   conda activate normet

This package depends on AutoML from flaml. Install FLAML first:

.. code-block:: bash

   conda install flaml -c conda-forge

Install normet using pip:

.. code-block:: bash

   pip install normet

Or install normet from source:

.. code-block:: bash

   git clone https://github.com/dsncas/normet.git
   cd normet
   python setup.py install
