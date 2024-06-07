normet.intervention.
==========================

.. function:: scm(df, poll_col, date_col, code_col, treat_target, control_pool, post_col)

    Performs Synthetic Control Method (SCM) for a single treatment target.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type poll_col: str
    :param date_col: Name of the column containing the date data.
    :type date_col: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param treat_target: Code of the treatment target.
    :type treat_target: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param post_col: Name of the column indicating the post-treatment period.
    :type post_col: str
    :return: DataFrame containing synthetic control results for the specified treatment target.
    :rtype: pandas.DataFrame

Example usage:

.. code-block:: python

    synthetic_result = scm(df, poll_col='poll', date_col='date', code_col='code', treat_target='X', control_pool=['A', 'B', 'C'], post_col='post')


.. function:: scm_parallel(df, poll_col, date_col, code_col, control_pool, post_col, n_cores=-1)

    Performs Synthetic Control Method (SCM) in parallel for multiple treatment targets.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type poll_col: str
    :param date_col: Name of the column containing the date data.
    :type date_col: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param post_col: Name of the column indicating the post-treatment period.
    :type post_col: str
    :param n_cores: Number of CPU cores to use. Default is -1 (uses all available cores).
    :type n_cores: int, optional
    :return: DataFrame containing synthetic control results for all treatment targets.
    :rtype: pandas.DataFrame

Example usage:

.. code-block:: python

    synthetic_results = scm_parallel(df, poll_col='poll', date_col='date', code_col='code', control_pool=['A', 'B', 'C'], post_col='post')



.. function:: ml_syn(df, poll_col, date_col, code_col, treat_target, control_pool, cutoff_date, training_time=60)

    Performs synthetic control using machine learning regression models.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type poll_col: str
    :param date_col: Name of the column containing the date data.
    :type date_col: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param treat_target: Code of the treatment target.
    :type treat_target: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param cutoff_date: Date for splitting pre- and post-treatment datasets.
    :type cutoff_date: str
    :param training_time: Total running time in seconds for the AutoML model. Default is 60.
    :type training_time: int, optional
    :return: DataFrame containing synthetic control results for the specified treatment target.
    :rtype: pandas.DataFrame

Example usage:

.. code-block:: python

    synthetic_result = ml_syn(df, poll_col='poll', date_col='date', code_col='code', treat_target='X', control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01')



.. function:: ml_syn_parallel(df, poll_col, date_col, code_col, control_pool, cutoff_date, training_time=60, n_cores=-1)

    Performs synthetic control using machine learning regression models in parallel for multiple treatment targets.

    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param poll_col: Name of the column containing the poll data.
    :type poll_col: str
    :param date_col: Name of the column containing the date data.
    :type date_col: str
    :param code_col: Name of the column containing the code data.
    :type code_col: str
    :param control_pool: List of control pool codes.
    :type control_pool: list
    :param cutoff_date: Date for splitting pre- and post-treatment datasets.
    :type cutoff_date: str
    :param training_time: Total running time in seconds for the AutoML model. Default is 60.
    :type training_time: int, optional
    :param n_cores: Number of CPU cores to use. Default is -1 (uses all available cores).
    :type n_cores: int, optional
    :return: DataFrame containing synthetic control results for all treatment targets.
    :rtype: pandas.DataFrame

Example usage:

.. code-block:: python

    synthetic_results = ml_syn_parallel(df, poll_col='poll', date_col='date', code_col='code', control_pool=['A', 'B', 'C'], cutoff_date='2020-01-01', training_time=60)
