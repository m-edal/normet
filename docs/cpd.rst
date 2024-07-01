normet.cpd.
==========================

.. function:: cpd_rupture(df, col_name='Normalised', window=12, n=5, model="l2")

    Detects change points in a time series using the ruptures package.

    :param df: Input DataFrame containing the time series data.
    :type df: DataFrame
    :param col_name: Name of the column containing the time series data. Default is 'Normalised'.
    :type col_name: str, optional
    :param window: Width of the sliding window. Default is 12.
    :type window: int, optional
    :param n: Number of change points to detect. Default is 5.
    :type n: int, optional
    :param model: Type of cost function model for the ruptures package. Default is "l2".
    :type model: str, optional
    :returns: Datetime indices of detected change points.
    :rtype: DatetimeIndex

    **Details:**

    - **Change Point Detection:** Detects change points in a time series using the ruptures package.
    - **Sliding Window:** Uses a sliding window approach with a specified width to detect change points.
    - **Model Selection:** Allows for selecting different cost function models such as "l1", "rbf", "linear", "normal", or "ar".

    **Returns:**

    - Datetime indices of detected change points.

    **Example:**

    .. code-block:: python

        >>> import normet.cpd as cpd
        >>> change_points_rupture = cpd.pd_rupture(df, col_name='Normalised', window=12, n=5, model="l2")
        >>> print("Change points detected using cpd_rupture function:")
        >>> print(change_points_rupture)

    This function can be used to detect change points in a time series, providing insights into structural shifts in the data.



.. function:: cpd_cumsum(df, col_name='Normalised', threshold_mean=10, threshold_std=3000)

    Detects change points in a time series using the cumulative sums method.

    :param df: Input DataFrame containing the time series data.
    :type df: DataFrame
    :param col_name: Name of the column containing the time series data. Default is 'Normalised'.
    :type col_name: str, optional
    :param threshold_mean: Threshold for mean change detection. Default is 10.
    :type threshold_mean: float, optional
    :param threshold_std: Threshold for standard deviation change detection. Default is 3000.
    :type threshold_std: float, optional
    :returns: Datetime indices of detected change points.
    :rtype: DatetimeIndex

    **Details:**

    - **Change Point Detection:** Detects change points in a time series using the cumulative sums method.
    - **Thresholds:** Employs threshold values for mean and standard deviation change detection.
    - **Cumulative Sums:** Utilizes cumulative sums of data to identify significant deviations.

    **Returns:**

    - Datetime indices of detected change points.

    **Example:**

    .. code-block:: python

        >>> import normet.cpd as cpd
        >>> change_points_cumsum = cpd.cpd_cumsum(df, col_name='Normalised', threshold_mean=10, threshold_std=3000)
        >>> print("Change points detected using cpd_cumsum function:")
        >>> print(change_points_cumsum)


    This function is useful for identifying significant changes in a time series based on deviations in mean and standard deviation.
