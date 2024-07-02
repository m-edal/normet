normet.pdp
==========================

.. function:: pdp_all(automl, df, feature_names=None, variables=None, training_only=True, n_cores=-1)

    Computes partial dependence plots for all specified features.

    :param automl: AutoML model object.
    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param feature_names: List of feature names to compute partial dependence plots for.
    :type feature_names: list
    :param variables: List of variables to compute partial dependence plots for. If None, defaults to feature_names.
    :type variables: list, optional
    :param training_only: If True, computes partial dependence plots only for the training set. Default is True.
    :type training_only: bool, optional
    :param n_cores: Number of CPU cores to use for parallel computation. Default is -1 (uses all available cores).
    :type n_cores: int, optional
    :return: DataFrame containing the computed partial dependence plots for all specified features.
    :rtype: pandas.DataFrame

    **Example:**

    .. code-block:: python

        import normet.pdp as pdp
        df_predict = pdp.pdp_all(automl, df, feature_names=['feature1', 'feature2', 'feature3'])



.. function:: pdp_plot(automl, df, feature_names, variables=None, kind='average', n_cores=-1, training_only=True, figsize=(8, 8), hspace=0.5)

    Plots partial dependence plots for specified features.

    :param automl: AutoML model object.
    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param feature_names: List of feature names to plot partial dependence plots for.
    :type feature_names: list
    :param variables: List of variables to plot partial dependence plots for. If None, defaults to feature_names.
    :type variables: list, optional
    :param kind: Type of plot to generate. Default is 'average'.
    :type kind: str, optional
    :param n_cores: Number of CPU cores to use for parallel computation. Default is -1 (uses all available cores).
    :type n_cores: int, optional
    :param training_only: If True, plots partial dependence plots only for the training set. Default is True.
    :type training_only: bool, optional
    :param figsize: Size of the figure. Default is (8, 8).
    :type figsize: tuple, optional
    :param hspace: Height space between subplots. Default is 0.5.
    :type hspace: float, optional
    :return: Partial dependence plot display object.
    :rtype: PartialDependenceDisplay

    **Example:**

    .. code-block:: python

        import normet.pdp as pdp
        pdp_display = pdp.pdp_plot(automl, df, feature_names=['feature1', 'feature2'])



.. function:: pdp_interaction(automl, df, variables, kind='average', training_only=True, ncols=3, figsize=(8, 4), constrained_layout=True)

    Plots interaction partial dependence plots for specified features.

    :param automl: AutoML model object.
    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param variables: List of feature names to plot interaction partial dependence plots for.
    :type variables: list
    :param kind: Type of plot to generate. Default is 'average'.
    :type kind: str, optional
    :param training_only: If True, plots interaction partial dependence plots only for the training set. Default is True.
    :type training_only: bool, optional
    :param ncols: Number of columns for subplots. Default is 3.
    :type ncols: int, optional
    :param figsize: Size of the figure. Default is (8, 4).
    :type figsize: tuple, optional
    :param constrained_layout: If True, adjusts subplots to fit into the figure area. Default is True.
    :type constrained_layout: bool, optional
    :return: Interaction partial dependence plot display object.
    :rtype: PartialDependenceDisplay

    **Example:**

    .. code-block:: python

        import normet.pdp as pdp
        interaction_pdp = pdp.pdp_interaction(automl, df, variables=['feature1', 'feature2'])



.. function:: pdp_nointeraction(automl, df, feature_names, variables=None, kind='average', training_only=True, ncols=3, figsize=(8, 4), constrained_layout=True)

    Plots partial dependence plots without interaction effects for specified features.

    :param automl: AutoML model object.
    :param df: Input DataFrame containing the dataset.
    :type df: pandas.DataFrame
    :param feature_names: List of feature names to plot partial dependence plots for.
    :type feature_names: list
    :param variables: List of variables to plot partial dependence plots for. If None, defaults to feature_names.
    :type variables: list, optional
    :param kind: Type of plot to generate. Default is 'average'.
    :type kind: str, optional
    :param training_only: If True, plots partial dependence plots only for the training set. Default is True.
    :type training_only: bool, optional
    :param ncols: Number of columns for subplots. Default is 3.
    :type ncols: int, optional
    :param figsize: Size of the figure. Default is (8, 4).
    :type figsize: tuple, optional
    :param constrained_layout: If True, adjusts subplots to fit into the figure area. Default is True.
    :type constrained_layout: bool, optional
    :return: Partial dependence plot display object.
    :rtype: PartialDependenceDisplay

    **Example:**

    .. code-block:: python

        import normet.pdp as pdp
        no_interaction_pdp = pdp.pdp_nointeraction(automl, df, feature_names=['feature1', 'feature2'])
