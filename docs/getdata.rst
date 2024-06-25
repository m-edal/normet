normet.getdata.
==========================

.. function:: download_era5(lat_list, lon_list, year_range, month_range, day_range, time_range, var_list, path='./')

    Downloads ERA5 weather data in parallel.

    :param lat_list: List of latitudes.
    :type lat_list: list
    :param lon_list: List of longitudes.
    :type lon_list: list
    :param year_range: Range of years.
    :type year_range: list
    :param month_range: Range of months (default is January to December).
    :type month_range: list, optional
    :param day_range: Range of days (default is 1 to 31).
    :type day_range: list, optional
    :param time_range: Range of times (default is 00:00 to 23:00).
    :type time_range: list, optional
    :param var_list: List of variables to download (default includes 10 common variables).
    :type var_list: list, optional
    :param path: Path to save downloaded files (default is current directory).
    :type path: str, optional
    :return: The last started thread object.
    :rtype: threading.Thread

Example usage:

.. code-block:: python

    latitudes = [34.05, 36.16]
    longitudes = [-118.24, -115.15]
    years = [2020, 2021]
    download_era5(lat_list=latitudes, lon_list=longitudes, year_range=years)


.. function:: download_era5_area(lat_lim, lon_lim, year_range, month_range, day_range, time_range, var_list, path='./')

    Downloads ERA5 weather data for a specified area in parallel.

    :param lat_lim: Latitude range [min_lat, max_lat].
    :type lat_lim: list
    :param lon_lim: Longitude range [min_lon, max_lon].
    :type lon_lim: list
    :param year_range: Range of years.
    :type year_range: list
    :param month_range: Range of months (default is January to December).
    :type month_range: list, optional
    :param day_range: Range of days (default is 1 to 31).
    :type day_range: list, optional
    :param time_range: Range of times (default is 00:00 to 23:00).
    :type time_range: list, optional
    :param var_list: List of variables to download (default includes 10 common variables).
    :type var_list: list, optional
    :param path: Path to save downloaded files (default is current directory).
    :type path: str, optional
    :return: The last started thread object.
    :rtype: threading.Thread

Example usage:

.. code-block:: python

    lat_lim = [30.0, 40.0]
    lon_lim = [-120.0, -110.0]
    years = [2020, 2021]
    download_era5_area(lat_lim=lat_lim, lon_lim=lon_lim, year_range=years)



.. function:: era5_dataframe(lat_list, lon_list, year_range, month_range, path='./', n_cores=-1)

    Reads ERA5 weather data in parallel and converts it to a DataFrame.

    :param lat_list: List of latitudes.
    :type lat_list: list
    :param lon_list: List of longitudes.
    :type lon_list: list
    :param year_range: Range of years.
    :type year_range: list
    :param month_range: Range of months.
    :type month_range: list, optional
    :param path: Path to save downloaded files.
    :type path: str, optional
    :param n_cores: Number of cores to use (default is all available cores).
    :type n_cores: int, optional
    :return: DataFrame containing data for all specified coordinates and years.
    :rtype: pd.DataFrame

Example usage:

.. code-block:: python

    lat_list = [30.0, 35.0, 40.0]
    lon_list = [-120.0, -115.0, -110.0]
    year_range = [2020, 2021]
    path = './data/'

    df = era5_dataframe(lat_list, lon_list, year_range, path)
    print(df.head())



.. function:: era5_area_dataframe(lat_list, lon_list, lat_lim, lon_lim, year_range, month_range, path='./', n_cores=-1)

    Reads ERA5 weather data for a specified area in parallel and converts it to a DataFrame.

    :param lat_list: List of latitudes.
    :type lat_list: list
    :param lon_list: List of longitudes.
    :type lon_list: list
    :param lat_lim: Latitude range [min_lat, max_lat].
    :type lat_lim: list
    :param lon_lim: Longitude range [min_lon, max_lon].
    :type lon_lim: list
    :param year_range: Range of years.
    :type year_range: list
    :param month_range: Range of months.
    :type month_range: list, optional
    :param path: Path to save downloaded files.
    :type path: str, optional
    :param n_cores: Number of cores to use (default is all available cores).
    :type n_cores: int, optional
    :return: DataFrame containing data for the specified area and years.
    :rtype: pd.DataFrame

Example usage:

.. code-block:: python

    lat_list = [30.0, 35.0, 40.0]
    lon_list = [-120.0, -115.0, -110.0]
    lat_lim = [20.0, 50.0]
    lon_lim = [-130.0, -100.0]
    year_range = [2020, 2021]
    path = './data/'

    df = era5_area_dataframe(lat_list, lon_list, lat_lim, lon_lim, year_range, path)
    print(df.head())



.. function:: era5_extract_data(ds, lat, lon, data_vars)

    Extracts specified variables from an ERA5 dataset for a given latitude and longitude.

    :param ds: The dataset from which to extract data.
    :type ds: xarray.Dataset
    :param lat: Latitude.
    :type lat: float
    :param lon: Longitude.
    :type lon: float
    :param data_vars: List of variable names to extract (default includes 9 common variables).
    :type data_vars: list
    :return: Dictionary containing extracted data for the specified variables, latitude, and longitude.
    :rtype: dict

Example usage:

.. code-block:: python

    import xarray as xr

    # Assuming 'ds' is an xarray.Dataset loaded with ERA5 data
    ds = xr.open_dataset('path_to_era5_data.nc')
    lat = 40.0
    lon = -75.0

    extracted_data = era5_extract_data(ds, lat, lon)
    print(extracted_data)



.. function:: UK_AURN_metadata(path='./')

    Downloads and reads the metadata for UK AURN data.

    :param path: Path to the directory where the metadata file will be saved.
    :type path: str
    :return: Tuple containing the metadata read from the RData file and a list of local authorities present in the metadata.
    :rtype: tuple

Example usage:

.. code-block:: python

    metadata, authorities = UK_AURN_metadata()
    print(metadata)
    print(authorities


.. function:: UK_AURN_download(year_lst, list_authorities=None, path='./')

    Downloads and processes UK AURN data for specified years and local authorities.

    :param year_lst: List of years or a single year for which the data is to be downloaded.
    :type year_lst: list or int
    :param list_authorities: List of local authorities for which the data is to be downloaded. If None, data for all authorities will be downloaded.
    :type list_authorities: list, optional
    :param path: Path to the directory where the data files will be saved.
    :type path: str
    :return: None

Example usage:

.. code-block:: python

    # Download data for the year 2022 for all local authorities
    UK_AURN_download(2022)

    # Download data for the years 2020, 2021, and 2022 for specific local authorities
    UK_AURN_download([2020, 2021, 2022], list_authorities=['Authority1', 'Authority2'])
