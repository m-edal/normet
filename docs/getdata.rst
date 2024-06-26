normet.getdata.
==========================

.. function:: download_era5(lat_list, lon_list, year_range, month_range, day_range, time_range, var_list, path='./')

    Downloads ERA5 weather data for specified locations and time periods in parallel.

    :param lat_list: List of latitudes.
    :type lat_list: list of float
    :param lon_list: List of longitudes.
    :type lon_list: list of float
    :param year_range: Range of years.
    :type year_range: list of int
    :param month_range: Range of months (default is January to December).
    :type month_range: list of int, optional
    :param day_range: Range of days (default is 1 to 31).
    :type day_range: list of int, optional
    :param time_range: Range of times (default is 00:00 to 23:00).
    :type time_range: list of str, optional
    :param var_list: List of variables to download (default includes 10 common variables).
    :type var_list: list of str, optional
    :param path: Path to save the downloaded files (default is current directory).
    :type path: str, optional
    :return: The last started thread object.
    :rtype: threading.Thread

    Example usage:

    .. code-block:: python

        latitudes = [50.0, 51.0]
        longitudes = [-0.1, 0.0]
        years = [2020, 2021]
        download_era5(lat_list=latitudes, lon_list=longitudes, year_range=years)


.. function:: download_era5_worker(lat, lon, var_list, year, month, day_range, time_range, path='./')

    Helper function to download ERA5 weather data for a single coordinate point.

    This function handles the download of ERA5 weather data for a specific latitude and longitude, and for specific time periods, using the Copernicus Climate Data Store (CDS) API.

    :param lat: Latitude.
    :type lat: float
    :param lon: Longitude.
    :type lon: float
    :param var_list: List of variables to download.
    :type var_list: list of str
    :param year: Year to download data for.
    :type year: int
    :param month: Month to download data for.
    :type month: str
    :param day_range: List of days to download data for.
    :type day_range: list of str
    :param time_range: List of times to download data for.
    :type time_range: list of str
    :param path: Path to save the downloaded files.
    :type path: str, optional

    Raises:
        Exception: If the CDS API call fails, an exception is raised with an error message.

    Example usage:

    .. code-block:: python

        download_era5_worker(50.0, -0.1, ['2m_temperature'], 2020, '01', ['01', '02'], ['00:00', '12:00'])


.. function:: download_era5_area_worker(lat_lim, lon_lim, var_list, year, month, day_range, time_range, path='./')

    Helper function to download ERA5 weather data for a specified area.

    This function handles the download of ERA5 weather data for a specific geographic area, and for specific time periods, using the Copernicus Climate Data Store (CDS) API.

    :param lat_lim: Latitude range [min_lat, max_lat].
    :type lat_lim: list of float
    :param lon_lim: Longitude range [min_lon, max_lon].
    :type lon_lim: list of float
    :param var_list: List of variables to download.
    :type var_list: list of str
    :param year: Year to download data for.
    :type year: int
    :param month: Month to download data for.
    :type month: str
    :param day_range: List of days to download data for.
    :type day_range: list of str
    :param time_range: List of times to download data for.
    :type time_range: list of str
    :param path: Path to save the downloaded files.
    :type path: str, optional

    Raises:
        Exception: If the CDS API call fails, an exception is raised with an error message.

    Example usage:

    .. code-block:: python

        download_era5_area_worker([49.5, 50.5], [-0.5, 0.5], ['2m_temperature'], 2020, '01', ['01', '02'], ['00:00', '12:00'])


.. function:: download_era5_area(lat_lim, lon_lim, year_range, month_range, day_range, time_range, var_list, path='./')

    Download ERA5 weather data for a specified area in parallel.

    :param lat_lim: Latitude range [min_lat, max_lat].
    :type lat_lim: list of float
    :param lon_lim: Longitude range [min_lon, max_lon].
    :type lon_lim: list of float
    :param year_range: Range of years.
    :type year_range: list of int
    :param month_range: Range of months (default is January to December).
    :type month_range: list of int, optional
    :param day_range: Range of days (default is 1 to 31).
    :type day_range: list of int, optional
    :param time_range: Range of times (default is 00:00 to 23:00).
    :type time_range: list of str, optional
    :param var_list: List of variables to download (default includes 10 common variables).
    :type var_list: list of str, optional
    :param path: Path to save the downloaded files (default is current directory).
    :type path: str, optional
    :return: The last started thread object.
    :rtype: threading.Thread

    Example usage:

    .. code-block:: python

        lat_lim = [49.0, 51.0]
        lon_lim = [-1.0, 1.0]
        year_range = [2020, 2021]
        download_era5_area(lat_lim, lon_lim, year_range)


.. function:: era5_dataframe(lat_list, lon_list, year_range, month_range, path='./', n_cores=-1)

    Read ERA5 weather data in parallel and convert to DataFrame.

    :param lat_list: List of latitudes.
    :type lat_list: list of float
    :param lon_list: List of longitudes.
    :type lon_list: list of float
    :param year_range: Range of years.
    :type year_range: list of int
    :param month_range: Range of months (default is January to December).
    :type month_range: list of int, optional
    :param path: Path to save downloaded files.
    :type path: str, optional
    :param n_cores: Number of cores to use (default is all available cores).
    :type n_cores: int, optional
    :return: DataFrame containing data for all specified coordinates and years.
    :rtype: pd.DataFrame

    Example usage:

    .. code-block:: python

        lat_list = [50.0, 51.0]
        lon_list = [-0.1, 0.0]
        year_range = [2020, 2021]
        df = era5_dataframe(lat_list, lon_list, year_range)


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


.. function:: UK_AURN_download(year_lst, list_authorities=None, molarv=23.235, path='./')

    Download and process UK AURN data for specified years and local authorities.

    :param year_lst: List of years or a single year for which the data is to be downloaded.
    :type year_lst: list or int
    :param list_authorities: List of local authorities for which the data is to be downloaded. If None, data for all authorities will be downloaded. Default is None.
    :type list_authorities: list, optional
    :param molarv: Molar volume value to use for calculating Ox and NOx entries. Defaults to 23.235.
    :type molarv: float, optional
    :param path: Path to the directory where the data files will be saved. Defaults to current directory.
    :type path: str, optional
    :returns: None

    **Example Usage:**

    .. code-block:: python

        UK_AURN_download([2020, 2021], list_authorities=['Birmingham', 'Manchester'])

    **Details:**

    This function downloads and processes UK Air Quality Archive (AURN) data for specified years and local authorities. It retrieves data files for each specified local authority and year from the UK Air Quality Archive website (https://uk-air.defra.gov.uk/openair/R_data/) and saves them in the specified path.

    - If `list_authorities` is None, data for all available local authorities will be downloaded.
    - The function checks for existing data files and updates them if necessary for the current year.
    - The downloaded data is processed to create additional columns such as Ox and NOx based on provided molar volume (`molarv`).
    - Each data file is saved in CSV format with columns cleaned for relevant air quality parameters.

    **Notes:**

    - The function relies on external data sources and requires an internet connection to download data files.
    - It handles exceptions for cases where data retrieval or processing fails, printing informative messages.
    - Ensure sufficient storage space and permissions for the specified download path.

    **See also:**

    - :func:`UK_AURN_metadata`: Function to retrieve metadata about available UK AURN data.
