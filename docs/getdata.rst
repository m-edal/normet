normet.getdata.
==========================

.. function:: download_era5(lat_list, lon_list, year_range, month_range=[str(num).zfill(2) for num in list(np.arange(12) + 1)], day_range=[str(num).zfill(2) for num in list(np.arange(31) + 1)], time_range=[str(num).zfill(2) + ':00' for num in list(np.arange(24))], var_list=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'boundary_layer_height', 'surface_pressure', 'surface_solar_radiation_downwards', 'total_cloud_cover', 'total_precipitation'], path='./')

This function downloads ERA5 weather data in parallel using threading.

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
    :returns: The last started thread object.
    :rtype: threading.Thread

    Example:

    .. code-block:: python

        lat_list = [40.0, 50.0]
        lon_list = [-75.0, 10.0]
        year_range = ['2020', '2021']
        download_era5(lat_list, lon_list, year_range)

    This will download ERA5 weather data for the specified latitudes and longitudes over the years 2020 and 2021, saving the data to the current directory.


.. function:: download_era5_area(lat_lim, lon_lim, year_range, month_range=[str(num).zfill(2) for num in list(np.arange(12) + 1)], day_range=[str(num).zfill(2) for num in list(np.arange(31) + 1)], time_range=[str(num).zfill(2) + ':00' for num in list(np.arange(24))], var_list=['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature', 'boundary_layer_height', 'surface_pressure', 'surface_solar_radiation_downwards', 'total_cloud_cover', 'total_precipitation'], path='./')

This function downloads ERA5 weather data for a specified area in parallel using threading.

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
    :returns: The last started thread object.
    :rtype: threading.Thread

    Example:

    .. code-block:: python

        lat_lim = [30.0, 40.0]
        lon_lim = [-80.0, -70.0]
        year_range = ['2020', '2021']
        download_era5_area(lat_lim, lon_lim, year_range)

    This will download ERA5 weather data for the specified latitude and longitude range over the years 2020 and 2021, saving the data to the current directory.


.. function:: era5_dataframe(lat_list, lon_list, year_range, path, n_cores=-1)
This function reads ERA5 weather data in parallel and converts it to a pandas DataFrame.

    :param lat_list: List of latitudes.
    :type lat_list: list
    :param lon_list: List of longitudes.
    :type lon_list: list
    :param year_range: Range of years.
    :type year_range: list
    :param path: Path to save downloaded files.
    :type path: str
    :param n_cores: Number of cores to use (default is all available cores).
    :type n_cores: int, optional
    :returns: DataFrame containing data for all specified coordinates and years.
    :rtype: pd.DataFrame

    Example:

    .. code-block:: python

        lat_list = [40.0, 50.0]
        lon_list = [-75.0, 10.0]
        year_range = ['2020', '2021']
        path = './data'
        df = era5_dataframe(lat_list, lon_list, year_range, path)

    This will read ERA5 weather data for the specified latitudes and longitudes over the years 2020 and 2021, saving the data to the specified path and returning a DataFrame.




.. function:: era5_area_dataframe(lat_list, lon_list, lat_lim, lon_lim, year_range, path, n_cores=-1)
This function reads ERA5 weather data for a specified area in parallel and converts it to a pandas DataFrame.

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
    :param path: Path to save downloaded files.
    :type path: str
    :param n_cores: Number of cores to use (default is all available cores).
    :type n_cores: int, optional
    :returns: DataFrame containing data for the specified area and years.
    :rtype: pd.DataFrame

    Example:

    .. code-block:: python

        lat_list = [40.0, 50.0]
        lon_list = [-75.0, 10.0]
        lat_lim = [30.0, 40.0]
        lon_lim = [-80.0, -70.0]
        year_range = ['2020', '2021']
        path = './data'
        df = era5_area_dataframe(lat_list, lon_list, lat_lim, lon_lim, year_range, path)

    This will read ERA5 weather data for the specified latitude and longitude range over the years 2020 and 2021, saving the data to the specified path and returning a DataFrame.


.. function:: era5_extract_data(ds, lat, lon, data_vars=['u10', 'v10', 'd2m', 't2m', 'blh', 'sp', 'ssrd', 'tcc', 'tp'])
This function extracts specified variables from an ERA5 dataset for a given latitude and longitude.

    Extract specified variables from an ERA5 dataset for a given latitude and longitude.

    :param ds: The dataset from which to extract data.
    :type ds: xarray.Dataset
    :param lat: Latitude.
    :type lat: float
    :param lon: Longitude.
    :type lon: float
    :param data_vars: List of variable names to extract (default includes 9 common variables).
    :type data_vars: list, optional
    :returns: Dictionary containing extracted data for the specified variables, latitude, and longitude.
    :rtype: dict

    Example:

    .. code-block:: python

        ds = xr.open_dataset('era5_data.nc')
        lat = 40.0
        lon = -75.0
        data = era5_extract_data(ds, lat, lon)

    This will extract the specified variables from the ERA5 dataset for the given latitude and longitude, returning the data in a dictionary format.


.. function:: UK_AURN_metadata(path='./')
This function downloads and reads the metadata for UK AURN data.

    Download and read the metadata for UK AURN data.

    :param path: Path to the directory where the metadata file will be saved.
    :type path: str, optional
    :returns:
    - metadata: Dictionary containing the metadata read from the RData file.
    - list_authorities: List of local authorities present in the metadata.
    :rtype: tuple

    Example:

    .. code-block:: python

        metadata, list_authorities = UK_AURN_metadata(path='./data')

    This will download the UK AURN metadata file to the specified path (if it does not already exist), read the metadata, and return it along with a list of local authorities present in the metadata.

    **Details:**

    - **Path to Save Metadata:** The metadata file will be saved in a subdirectory called `AURN_data_download` within the specified path.
    - **Metadata URL:** The metadata is downloaded from the URL `https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData`.
    - **Downloaded File Check:** If the metadata file already exists in the directory, the download is skipped.
    - **Reading Metadata:** The metadata is read using `pyreadr`, and the list of local authorities is extracted from the metadata.

    **Returns:**

    - `metadata`: A dictionary containing the metadata read from the RData file.
    - `list_authorities`: A list of local authorities present in the metadata.

    **Example Usage:**

    .. code-block:: python

        metadata, list_authorities = UK_AURN_metadata(path='./data')
        print(list_authorities)


.. function:: UK_AURN_download(year_lst, list_authorities=None, path='./')
This function downloads and processes UK AURN data for specified years and local authorities.

    Download and process UK AURN data for specified years and local authorities.

    :param year_lst: List of years or a single year for which the data is to be downloaded.
    :type year_lst: list or int
    :param list_authorities: List of local authorities for which the data is to be downloaded. If None, data for all authorities will be downloaded.
    :type list_authorities: list, optional
    :param path: Path to the directory where the data files will be saved.
    :type path: str, optional
    :returns: None

    Example:

    .. code-block:: python

        year_lst = [2020, 2021]
        list_authorities = ['London', 'Manchester']
        path = './data'
        UK_AURN_download(year_lst, list_authorities, path)

    This will download and process UK AURN data for the specified years and local authorities, saving the data to the specified path.

    **Details:**

    - **Path to Save Data:** The data files will be saved in a subdirectory called `AURN_data_download` within the specified path.
    - **Metadata Retrieval:** Metadata is retrieved using the `UK_AURN_metadata` function.
    - **Year Handling:** If a single year is provided, it is converted to a list. The years are sorted and validated against the available range.
    - **Authority Validation:** If `list_authorities` is None, data for all authorities will be downloaded. Authorities are validated against the metadata.
    - **Data Download:** Data is downloaded for each site within the specified authorities and years. Existing files are updated for the current year.
    - **Data Processing:** Downloaded data is combined into a DataFrame, additional columns are calculated (Ox and NOx), and the data is cleaned and saved as a CSV file.

    **Returns:**

    - None

    **Example Usage:**

    .. code-block:: python

        UK_AURN_download(year_lst=[2020, 2021], list_authorities=['London', 'Manchester'], path='./data')

    This will download the UK AURN data for the years 2020 and 2021 for London and Manchester, saving the data to the `./data` directory.
