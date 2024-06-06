getdata
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
