import threading
import cdsapi
from joblib import Parallel, delayed
import xarray as xr
import pandas as pd
import pyreadr
import os
import numpy as np
import datetime
import wget

#install the CDS API key, https://cds.climate.copernicus.eu/api-how-to
def download_era5(lat_list, lon_list, year_range,
                  month_range=None, day_range=None, time_range=None,
                  var_list=None, path='./'):
    """
    Download ERA5 weather data in parallel.

    Parameters:
        lat_list (list): List of latitudes.
        lon_list (list): List of longitudes.
        year_range (list): Range of years.
        month_range (list): Range of months (default is January to December).
        day_range (list): Range of days (default is 1 to 31).
        time_range (list): Range of times (default is 00:00 to 23:00).
        var_list (list): List of variables to download (default includes 10 common variables).
        path (str): Path to save downloaded files (default is current directory).

    Returns:
        threading.Thread: The last started thread object.
    """
    if month_range is None:
        month_range = [f"{num:02d}" for num in range(1, 13)]
    if day_range is None:
        day_range = [f"{num:02d}" for num in range(1, 32)]
    if time_range is None:
        time_range = [f"{num:02d}:00" for num in range(24)]
    if var_list is None:
        var_list = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                    '2m_dewpoint_temperature', '2m_temperature', 'boundary_layer_height',
                    'surface_pressure', 'surface_solar_radiation_downwards',
                    'total_cloud_cover', 'total_precipitation']

    threads = []
    for lat, lon in zip(lat_list, lon_list):
        for year in year_range:
            for month in month_range:
                t = threading.Thread(target=download_era5_worker,
                                     args=(lat, lon, var_list, year, month,
                                           day_range, time_range, path))
                t.start()
                threads.append(t)
    for t in threads:
        t.join()
    return t


def download_era5_worker(lat, lon, var_list, year, month,
    day_range, time_range,path='./'):
    """
    Download ERA5 weather data for a single coordinate point.

    Parameters:
        lat (float): Latitude.
        lon (float): Longitude.
        var_list (list): List of variables to download.
        year (int): Year.
        month (int): Month.
        day_range (list): Range of days.
        time_range (list): Range of times.
        path (str): Path to save downloaded files (default is current directory).
    """
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        c = cdsapi.Client()
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var_list,
            'year': year,
            'month': month,
            'day': day_range,
            'time': time_range,
            'area': [
                lat+0.25, lon-0.25, lat-0.25,
                lon+0.25,
            ],
        }
        filename = path+f"era5_{lat}_{lon}_{year}_{month}.nc"
        c.retrieve('reanalysis-era5-single-levels', request, filename)
    except Exception as e:
        print(f"CDS API call failed. Make sure to install the CDS API KEY, https://cds.climate.copernicus.eu/api-how-to")
        print(f"Error message: {str(e)}")

def download_era5_area_worker(lat_lim, lon_lim,
    var_list, year, month,
    day_range, time_range,path='./'):
    """
    Download ERA5 weather data for a specified area.

    Parameters:
        lat_lim (list): Latitude range [min_lat, max_lat].
        lon_lim (list): Longitude range [min_lon, max_lon].
        var_list (list): List of variables to download.
        year (int): Year.
        month (int): Month.
        day_range (list): Range of days.
        time_range (list): Range of times.
        path (str): Path to save downloaded files (default is current directory).
    """
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        c = cdsapi.Client()
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var_list,
            'year': year,
            'month': month,
            'day': day_range,
            'time': time_range,
            'area': [lat_lim[1], lon_lim[0],
                    lat_lim[0],lon_lim[1],],
        }

        filename = path+f"era5_{lat_lim}_{lon_lim}_{year}_{month}.nc"
        c.retrieve('reanalysis-era5-single-levels', request, filename)
    except Exception as e:
        print(f"CDS API call failed. Make sure to install the CDS API KEY, https://cds.climate.copernicus.eu/api-how-to")
        print(f"Error message: {str(e)}")

def download_era5_area(lat_lim, lon_lim, year_range,
                       month_range=None, day_range=None, time_range=None,
                       var_list=None, path='./'):
    """
    Download ERA5 weather data for a specified area in parallel.

    Parameters:
        lat_lim (list): Latitude range [min_lat, max_lat].
        lon_lim (list): Longitude range [min_lon, max_lon].
        year_range (list): Range of years.
        month_range (list): Range of months (default is January to December).
        day_range (list): Range of days (default is 1 to 31).
        time_range (list): Range of times (default is 00:00 to 23:00).
        var_list (list): List of variables to download (default includes 10 common variables).
        path (str): Path to save downloaded files (default is current directory).

    Returns:
        threading.Thread: The last started thread object.
    """
    if month_range is None:
        month_range = [f"{num:02d}" for num in range(1, 13)]
    if day_range is None:
        day_range = [f"{num:02d}" for num in range(1, 32)]
    if time_range is None:
        time_range = [f"{num:02d}:00" for num in range(24)]
    if var_list is None:
        var_list = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                    '2m_dewpoint_temperature', '2m_temperature', 'boundary_layer_height',
                    'surface_pressure', 'surface_solar_radiation_downwards',
                    'total_cloud_cover', 'total_precipitation']

    threads = []
    for year in year_range:
        for month in month_range:
            t = threading.Thread(target=download_era5_area_worker,
                                 args=(lat_lim, lon_lim, var_list, year, month,
                                       day_range, time_range, path))
            t.start()
            threads.append(t)
    for t in threads:
        t.join()
    return t

def era5_dataframe(lat_list, lon_list, year_range,
                   month_range=None, path='./', n_cores=-1):
    """
    Read ERA5 weather data in parallel and convert to DataFrame.

    Parameters:
        lat_list (list): List of latitudes.
        lon_list (list): List of longitudes.
        year_range (list): Range of years.
        month_range (list): Range of months (default is January to December).
        path (str): Path to save downloaded files.
        n_cores (int): Number of cores to use (default is all available cores).

    Returns:
        pd.DataFrame: DataFrame containing data for all specified coordinates and years.
    """
    if month_range is None:
        month_range = [f"{num:02d}" for num in range(1, 13)]

    results = Parallel(n_jobs=n_cores)(
        delayed(era5_dataframe_worker)(lat, lon, year_range, month_range, path)
        for lat, lon in zip(lat_list, lon_list)
    )

    df = pd.concat(results)
    return df


def era5_dataframe_worker(lat,lon,year_range,month_range,path):
    """
    Read ERA5 weather data for a single coordinate and range of years and convert to DataFrame.

    Parameters:
        lat (float): Latitude.
        lon (float): Longitude.
        year_range (list): Range of years.
        year_month (list): Range of months.
        path (str): Path to save downloaded files.

    Returns:
        pd.DataFrame: DataFrame containing data for the specified coordinate and years.
    """
    results = []
    for year in year_range:
        for month in month_range:
            filename = f"era5_{lat}_{lon}_{year}_{month}.nc"
            filepath = path + filename
            result = era5_nc_worker(lat, lon, filepath)
            results.append(result)
    df = pd.concat(results)

    return df

def era5_area_dataframe(lat_lim, lon_lim, lat_list, lon_list, year_range,
                        month_range=None, path='./', n_cores=-1):
    """
    Read ERA5 weather data for a specified area in parallel and convert to DataFrame.

    Parameters:
        lat_list (list): List of latitudes.
        lon_list (list): List of longitudes.
        lat_lim (list): Latitude range [min_lat, max_lat].
        lon_lim (list): Longitude range [min_lon, max_lon].
        year_range (list): Range of years.
        month_range (list): Range of months (default is January to December).
        path (str): Path to save downloaded files.
        n_cores (int): Number of cores to use (default is all available cores).

    Returns:
        pd.DataFrame: DataFrame containing data for the specified area and years.
    """
    if month_range is None:
        month_range = [f"{num:02d}" for num in range(1, 13)]

    results = Parallel(n_jobs=n_cores)(
        delayed(era5_area_dataframe_worker)(lat, lon, lat_lim, lon_lim, year_range, month_range, path)
        for lat, lon in zip(lat_list, lon_list)
    )

    df = pd.concat(results)
    return df

def era5_area_dataframe_worker(lat, lon, lat_lim, lon_lim, year_range, month_range,path):
    """
    Read ERA5 weather data for a specified area and range of years and convert to DataFrame.

    Parameters:
        lat (float): Latitude.
        lon (float): Longitude.
        lat_lim (list): Latitude range [min_lat, max_lat].
        lon_lim (list): Longitude range [min_lon, max_lon].
        year_range (list): Range of years.
        year_month (list): Range of months.
        path (str): Path to save downloaded files.

    Returns:
        pd.DataFrame: DataFrame containing data for the specified area and years.
    """
    results = []
    for year in year_range:
        for month in month_range:
            # Construct the filename with year information
            filename = f"era5_{lat_lim}_{lon_lim}_{year}_{month}.nc"
            filepath = path + filename
            # Call the era5_nc_worker function to read the data and append the result to the list
            result = era5_nc_worker(lat, lon, filepath)
            results.append(result)
    # Concatenate the results using pd.concat
    df = pd.concat(results)
    return df

def era5_extract_data(ds, lat, lon, data_vars=None):
    """
    Extract specified variables from an ERA5 dataset for a given latitude and longitude.

    Parameters:
        ds (xarray.Dataset): The dataset from which to extract data.
        lat (float): Latitude.
        lon (float): Longitude.
        data_vars (list): List of variable names to extract (default includes 9 common variables).

    Returns:
        dict: Dictionary containing extracted data for the specified variables, latitude, and longitude.
    """
    if data_vars is None:
        data_vars = ['u10', 'v10', 'd2m', 't2m', 'blh', 'sp', 'ssrd', 'tcc', 'tp']
    data = {}
    for var in data_vars:
        if var in ds.data_vars:
            data[var] = ds[var].sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    return data

def era5_nc_worker(lat, lon, filepath):
    """
    Read ERA5 netCDF file and convert to DataFrame.

    Parameters:
        lat (float): Latitude.
        lon (float): Longitude.
        filepath (str): Path to the netCDF file.

    Returns:
        pd.DataFrame: DataFrame containing data for the specified coordinate.
    """
    ds_raw = xr.open_dataset(filepath)
    ds_raw = ds_raw.sel(**{'latitude': slice(lat+0.25, lat-0.25),
                'longitude': slice(lon-0.25, lon+0.25)})
    if 'expver' in ds_raw.coords:
        ds1 = ds_raw.sel(expver=1)
        data1 = era5_extract_data(ds1, lat, lon)
        ds5 = ds_raw.sel(expver=5)
        data5 = era5_extract_data(ds5, lat, lon)

        df = pd.DataFrame(data1, index=ds1.time.values)
        df5 = pd.DataFrame(data5, index=ds5.time.values)
        df_final =  df.combine_first(df5)
    else:
        data_raw = era5_extract_data(ds_raw, lat, lon)
        df_final = pd.DataFrame(data_raw, index=ds_raw.time.values)
    df_final['rh2m']=100*((6.112 * np.exp((17.67 * (df_final['d2m']-273.15)) / ((df_final['d2m']-273.15) + 243.5)))/(6.112 * np.exp((17.67 * (df_final['t2m']-273.15)) / ((df_final['t2m']-273.15) + 243.5))))
    df_final['lat'] = lat
    df_final['lon'] = lon
    df_final.index.name='date'
    return df_final


def UK_AURN_metadata(path='./'):
    """
    Download and read the metadata for UK AURN data.

    Parameters:
        path (str): Path to the directory where the metadata file will be saved.

    Returns:
        tuple:
            - metadata (dict): Dictionary containing the metadata read from the RData file.
            - list_authorities (list): List of local authorities present in the metadata.
    """
    download_path = path+"AURN_data_download"
    os.makedirs(download_path, exist_ok=True)
    metadata_url = "https://uk-air.defra.gov.uk/openair/R_data/AURN_metadata.RData"
    metadata_file = "AURN_metadata.RData"
    if metadata_file in os.listdir(download_path):
        print("Metadata file already exists, skipping download.")
    else:
        print("Downloading metadata file...")
        wget.download(metadata_url,download_path+'/'+metadata_file)
    metadata = pyreadr.read_r(download_path+'/'+metadata_file)
    list_authorities = list(metadata['AURN_metadata'].local_authority.unique())
    return metadata,list_authorities

def UK_AURN_download(year_lst,list_authorities=None,path='./'):
    """
    Download and process UK AURN data for specified years and local authorities.

    Parameters:
        year_lst (list or int): List of years or a single year for which the data is to be downloaded.
        list_authorities (list): List of local authorities for which the data is to be downloaded.
                                 If None, data for all authorities will be downloaded.
        path (str): Path to the directory where the data files will be saved.

    Returns:
        None
    """
    download_path = path+"AURN_data_download"
    os.makedirs(download_path, exist_ok=True)
    years = year_lst
    if isinstance(years, int):
        years = [years]
    years = sorted(years)
    current_year = datetime.datetime.now().year
    if list_authorities is None:
        list_authorities = UK_AURN_metadata(path=path)[1]
    metadata=UK_AURN_metadata(path=path)[0]

    for local_authority in list_authorities:
        if local_authority not in UK_AURN_metadata(path=path)[1]:
            print("Please select the authorities in the below list: ",UK_AURN_metadata(path=path)[1])
        else:
            data_path = download_path+"/"+str(local_authority)+"/"
            subset_df = metadata['AURN_metadata'][metadata['AURN_metadata'].local_authority == local_authority]
            datetime_start = pd.to_datetime(subset_df['start_date'].values).year
            # Handle 'ongoing' and convert other dates
            end_date_values = subset_df['end_date'].apply(lambda x: datetime.datetime.now() if x == 'ongoing' else x)
            datetime_end_temp = pd.to_datetime(end_date_values)
            # Extract year from end_date values
            datetime_end = datetime_end_temp.dt.year

            now = datetime.datetime.now()
            earliest_year = np.min(datetime_start)
            latest_year = np.max(datetime_end)
            proceed = True
            if latest_year < np.min(years):
                print("Invalid end year, out of range for ", local_authority)
                proceed = False
            if earliest_year > np.max(years):
                print("Invalid start year, out of range for ", local_authority)
                proceed = False
            years_temp = years
            if np.min(years) < earliest_year:
                print("Invalid start year. The earliest you can select for ", local_authority ," is ", str(earliest_year))
                try:
                    years_temp = years_temp[np.where(np.array(years_temp)==earliest_year)[0][0]::]
                except:
                    pass
            if np.max(years) > latest_year:
                print("Invalid end year. The latest you can select for ", local_authority ," is ", str(latest_year))
                try:
                    years_temp = years_temp[0:np.where(np.array(years_temp)==latest_year)[0][0]]
                except:
                    pass
            if not years_temp:
                print("No valid year range")
                proceed = False
            clean_site_data=True
            if proceed:
                os.makedirs(data_path, exist_ok=True)
                for site in subset_df['site_id'].unique():
                    site_type = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['location_type'].unique()[0]
                    station_name = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site]['site_name'].values[0]
                    downloaded_site_data = []
                    for year in years_temp:
                        try:
                            downloaded_file = site + "_" + str(year) + ".RData"
                            filename_path = download_path + "/" + local_authority + "/" + downloaded_file
                            if os.path.isfile(filename_path) and year != current_year:
                                print("Data file already exists", station_name, " in ", str(year))
                            elif os.path.isfile(filename_path) and year == current_year:
                                os.remove(filename_path)
                                print("Updating file for ", station_name, " in ", str(year))
                            else:
                                print("Downloading data file for ", station_name, " in ", str(year))
                                wget.download("https://uk-air.defra.gov.uk/openair/R_data/" + site + "_" + str(year) + ".RData", out=download_path + "/" + local_authority + "/")
                            downloaded_data = pyreadr.read_r(filename_path)
                            downloaded_data[site + "_" + str(year)]['latitude'] = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].latitude.values[0]
                            downloaded_data[site + "_" + str(year)]['longitude'] = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].longitude.values[0]
                            downloaded_data[site + "_" + str(year)]['location_type'] = metadata['AURN_metadata'][metadata['AURN_metadata'].site_id == site].location_type.values[0]
                            downloaded_site_data.append(downloaded_data[site + "_" + str(year)])
                        except:
                            print("Could not download data from ", year, " for ", station_name)
                    if not downloaded_site_data:
                        print("No data could be downloaded for ", station_name)
                    else:
                        final_dataframe = pd.concat(downloaded_site_data, axis=0, ignore_index=True)
                        final_dataframe['datetime'] = pd.to_datetime(final_dataframe['date'])
                        final_dataframe = final_dataframe.sort_values(by='datetime', ascending=True).set_index('datetime')
                        try:
                            final_dataframe['Ox'] = final_dataframe['NO2'] * 23.235 / 46 + final_dataframe['O3'] * 23.235 / 48
                        except:
                            print("Could not create Ox entry for ", site)
                        try:
                            final_dataframe['NOx'] = final_dataframe['NO2'] * 23.235 / 46 + final_dataframe['NO'] * 23.235 / 30
                        except:
                            print("Could not create NOx entry for ", site)
                        if clean_site_data is True:
                            for entry in ['O3', 'NO2', 'NO', 'PM2.5', 'Ox', 'NOx','temp', 'ws', 'wd']:
                                if entry in final_dataframe.columns.values:
                                    final_dataframe=final_dataframe.dropna(subset=[entry])
                            print("Creating .csv file for ", station_name)
                            final_dataframe.to_csv(download_path + "/" + local_authority + "/" + site + '.csv', index=False, header=True)
