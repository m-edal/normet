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
def download_era5(lat_list,lon_list,year_range,
    month_range=[str(num).zfill(2) for num in list(np.arange(12)+1)],
    day_range=[str(num).zfill(2) for num in list(np.arange(31)+1)],
    time_range=[str(num).zfill(2)+ ':00' for num in list(np.arange(24))],
    var_list = ['10m_u_component_of_wind', '10m_v_component_of_wind',
        '2m_dewpoint_temperature','2m_temperature','boundary_layer_height',
        'surface_pressure','surface_solar_radiation_downwards',
        'total_cloud_cover','total_precipitation'],path='./'):
    # 启动多个线程进行并行下载
    threads = []
    for lat, lon in zip(lat_list, lon_list):
        t = threading.Thread(target=download_era5_worker,
            args=(lat, lon, var_list, year_range, month_range,
            day_range, time_range,path ))
        t.start()
        threads.append(t)
    # 等待所有线程完成
    for t in threads:
        t.join()
    return t

def download_era5_worker(lat, lon, var_list, year_range, month_range,
    day_range, time_range,path='./'):
    try:
        # 创建一个CDS API客户端对象
        c = cdsapi.Client()

        # 定义下载请求的参数
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var_list,
            'year': year_range,
            'month': month_range,
            'day': day_range,
            'time': time_range,
            'area': [
                lat+0.25, lon-0.25, lat-0.25,
                lon+0.25,
            ],
        }

        # 执行下载请求，并将数据保存到本地文件
        filename = path+f"era5_data_{lat}_{lon}.nc"
        c.retrieve('reanalysis-era5-single-levels', request, filename)
    except Exception as e:
        print(f"CDS API call failed. Make sure to install the CDS API KEY, https://cds.climate.copernicus.eu/api-how-to")
        print(f"Error message: {str(e)}")

def era5_dataframe(lat_list,lon_list,path,n_cores=-1):
    results = Parallel(n_jobs=n_cores)(delayed(era5_dataframe_worker)(lat,lon,path) for (lat,lon) in zip(lat_list,lon_list))
    df = pd.concat(results)
    return df

def era5_dataframe_worker(lat,lon,path):
    # 从数据集中选择与经纬度最接近的点位
    filename = path+f"era5_data_{lat}_{lon}.nc"
    df=era5_nc_worker(filename,lat,lon)
    return df

def download_era5_area(lat_lim, lon_lim, year_range,
    month_range=[str(num).zfill(2) for num in list(np.arange(12)+1)],
    day_range=[str(num).zfill(2) for num in list(np.arange(31)+1)],
    time_range=[str(num).zfill(2)+ ':00' for num in list(np.arange(24))],
    var_list = ['10m_u_component_of_wind', '10m_v_component_of_wind',
        '2m_dewpoint_temperature','2m_temperature','boundary_layer_height',
        'surface_pressure','surface_solar_radiation_downwards',
        'total_cloud_cover','total_precipitation'],path='./'):
    try:
        # 创建一个CDS API客户端对象
        c = cdsapi.Client()

        # 定义下载请求的参数
        request = {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': var_list,
            'year': year_range,
            'month': month_range,
            'day': day_range,
            'time': time_range,
            'area': [
                lat_lim[1], lon_lim[0], lat_lim[0],
                lon_lim[1],
            ],
        }

        # 执行下载请求，并将数据保存到本地文件
        filename = path+f"era5_data_{lat_lim}_{lon_lim}.nc"
        c.retrieve('reanalysis-era5-single-levels', request, filename)
    except Exception as e:
        print(f"CDS API call failed. Make sure to install the CDS API KEY, https://cds.climate.copernicus.eu/api-how-to")
        print(f"Error message: {str(e)}")

def era5_area_dataframe(lat_list,lon_list,filepath,n_cores=-1):
    results = Parallel(n_jobs=n_cores)(delayed(era5_area_dataframe_worker)(lat,lon,filepath) for (lat,lon) in zip(lat_list,lon_list))
    df = pd.concat(results)
    return df

def era5_area_dataframe_worker(lat,lon,filepath):
    df=era5_nc_worker(filepath,lat,lon)
    return df

def era5_extract_data(ds, lat, lon):
    data_vars = ['u10', 'v10', 'd2m', 't2m', 'blh', 'sp', 'ssrd', 'tcc', 'tp']
    data = {}

    for var in data_vars:
        if var in ds.data_vars:
            data[var] = ds[var].sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    return data

def era5_nc_worker(nc_filepath, lat, lon):
    ds_raw = xr.open_dataset(nc_filepath)
    if 'expver' in ds_raw.coords:
        ds1 = ds_raw.sel(expver=1)
        data1 = era5_extract_data(ds1, lat, lon)

        last_valid_time = pd.to_datetime(ds1.time.max().values)

        ds5 = ds_raw.sel(expver=5)
        data5 = era5_extract_data(ds5, lat, lon)

        df = pd.DataFrame(data1, index=ds1.time.values)
        df5 = pd.DataFrame(data5, index=ds5.time.values)
        df_final = pd.concat([df[df.index > last_valid_time], df5], axis=0)
    else:
        data_raw = era5_extract_data(ds_raw, lat, lon)
        df_final = pd.DataFrame(data_raw, index=ds_raw.time.values)

    df_final['lat'] = lat
    df_final['lon'] = lon

    return df_final.dropna()


def UK_AURN_metadata(path='./'):
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
            datetime_start = pd.to_datetime(subset_df['start_date'].values, format='%Y/%m/%d').year
            datetime_end_temp = subset_df['end_date'].values
            now = datetime.datetime.now()
            datetime_end = [now.year]*len(datetime_end_temp) if 'ongoing' in datetime_end_temp else pd.to_datetime(datetime_end_temp).year
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
