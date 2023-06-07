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
def download_era5(lat_list,lon_list,year_range,month_range,day_range,time_range,path='./',
                 var_list = ['10m_u_component_of_wind', '10m_v_component_of_wind',
                                 '2m_dewpoint_temperature','2m_temperature',
                                 'boundary_layer_height', 'downward_uv_radiation_at_the_surface',
                                  'surface_pressure','surface_solar_radiation_downwards',
                                  'surface_net_solar_radiation','total_cloud_cover','total_precipitation']):
    # 启动多个线程进行并行下载
    threads = []
    for lat, lon in zip(lat_list, lon_list):
        t = threading.Thread(target=download_era5_worker, args=(lat, lon, var_list, year_range, month_range, day_range, time_range,path ))
        t.start()
        threads.append(t)
    # 等待所有线程完成
    for t in threads:
        t.join()
    return t

def era5_dataframe(lat_list,lon_list,path,n_cores=-1):
    results = Parallel(n_jobs=n_cores)(delayed(era5_dataframe_worker)(lat,lon,path) for (lat,lon) in zip(lat_list,lon_list))
    df = pd.concat(results)
    return df


def download_era5_worker(lat, lon, var_list, year_range, month_range, day_range, time_range,path='./'):

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



def era5_dataframe_worker(lat,lon,path):
    # 从数据集中选择与经纬度最接近的点位
    filename = path+f"era5_data_{lat}_{lon}.nc"

    # 读取netcdf文件中的数据
    ds = xr.open_dataset(filename)

    # 提取指定经纬度点位的各种气象参数数据
    if "u10" in list(ds.data_vars):
        u10 = ds.u10.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "v10" in list(ds.data_vars):
        v10 = ds.v10.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "d2m" in list(ds.data_vars):
        d2m = ds.d2m.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "t2m" in list(ds.data_vars):
        t2m = ds.t2m.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "blh" in list(ds.data_vars):
        blh = ds.blh.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "uvb" in list(ds.data_vars):
        uvb = ds.uvb.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "sp" in list(ds.data_vars):
        sp = ds.sp.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "ssrd" in list(ds.data_vars):
        ssrd = ds.ssrd.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "ssr" in list(ds.data_vars):
        ssr = ds.ssr.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "tcc" in list(ds.data_vars):
        tcc = ds.tcc.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    if "tp" in list(ds.data_vars):
        tp = ds.tp.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()

    # 获取时间坐标数组
    time_arr = ds.time.values

    # 将时间坐标数组转换为Pandas DatetimeIndex对象
    time_index = pd.to_datetime(time_arr)
    results = {'u10':u10, 'v10':v10,'d2m':d2m, 't2m':t2m, 'blh':blh, 'uvb':uvb,
        'sp':sp,'ssrd':ssrd, 'ssr':ssr, 'tcc':tcc,'tp':tp}

    # 将气象参数数据存储到Pandas DataFrame中
    df = pd.DataFrame(results, index=time_index)
    df['lat']=lat
    df['lon']=lon
    return df

def UK_AURN(year_lst,authorities_lst=['Manchester'],manual_selection=True,path='./'):
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
    years = year_lst
    if isinstance(years, int):
        years = [years]
    years = sorted(years)
    current_year = datetime.datetime.now().year
    list_authorities = authorities_lst if manual_selection else metadata['AURN_metadata'].local_authority.unique().tolist()

    for local_authority in list_authorities:
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
