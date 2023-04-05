import threading
import cdsapi
from joblib import Parallel, delayed
import xarray as xr
import pandas as pd

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

    lat_idx = abs(ds.latitude - lat).argmin().item()
    lon_idx = abs(ds.longitude - lon).argmin().item()

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
