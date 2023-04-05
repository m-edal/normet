import threading
import cdsapi
from joblib import Parallel, delayed

def download_era5(lat_list,lon_list,year_range,month_range,day_range,time_range):
    # 启动多个线程进行并行下载
    threads = []
    for lat, lon in zip(lat_list, lon_list):
        t = threading.Thread(target=download_era5_worker, args=(lat, lon, year_range, month_range, day_range, time_range, ))
        t.start()
        threads.append(t)
    # 等待所有线程完成
    for t in threads:
        t.join()
    return t

def era5_dataframe(lat_list,lon_list,n_cores=-1):
    results = Parallel(n_jobs=n_cores)(delayed(era5_dataframe_worker)(lat,lon) for (lat,lon) in zip(lat_list,lon_list))
    df = pd.concat(results)
    return df


def download_era5_worker(lat, lon, year_range, month_range, day_range, time_range):
    # 定义要下载的气象参数
    variables = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'relative_humidity']

    # 创建一个CDS API客户端对象
    c = cdsapi.Client()

    # 定义下载请求的参数
    request = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': variables,
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
    filename = f"era5_data_{lat}_{lon}.nc"
    c.retrieve('reanalysis-era5-single-levels', request, filename)



def era5_dataframe_worker(lat,lon):
    # 从数据集中选择与经纬度最接近的点位
    filename = f"era5_data_{lat}_{lon}.nc"

    # 读取netcdf文件中的数据
    ds = xr.open_dataset(filename)

    lat_idx = abs(ds.latitude - lat).argmin().item()
    lon_idx = abs(ds.longitude - lon).argmin().item()

    # 提取指定经纬度点位的各种气象参数数据
    u10 = ds.u10.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    v10 = ds.v10.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    d2m = ds.d2m.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    t2m = ds.t2m.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    blh = ds.blh.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    uvb = ds.uvb.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    sst = ds.sst.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    sp = ds.sp.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    ssrd = ds.ssrd.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    ssr = ds.ssr.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    tcc = ds.tcc.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()
    tp = ds.tp.sel(latitude=lat, longitude=lon, method='nearest').values.tolist()

    # 获取时间坐标数组
    time_arr = ds.time.values

    # 将时间坐标数组转换为Pandas DatetimeIndex对象
    time_index = pd.to_datetime(time_arr)

    # 将气象参数数据存储到Pandas DataFrame中
    df = pd.DataFrame({'u10':u10, 'v10':v10,'d2m':d2m, 't2m':t2m, 'blh':blh, 'uvb':uvb,
        'sst':sst, 'sp':sp,'ssrd':ssrd, 'ssr':ssr, 'tcc':tcc,
        'tp':tp}, index=time_index)
    df['lat']=lat
    df['lon']=lon
    return df
