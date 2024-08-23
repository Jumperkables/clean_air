import os, sys
from loguru import logger
import argparse
import numpy as np
import cfgrib
import pandas as pd
import xarray as xr
#xr.set_options(keep_attrs=True, parallel=True)
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from line_profiler import LineProfiler
import dask.distributed

# Local Imports
from utils import *



def find_closest_float(given_float, float_list):
    closest = min(float_list, key=lambda x: abs(x - given_float))
    return closest



def check_ERA_json_integrity():
    with open(os.path.join(utils.ERA5_PATH, 'works.txt'), 'r') as f:
        definitely_works = f.read().split('\n')
    definitely_works = [ jfile for jfile in definitely_works if jfile != '' ]
    jfiles = [f for f in os.listdir(utils.ERA5_PATH) if f.endswith("json") and (f not in definitely_works) ]
    jfiles = sorted(jfiles)
    for jfile in jfiles:
        try:
            with open(os.path.join(utils.ERA5_PATH, jfile), 'r') as f:
                aa = json.load(f)
            print(len(aa))
            if len(aa) != 14911:
                jfid = int(jfile.split("-")[-1].split(".")[0])
                raise Exception(f"Incomplete data, finish me off `python era5_process_final.py -f {jfid}` ")
            logger.success(jfile)
        except Exception as e:
            logger.error(jfile)
            print(e)



def main_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=int)
    parser.add_argument("-f", type=int, help="Which one to fix")
    args = parser.parse_args()

    stations = LOAD_WAQI_STATIONS()
    #data = cfgrib.open_dataset("/home/jumperkables/TEST.grib", engine="cfgrib")

    # Calculate the wanted latitudes and longitudes ahead of time such that i can select only them from these large grib datasets, seriously improving processing speed
    grib0_lons = np.array([(i*0.5)-180 for i in range(720)])
    grib12_lons = np.array([(i*0.25)-180 for i in range(1440)])
    grib0_lats = np.array([90-(i*0.5) for i in range(361)])
    grib12_lats = np.array([90-(i*0.25) for i in range(721)])

    #station_2_data_tsteps = {f"{station_dict['x']}": {} for station_dict in stations}
    DATA_PATH = "/home/jumperkables/clean_air/data/ERA5Weather"
    station_2_data_tsteps = {}
    assert (args.d == None) ^ (args.f == None), f"Only one of these can be enabled"
    if args.f != None:
        day_idx = args.f 
        with open(os.path.join(DATA_PATH, f"day-{day_idx:02d}.json"), 'r') as f:
            station_2_data_tsteps = json.load(f)
        logger.info( f"{len(station_2_data_tsteps)} stations already processed" )
    else:
        day_idx = args.d
    grib_path = os.path.join(DATA_PATH, f"day-{day_idx:02d}.grib")

    #combined_data = xr.open_mfdataset(grib_paths[0], engine="cfgrib", combine="nested", concat_dim="time")
    combined_data = xr.open_dataset(grib_path, engine="cfgrib")
    combined_data = combined_data.drop_vars(["u100", "v100", "t2m", "d2m", "msl", "meanSea", "sst"])
    client = dask.distributed.Client()
    all_lats = []
    all_lons = []
    CHUNK_SIZE = 15000
    for r_idx, row in enumerate(tqdm(stations, total=len(stations))):
        lat = row["g"][0]
        lon = row["g"][1]
        uid = row["x"]
        # grib0
        closest_lat_grib0 = find_closest_float(lat, grib0_lats)
        closest_lon_grib0 = find_closest_float(lon, grib0_lons)
        closest_lat_idx_grib0 = np.where( grib0_lats == closest_lat_grib0)[0][0]
        closest_lon_idx_grib0 = np.where( grib0_lons == closest_lon_grib0)[0][0]
        # grib12
        closest_lat_grib12 = find_closest_float(lat, grib12_lats)
        closest_lon_grib12 = find_closest_float(lon, grib12_lons)
        closest_lat_idx_grib12 = np.where( grib12_lats == closest_lat_grib12)[0][0]
        closest_lon_idx_grib12 = np.where( grib12_lons == closest_lon_grib12)[0][0]
        feature_dict = {}
        all_lats.append(closest_lat_grib12)
        all_lons.append(closest_lon_grib12)
    grib1 = combined_data.sel(latitude=xr.DataArray(all_lats[:CHUNK_SIZE], dims='z'), longitude=xr.DataArray(all_lons[:CHUNK_SIZE], dims='z'))
    grib1.load()
    feature_dict = {}
    #feature_dict["f_sea_surface_temperature"] = grib1.sst
    feature_dict["f_surface_pressure"]        = grib1.sp
    #feature_dict["f_mean_sea_level_pressure"] = grib1.msl
    feature_dict["f_10m_u_component_of_wind"] = grib1.u10
    feature_dict["f_10m_v_component_of_wind"] = grib1.v10
    #feature_dict["f_2m_temperature"]          = grib1.t2m
    #feature_dict["f_2m_dewpoint_temperature"] = grib1.d2m
    feature_dict["f_skin_temperature"]        = grib1.skt
    #feature_dict["f_100m_u_component_of_wind"]= grib1.u100
    #feature_dict["f_100m_v_component_of_wind"]= grib1.v100

    for r_idx, row in enumerate(tqdm(stations, total=len(stations))):
        lat = row["g"][0]
        lon = row["g"][1]
        uid = row["x"]
        if uid in station_2_data_tsteps.keys():
            continue
        station_2_data_tsteps[uid] = {}
        for ts_idx in range(grib1.time.data.shape[0]):
            timestamp = str(grib1.u10[ts_idx].time.data)
            for feature_key, features in feature_dict.items():
                fs = features[ts_idx][r_idx].values
                if timestamp not in station_2_data_tsteps[uid]:
                    station_2_data_tsteps[uid][timestamp] = {}
                station_2_data_tsteps[uid][timestamp][feature_key] = float(fs)
    with open(os.path.join(DATA_PATH, f"day-{day_idx:02d}.json"), "w") as f:
        json.dump(station_2_data_tsteps, f, indent=4)




if __name__ == "__main__":
    ##'100m_u_component_of_wind'
    ##'100m_v_component_of_wind'
    ##'10m_u_component_of_wind'
    ##'10m_v_component_of_wind'
    ##'2m_dewpoint_temperature'
    ##'2m_temperature'
    ##'mean_sea_level_pressure'
    #'mean_wave_direction'
    #'mean_wave_period'
    ##'sea_surface_temperature'
    ##'skin_temperature'
    ##'surface_pressure'
    ###'total_precipitation'

    # Load stations data
    main_wrapper()
    #profiler = LineProfiler()
    #profiler.add_function(main_wrapper)
    #profiler.run('main_wrapper()')  # Replace 'your_function()' with the function you want to profile
    #profiler.print_stats()
