import os, sys
from tqdm import tqdm
import numpy as np
import h5py

# Local Imports
from utils import *


if __name__ == "__main__":
    TIME_VARY_DATA_PATH = "/".join( ALL_DATA_PATH.split("/")[:-1] + [f"t_{ALL_DATA_PATH.split('/')[-1]}"] )
    static_data = h5py.File(ALL_DATA_PATH, 'r+')
    time_data = h5py.File(TIME_VARY_DATA_PATH, 'r+')
    t_station_container_group = time_data["stations"]
    MOVE_STATIONS = False
    if MOVE_STATIONS:
        for group_name in tqdm(time_data, total=14911):
            if group_name in ['encoded_dates_for_time_feats_in_index_order', 'stations']:
                continue
            time_data.move(group_name, f"stations/{group_name}")

    MOVE_STATIC = False
    if MOVE_STATIC:
        for group_name in tqdm(static_data, total=14911):
            if group_name == "encoded_dates_for_time_feats_in_index_order":
                continue

            static_data.copy( static_data[f"{group_name}/elevation"]  , time_data[f"stations/{group_name}/"] )
            static_data.copy( static_data[f"{group_name}/geolocation"], time_data[f"stations/{group_name}/"] )
            static_data.copy( static_data[f"{group_name}/landuse"]    , time_data[f"stations/{group_name}/"] )
            static_data.copy( static_data[f"{group_name}/name"]       , time_data[f"stations/{group_name}/"] )
            static_data.copy( static_data[f"{group_name}/open_street_map"], time_data[f"stations/{group_name}/"] )

            #time_data_o = time_data[f"stations/{group_name}"].create_group("open_street_map")
            #static_data.copy( static_data[f"{group_name}/open_street_map"] , time_data_o )
