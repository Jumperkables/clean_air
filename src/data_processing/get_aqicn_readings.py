__author__ = "Jumperkables (Tom Winterbottom)"

# Standard Imports
import os, sys
import time
import json
from loguru import logger
from tqdm import tqdm
import pandas as pd
import requests


# Local Imports
from utils import *

# Get Token
#TOKEN = utils.get_waqi_token()
DELAY = 0.5


#"""
# Get as many readings as possible from stations
#"""
"""
https://api.waqi.info/api/attsse/9049/yd.json
https://api.waqi.info/feed/@4981/?token=d863f2cdb7c923064bc8d1955d49b0c33d7d8931
https://airnet.waqi.info/airnet/sse/historic/daily/145219?specie=pm25
"""
if __name__ == "__main__":
    #url = "https://aqicn.org/map/world/"
    waqi_stats = LOAD_WAQI_STATIONS()
    total_iterations = len(waqi_stats)
    ###########################
    # Example loop
    success_count = 0
    failure_count = 0
    fail_ids = []
    # Setting up the tqdm progress bars
    success_bar = tqdm(total=total_iterations, desc='Successes', position=0)
    failure_bar = tqdm(total=total_iterations, desc='Failures', position=1)
    ###########################

    base_url = "https://airnet.waqi.info/airnet/sse/historic/daily"
    species = ["pm25", "pm10", "pm1", "o3", "no2", "so2", "co"]
    #all_data = {}
    for idx, w_stat in tqdm(enumerate(waqi_stats), total=total_iterations):
        uid = w_stat["x"]
        lat = w_stat["g"][0]
        lon = w_stat["g"][1]
        name = w_stat["n"]
        save_path = os.path.join(READINGS_PATH, f"{uid}.json")
        if os.path.exists(save_path):
            continue

        station_url = f"{base_url}/{uid}"
        r = requests.get(station_url)
        if r.status_code != 200:
            failure_count += 1
            failure_bar.update(1)
            fail_ids.append(row[1]["uid"])
        else:
            success_count += 1
            success_bar.update(1)
        time.sleep(DELAY)
        readings_data = {}
        for specie in species:
            url = f"{station_url}?specie={specie}"
            r = requests.get(url)
            if r.status_code != 200:
                continue
            text = r.text
            readings = r.text.split("\n\nevent: data\ndata: ")
            r_data = []
            try:
                station_data = readings[0].split("event: station\ndata: ")[-1]
                # station_data = json.loads(station_data) TODO Fix this JSON decode error
                readings_data["station_data"] = station_data
                for reading in readings[1:-1]:
                    data = json.loads(reading)
                    r_data.append(data)
            except IndexError:
                r_data = []
            readings_data[specie] = r_data
            time.sleep(DELAY)   # Respect the API
        #all_data[uid] = readings_data
        print([ k for k, v in readings_data.items() if v!= [] ])
        with open(save_path, "w") as f:
            json.dump(readings_data, f)
        time.sleep(DELAY)       # Respect the API

