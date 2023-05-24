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
import utils

# Get Token
TOKEN = utils.get_waqi_token()


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
    df = pd.read_csv("all_stations.csv")
    base_url = "https://airnet.waqi.info/airnet/sse/historic/daily"
    species = ["pm25", "pm10", "pm1", "o3", "no2", "so2", "co"]
    all_data = {}
    for idx, row in tqdm(enumerate(df.iterrows()), total=len(df)):
        uid = row[1]["uid"]
        lat = row[1]["lat"]
        lon = row[1]["lon"]
        name = row[1]["name"]
        station_url = f"{base_url}/{uid}"
        r = requests.get(station_url)
        if r.status_code != 200:
            time.sleep(0.5)
            continue
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
            time.sleep(1)   # Respect the API
        all_data[uid] = readings_data
        time.sleep(1)       # Respect the API
    with open(utils.READINGS_PATH, "w") as f:
        json.dump(all_data, f)
