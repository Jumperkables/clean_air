import os, sys
import time
import json
import requests
from tqdm import tqdm

# Local Imports
from utils import *

POLITENESS_DELAY = 2.0

if __name__ == "__main__":
    waqi_stations = LOAD_WAQI_STATIONS() 
    elevation_dict = {}
    for row in tqdm(waqi_stations, total=len(waqi_stations)):
        lat = row["g"][0]
        lon = row["g"][1]
        uid = row["x"]
        url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
        response = requests.get(url)
        data = response.json()
        elev = data["results"][0]["elevation"]
        assert type(elev) == float
        elevation_dict[uid] = elev
        time.sleep(POLITENESS_DELAY)
    with open(ELEVATION_FEATURES_PATH, "w") as f:
        json.dump(elevation_dict, f)


