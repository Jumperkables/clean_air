# Standard Imports
import os, sys
import osmnx as ox
import pandas as pd
import time
from tqdm import tqdm
import pickle

# Local Imports
from utils import *

tags = {
    "power": True,
    "man_made": ["petroleum_well", "monitoring_station"],
    "landuse": ["industrial", "highway", "motorway", "quarry", "depot", "farm"],
}


if __name__ == "__main__":
    waqi_stations = LOAD_WAQI_STATIONS()
    all_data = {}
    for row in tqdm(waqi_stations, total=len(waqi_stations)):
        datum = {}
        lat = row["g"][0]
        lon = row["g"][1]
        uid = row["x"]
        name = row["n"]

        save_path = os.path.join(OSM_FEATURE_PATH, f"{uid}.pickle")
        if os.path.exists(save_path):
            continue

        datum["lat"] = lat
        datum["lon"] = lon
        datum["uid"] = uid
        datum["name"] = name
        if (lat == None) or (lon == None):
            continue
        place = (lat, lon)
        try:
            p_areas = ox.features_from_point(place, dist=RADIUS_AROUND_STATION, tags=tags)#ox.geometries_from_point(place, dist=RADIUS_AROUND_STATION, tags=tags)
        except:
            p_areas = None
            print(f"{uid} failed")
        datum["osm_geoms"] = p_areas
        save_path = os.path.join(OSM_FEATURE_PATH, f"{uid}.pickle")
        with open(save_path, "wb") as f:
            pickle.dump(datum, f)
        time.sleep(2)
