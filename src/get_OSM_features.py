# Standard Imports
import os, sys
import osmnx as ox
import pandas as pd
import time
from tqdm import tqdm
import pickle

# Local Imports
import utils

tags = {
    "power": "generator",
    "man_made": "petroleum_well",
    "man_made":"monitoring_station",
    "landuse": "industrial",
    "landuse": "highway",
    "landuse": "motorway",
    "landuse": "quarry",
    "landuse": "depot",
    "landuse": "farm",
}


if __name__ == "__main__":
    root_dir = os.path.dirname( os.path.dirname(__file__) )
    df = pd.read_csv(utils.STATIONS_PATH)
    all_data = {}
    for row in tqdm(df.iterrows(), total=len(df)):
        datum = {}
        lat = row[1]["lat"]
        lon = row[1]["lon"]
        uid = row[1]["uid"]
        name = row[1]["name"]
        datum["lat"] = lat
        datum["lon"] = lon
        datum["uid"] = uid
        datum["name"] = name
        if (lat == None) or (lon == None):
            continue
        place = (lat, lon)
        p_areas = ox.geometries_from_point(place, dist=15000, tags=tags)
        datum["osm_geoms"] = p_areas
        all_data[uid] = datum
        time.sleep(3)
    with open(utils.OSM_FEATURE_PATH, "wb") as f:
        pickle.dump(all_data, f)
