import os, sys

STATIONS_PATH = os.path.join(root_dir, "data", "WAQIData", "all_stations.csv")
OSM_FEATURE_PATH = os.path.join(root_dir, "data", "OpenStreetMapData", "all_stations_osm.pickle")
READINGS_PATH = os.path.join(root_dir, "data", "OpenStreetMapData", "historical_readings.json")

def get_waqi_token():
    with open( os.path.dirname( os.path.dirname( ".WAQI_API_TOKEN" ) ), "r") as f:
        TOKEN = f.read().split("\n")[0]
    return TOKEN
