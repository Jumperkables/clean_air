# clean_air
Placeholder repo for the clean air work for before i choose a name.

## Installation
- `git clone git@github.com:Jumperkables/clean_air.git`
- Make and source a python virtual env
- `pip install -r requirements.txt`

## Data Preparation
TODO RE-TEST THIS
0. Acquire a [WAQI API access token](https://aqicn.org/data-platform/token/). Save it in this directory as `.WAQI_API_TOKEN` 
1. `python src/get_aqicn_station.py`: Generate a csv file for all stations. Includes name, id, lat-lon co-ords.
You can run the next two steps at the same time if you want.
2. `python src/get_OSM_features.py`: For each station, get the relevant OpenStreetMap features.
3. `python src/get_aqicn_readings.py`: Get all available readings for particulates at daily granularity.

## Data Explortation
`python src/explore_osm.py`
