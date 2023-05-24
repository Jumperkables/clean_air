__author__ = "Jumperkables (Tom Winterbottom)"

# Standard Imports
import os, sys
import time
from loguru import logger
from tqdm import tqdm
import pandas as pd
import requests

# Local Imports
import utils

# Get Token
TOKEN = utils.get_waqi_token()


def enumerate_latlngs(step=4):
    """
    Return a list of latlng co-ordinate pairs such that the earth is fully covered in steps of 'step'
    e.g. '-90,-180,-89,-179' -> '-89,-180,-88,-179' -> ...
    """
    latlngs = []
    for lat0 in range(-90, 90, step):
        for lng0 in range(-180, 180, step):
            lat1 = lat0+step
            lng1 = lng0+step
            latlng = f"{lat0},{lng0},{lat1},{lng1}"
            latlngs.append(latlng)
    return latlngs



#"""
#Get the lat, long, and name of all sensors
#"""
if __name__ == "__main__":
    #url = "https://aqicn.org/map/world/"
    step = 4
    all_data = []
    uids = []
    latlngs = enumerate_latlngs(step=step)
    logger.info(f"Number of Lat Lng regions by step {step}: {len(latlngs)}")
    for idx, latlng in tqdm(enumerate(latlngs), total=len(latlngs)):
        #latlng="22,74,26,80" NOTE DEBUG
        url = f"https://api.waqi.info/v2/map/bounds?token={TOKEN}&latlng={latlng}"
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception(f"Requests status {r.status_code}. Wanted 200 (ofc)")
        r = r.json()
        data = r["data"]
        for datum in data:
            uid = datum.get("uid", None)
            lat = datum.get("lat", None)
            lon = datum.get("lon", None)
            aqi = datum.get("aqi", None)
            station = datum.get("station", {"name": None, "time": None} )
            name = station.get("name", None)
            t = station.get("time", None)
            data_dict = {
                "uid": uid,
                "lat": lat,
                "lon": lon,
                "aqi": aqi,
                "name": name,
                "time": t,
            }
            all_data.append(data_dict)
            uids.append(uid)
        time.sleep(1)
    df = pd.DataFrame(all_data)
    df.to_csv(utils.STATIONS_PATH)
