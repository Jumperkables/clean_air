import os, sys
from math import radians, sin, cos, sqrt, atan2
import json
import pandas as pd
import pickle
import shapely
from tqdm import tqdm
from loguru import logger
from datetime import datetime, timedelta

# PATHS
ROOT_DIR = os.path.abspath(os.path.expanduser("../../"))
DATA_DIR = os.path.join(ROOT_DIR, "data")
ALL_DATA_PATH       = os.path.join(DATA_DIR, "ALLDATA", "all_data.h5")
STATIONS_PATH       = os.path.join(DATA_DIR, "WAQIData", "waqi_stations.json")
READINGS_PATH       = os.path.join(DATA_DIR, "WAQIData", "sensor_readings")
#OSM_FEATURE_PATH    = os.path.join(DATA_DIR, "OpenStreetMapData", "all_stations_osm.pickle")
OSM_FEATURE_PATH    = os.path.join(DATA_DIR, "OpenStreetMapData", "processed")
LANDUSE_PATH        = os.path.join(DATA_DIR, "landuse_map")
ERA5_PATH           = os.path.join(DATA_DIR, "ERA5Weather")
LANDUSE_FEATURES_PROCESSED_PATH = os.path.join(DATA_DIR, "landuse_processed")#"landuse_stats.json")
LANDUSE_FEATURES_PATH = os.path.join(DATA_DIR, "landuse_final.csv")
ELEVATION_FEATURES_PATH = os.path.join(DATA_DIR, "elevation_stats.json")

RADIUS_AROUND_STATION = 15000 # Consider everything 15000 meters around the station
LANDUSE_IMAGE_SIZE = 36000
WAQI_POLLUTANTS = ["pm25", "pm10", "pm1", "o3", "no2", "so2", "co"]
ERA5_FEATS = ["f_surface_pressure", "f_10m_u_component_of_wind", "f_10m_v_component_of_wind", "f_skin_temperature"]


OSM_TAGS = {
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

OSM_GREENLIST_COLUMNS = [
    'geometry', 'power', 'nodes', 'generator:source', 'generator:method', 'landuse', 'generator:type', 'generator:output:electricity'
]

OSM_ACCEPTABLE_GEOMETRIES = [
    shapely.geometry.point.Point,
    shapely.geometry.polygon.Polygon,
    shapely.geometry.linestring.LineString,
    shapely.geometry.multipolygon.MultiPolygon,
]

OSM_FEATURE_GREENLIST_DICT = {
    "landuse": [
        'industrial', 'quarry', 'highway', 'farm'
    ],
    "power": [
        'tower', 'pole', 'generator', 'substation', 'line', 'minor_line', 'transformer', 'cable', 'switch', 'plant', 'terminal', 'converter', 'station'
    ],
    "generator:method": [
        'photovoltaic', 'combustion', 'wind_turbine', 'thermal', 'water-storage', 'run-of-the-river', 'battery-storage', 'solar_thermal', 'motor_generator', 'flywheel-storage', 'fission', 'anaerobic_digestion', 'internal_combustion', 'nuclear_fusion'
    ],
    "generator:type": [
        'solar_photovoltaic_panel', 'solar_thermal_collector', 'gas_turbine', 'reciprocating_engine', 'steam_turbine', 'combustion', 'coal', 'bioreactor', 'solid_oxide_fuel_cell', 'francis_turbine', 'horizontal_axis'
    ],
    "generator:source": [
        'solar', 'gas', 'wind','diesel','oil','hydro','gasoline','coal','biogas','battery','biomass','waste','nuclear','gas;oil','flywheel','landfill_gas','fossil','geothermal','electricity_network','electricity','generator','biofuel','solar;wind','steam','tidal'
    ],
}

####################################################################################################
# ERA5 WEATHER
def LOAD_ERA5_WEATHER():
    logger.info("Returning ERA5 json file paths...")
    return(sorted([ ep for ep in os.listdir(ERA5_PATH) if ep.endswith(".json") ]))

####################################################################################################
# Landuse
def LOAD_LANDUSE():
    logger.info("Loading landuse features...")
    #with open(LANDUSE_FEATURES_PATH, 'r') as f:
    #    landuse = json.load(f)
    landuse = pd.read_csv(LANDUSE_FEATURES_PATH, index_col=0)
    print(f"Loaded {len(landuse)} stations worth of elevation data!")
    return landuse
####################################################################################################


####################################################################################################
# Elevation
def LOAD_ELEVATION():
    logger.info("Loading already processed elevation readings...")
    with open(ELEVATION_FEATURES_PATH, 'r') as f:
        ele_data = json.load(f)
    print(f"Loaded {len(ele_data)} stations worth of elevation data!")
    return ele_data
####################################################################################################


####################################################################################################
# Open Street Map
def LOAD_OSM_FEATURES():
    logger.info("Loading already processed geographical data for each WAQI station...")
    all_osm_features = []
    osm_files = os.listdir(OSM_FEATURE_PATH)[:1000]
    for osmf in tqdm(osm_files, total=len(osm_files)):
        osmf = os.path.join(OSM_FEATURE_PATH, osmf)
        with open(osmf, "rb") as f:
            osm_features = pickle.load(f)
        all_osm_features.append(osm_features)
    logger.info(f"Loaded {len(all_osm_features)} stations worth of processed OSM data!")
    return all_osm_features
####################################################################################################


####################################################################################################
# WAQI
def LOAD_WAQI_STATIONS():
    logger.info("Loading WAQI station information. lats, longs etc...")
    #waqi_stat_info = pd.read_csv(STATIONS_PATH)
    with open(STATIONS_PATH, "r") as f:
        waqi_stat_info = json.load(f)
    logger.info(f"Loaded {len(waqi_stat_info)} stations worth of data!")
    return waqi_stat_info

def LOAD_WAQI_READINGS():
    logger.info("Loading scraped WAQI readings...")
    all_readings_json_files = os.listdir(READINGS_PATH)
    all_readings = []
    for jf in tqdm(all_readings_json_files, total=len(all_readings_json_files)):
        jf = os.path.join(READINGS_PATH, jf)
        with open(jf, 'r') as f:
            all_readings.append(json.load(f))
    logger.info(f"Loaded {len(all_readings)} statiosn worth of air quality readings!")    
    return all_readings

def GET_MAX_AND_MIN_WAQI_READING_DATES():
    from datetime import datetime
    waqi_readings = LOAD_WAQI_READINGS()
    print(len(waqi_readings))
    all_dates = []
    for r_dict in tqdm(waqi_readings, total=len(waqi_readings)):
        r_dict.pop("station_data")
        for _, v in r_dict.items(): # k = polluant, v = dictionary of readings for that pollutant
            for rsd in v:   # rsd = reading sub dict
                date = rsd['day']
                date = datetime.strptime(date, '%Y-%m-%d').date()
                all_dates.append(date)
    #print(max(all_dates))
    #print(min(all_dates))
    return(max(all_dates), min(all_dates))

def get_waqi_token():
    with open( os.path.dirname( os.path.dirname( ".WAQI_API_TOKEN" ) ), "r") as f:
        TOKEN = f.read().split("\n")[0]
    return TOKEN
####################################################################################################



# Calculate the distance between 2 geolocations using haversine formula
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert latitude and longitude from decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371.0088 * c  # Radius of the Earth in kilometers
    return distance













#########################################################################################################
#####################
# TODO Potentially remove
#####################
# TODO Potentially remove
def generate_dates(start_date, end_date):
    # Convert start and end dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    # Initialize an empty list to store generated dates
    all_dates = []

    # Loop through all dates between start and end dates
    current_date = start
    while current_date <= end:
        # Append the current date to the list of all dates
        all_dates.append(current_date.strftime('%Y-%m-%d'))
        # Move to the next day
        current_date += timedelta(days=1)
    return all_dates





def gpd_geographic_area(geodf):
    if not geodf.crs and geodf.crs.is_geographic:
        raise TypeError('geodataframe should have geographic coordinate system')
        
    geod = geodf.crs.get_geod()
    def area_calc(geom):
        if geom.geom_type not in ['MultiPolygon','Polygon']:
            return np.nan
        
        # For MultiPolygon do each separately
        if geom.geom_type=='MultiPolygon':
            return np.sum([area_calc(p) for p in geom.geoms])

        # orient to ensure a counter-clockwise traversal. 
        # See https://pyproj4.github.io/pyproj/stable/api/geod.html
        # geometry_area_perimeter returns (area, perimeter)
        return geod.geometry_area_perimeter(orient(geom, 1))[0]
    
    return geodf.geometry.apply(area_calc)


def line_integral_polygon_area(geom, radius = 6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth. 
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    
    from https://stackoverflow.com/a/61184491/6615512
    """
    if geom.geom_type not in ['MultiPolygon','Polygon']:
        return np.nan

    # For MultiPolygon do each separately
    if geom.geom_type=='MultiPolygon':
        return np.sum([line_integral_polygon_area(p) for p in geom.geoms])

    # parse out interior rings when present. These are "holes" in polygons.
    if len(geom.interiors)>0:
        interior_area = np.sum([line_integral_polygon_area(Polygon(g)) for g in geom.interiors])
        geom = Polygon(geom.exterior)
    else:
        interior_area = 0
        
    # Convert shapely polygon to a 2 column numpy array of lat/lon coordinates.
    geom = np.array(geom.boundary.coords)

    lats = np.deg2rad(geom[:,1])
    lons = np.deg2rad(geom[:,0])

    # Line integral based on Green's Theorem, assumes spherical Earth

    #close polygon
    if lats[0]!=lats[-1]:
        lats = np.append(lats, lats[0])
        lons = np.append(lons, lons[0])

    #colatitudes relative to (0,0)
    a = np.sin(lats/2)**2 + np.cos(lats)* np.sin(lons/2)**2
    colat = 2*np.arctan2( np.sqrt(a), np.sqrt(1-a) )

    #azimuths relative to (0,0)
    az = np.arctan2(np.cos(lats) * np.sin(lons), np.sin(lats)) % (2*np.pi)

    # Calculate diffs
    # daz = np.diff(az) % (2*pi)
    daz = np.diff(az)
    daz = (daz + np.pi) % (2 * np.pi) - np.pi

    deltas=np.diff(colat)/2
    colat=colat[0:-1]+deltas

    # Perform integral
    integrands = (1-np.cos(colat)) * daz

    # Integrate 
    area = abs(sum(integrands))/(4*np.pi)

    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return (area * 4*np.pi*radius**2) - interior_area
    else: #return in ratio of sphere total area 
        return area - interior_area
        
# a wrapper to apply the method to a geo data.frame
def gpd_geographic_area_line_integral(geodf):
    return geodf.geometry.apply(line_integral_polygon_area)
