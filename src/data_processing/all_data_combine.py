import os, sys
import pandas as pd
import json
import shapely
import pickle
from tqdm import tqdm
import numpy as np
import h5py

# Local Imports
from utils import *




if __name__ == "__main__":
    # FLAGS
    NAN_FILLING = True


    all_elev = LOAD_ELEVATION()     # dict: {'75622': 909.0} station_uid to elevation
    all_landuse = LOAD_LANDUSE()    # pd.DataFrame: rows with station_uid and landuse stats
    all_era5_weather_files = LOAD_ERA5_WEATHER()
    station_info_dict = { sdict['x']: {'g': sdict['g'], 'name': sdict['n'].encode('utf-8')} for sdict in LOAD_WAQI_STATIONS() }

    station_uids = list(all_elev.keys())
    assert len(station_uids) == 14911

    earliest_date_waqi = "2019-02-06"   # The earliest date from WAQI
    earliest_date_era5 = "2019-01-01"   # Earliest date from Weather data
    latest_date_waqi = "2023-10-21"     # The latest date from WAQI
    latest_date_era5 = "2023-12-31"     # The latest date from ERA5 weather data
    possible_dates = generate_dates(earliest_date_waqi, latest_date_waqi)
    date_index_mapper = { pd:pd_idx for pd_idx, pd in enumerate(possible_dates) }
    possible_dates = [ pd.encode("utf-8") for pd in possible_dates ]
    # max_date, min_date = GET_MAX_AND_MIN_WAQI_READING_DATES() # TODO If needed to get new min/max dates
    fdf = h5py.File(ALL_DATA_PATH, 'r+')  # fdf = final data file


    ############################################################################
    # START WITH THE FEATURES THAT DONT VARY WITH TIME
    ############################################################################
    if False:
        print("Doing the fixed variables... (about a day)")
        for suid in tqdm(station_uids, total=len(station_uids)):
            print(suid)
            sg = fdf.require_group(suid)    # sg = station_group
            ############################################################################
            # STATION INFO
            sg.create_dataset('name', data=[ station_info_dict[suid]['name'] ], dtype=h5py.special_dtype(vlen=str))
            sg.create_dataset('geolocation', data=station_info_dict[suid]['g'], dtype='float64')


            #############################################################################
            # Open Street Map
            osm_path = os.path.join(OSM_FEATURE_PATH, f"{suid}.pickle")
            if os.path.exists(osm_path):
                with open(osm_path, 'rb') as f:
                    try:
                        osm = pickle.load(f)
                        osm = osm["osm_geoms"]
                    except:
                        logger.error(f"Pickle load failed: {suid}")
                        osm = None
                if isinstance(osm, type(None)):
                    osm = pd.DataFrame(columns=OSM_FEATURE_GREENLIST_DICT.keys())
            else:
                osm = pd.DataFrame(columns=OSM_FEATURE_GREENLIST_DICT.keys())

            # Consider only a certain number of features
            for col in OSM_FEATURE_GREENLIST_DICT.keys():
                if col not in osm.columns:
                    osm[col] = float('nan')
            osm = osm.drop(columns=set(osm.columns)-set(OSM_GREENLIST_COLUMNS))
            if "geometry" not in osm.columns:
                osm["geometry"] = float('nan')

            # Remove any row that contains NO valid reading
            osm = osm[osm.apply(lambda row: any(row[col] in OSM_FEATURE_GREENLIST_DICT[col] for col in OSM_FEATURE_GREENLIST_DICT.keys()), axis=1)]

            # Remove any row that does not have one of four specific geometry type
            osm = osm[osm.apply(lambda row: type(row["geometry"]) in OSM_ACCEPTABLE_GEOMETRIES, axis=1)]
            num_osm = len(osm)
            osm_group = sg.require_group("open_street_map")

            # All non-geometry features
            for greenlit_feat in OSM_FEATURE_GREENLIST_DICT.keys():
                osm_greenlit_feat_group = osm_group.require_group(greenlit_feat)
                for key in OSM_FEATURE_GREENLIST_DICT[greenlit_feat]:
                    data = osm_greenlit_feat_group.create_dataset(key, shape=(num_osm,), dtype=bool)
                    data[:] = False
                    for idx in range(num_osm):
                        data[idx] = (osm.iloc[idx][greenlit_feat] == key)

            # Geometry Features 
            geometry_group = osm_group.require_group("geometry")
            Point_data = geometry_group.create_dataset("Point", shape=(num_osm,), dtype=bool)
            Point_data[:] = False
            Polygon_data = geometry_group.create_dataset("Polygon", shape=(num_osm,), dtype=bool)
            Polygon_data[:] = False
            LineString_data = geometry_group.create_dataset("LineString", shape=(num_osm,), dtype=bool)
            LineString_data[:] = False
            MultiPolygon_data = geometry_group.create_dataset("MultiPolygon", shape=(num_osm,), dtype=bool)
            MultiPolygon_data[:] = False
            area_data = geometry_group.create_dataset("area", shape=(num_osm,), dtype="float64")
            area_data[:] = np.nan
            length_data = geometry_group.create_dataset("length", shape=(num_osm,), dtype="float64")
            length_data[:] = np.nan
            distance_from_station_data = geometry_group.create_dataset("distance_from_station", shape=(num_osm,), dtype="float64")
            distance_from_station_data[:] = np.nan
            for idx in range(num_osm):
                geo = osm.iloc[idx]['geometry']
                area = geo.area
                length = geo.length
                lat2, lon2 = geo.centroid.y, geo.centroid.x
                lat1, lon1 = sg['geolocation'][:].tolist()
                dist = haversine(lat1, lon1, lat2, lon2)  # (lat1, lon1, lat2, lon2)
                if isinstance(geo, shapely.geometry.point.Point):
                    Point_data[idx] = True
                elif isinstance(geo, shapely.geometry.polygon.Polygon):
                    Polygon_data[idx] = True
                elif isinstance(geo, shapely.geometry.linestring.LineString):
                    LineString_data[idx] = True
                elif isinstance(geo, shapely.geometry.multipolygon.MultiPolygon):
                    MultiPolygon_data[idx] = True
                else:
                    raise ValueError("Shapely polygon shoud be one of the four above types only")
                distance_from_station_data[idx] = dist 
                area_data[idx] = area
                length_data[idx] = length



            ############################################################################
            # ELEVATION
            elev = all_elev[suid]
            sg.create_dataset("elevation", data=np.array([elev]))

              
            ############################################################################
            # LANDUSE
            landuse = all_landuse.loc[all_landuse['station_id'] == int(suid)]
            landuse = landuse.drop(columns=["station_id"])
            sg.create_dataset("landuse", data=landuse)

        logger.success("Time-static variables processed!")
        fdf.close()
        sys.exit()



    if True:
        #fdf.create_dataset('encoded_dates_for_time_feats_in_index_order', data=possible_dates, dtype=h5py.special_dtype(vlen=str))
        #logger.info("Processing the time varying features, this will take longer...")
        #############################################################################
        ## Generate the initial temporal data filled with NaNs
        #logger.info("Make initial station groups... (under a minute)")
        #for suid in tqdm(station_uids, total=len(station_uids)):
        #    sg = fdf.require_group(suid)    # sg = station_group
        #    #sg = fdf[suid]
        #    waqi_group = sg.require_group("waqi_readings")
        #    ############################################################################
        #    # WAQI Readings
        #    for pol in WAQI_POLLUTANTS:
        #        pol_group = waqi_group.require_group(pol)
        #        for pol_feat in ["q1", "q3", "max", "min", "median", "mean", "stdev", "count"]:
        #            pg_dataset = pol_group.create_dataset(pol_feat, shape=(len(possible_dates),), dtype='float64')
        #            if NAN_FILLING:
        #                pg_dataset[:] = np.nan
        #        #for pol_feat in ["count"]:
        #        #    pg_dataset = pol_group.create_dataset(pol_feat, shape=(len(possible_dates),), dtype='uint8')
        #        #    pg_dataset[:] = 0

        #    ############################################################################
        #    # Weather ERA5
        #    era5_group = sg.require_group("era5_weather")
        #    for efeat in ERA5_FEATS:
        #        # 24* the number of dates because we have hourly ERA5 estimations for each day
        #        ef_dataset = era5_group.create_dataset(efeat, shape=(len(possible_dates)*24,), dtype='float64')
        #        if NAN_FILLING:
        #            ef_dataset[:] = np.nan

        #logger.success("Time-varying features have been initialised!")




        ############################################################################
        # Overwrite the NaNs with existing data
        #logger.info("Filling in WAQI readings... (takes a few hours)")
        #for suid in tqdm(station_uids, total=len(station_uids)):
        #    waqi_group = fdf[suid]["waqi_readings"]
        #    ############################################################################
        #    # WAQI Readings
        #    r_path = os.path.join( os.path.join(READINGS_PATH, f"{suid}.json") )
        #    with open(r_path, 'r') as f:
        #        readings = json.load(f)
        #    _ = readings.pop("station_data")
        #    for pollutant, r_data in readings.items(): # pm25, pm10, pm1, o3, no2, so2, co
        #        pol_group = waqi_group[pollutant]
        #        for r_d in r_data:
        #            date = r_d['day']
        #            date_idx = date_index_mapper[date]
        #            pol_group['q1'][date_idx] = r_d['q1']
        #            pol_group['q3'][date_idx] = r_d['q3']
        #            pol_group['max'][date_idx] = r_d['max']
        #            pol_group['median'][date_idx] = r_d['median']
        #            pol_group['min'][date_idx] = r_d['min']
        #            pol_group['stdev'][date_idx] = r_d['stdev']
        #            pol_group['count'][date_idx] = r_d['count']
        #logger.success("WAQI readings processed!")


        logger.info("Filling in Weather ERA5 readings... (70 hours)")
        for eidx, era5_dayf in tqdm(enumerate(all_era5_weather_files), total=len(all_era5_weather_files)):
            print(era5_dayf)
            if eidx < 30:
                continue
            e_path = os.path.join(ERA5_PATH, era5_dayf)
            try: 
                with open(e_path, 'r') as f:
                    era5_f = json.load(f)
            except:
                try:
                    with open(e_path, 'r') as f:
                        era5_f = json.load(f)
                except:
                    with open(e_path, 'r') as f:
                        era5_f = json.load(f)
            for suidx, suid in tqdm(enumerate(station_uids), total=len(station_uids)):
                ############################################################################       
                # Weather ERA5
                era5_group = fdf[suid]["era5_weather"]
                era5 = era5_f[suid]
                for date, era5_data in era5.items():
                    dlist = date.split("T")
                    if len(dlist) == 1:
                        day = date[0:10]
                        hour = date[-18:]
                    else:
                        day = dlist[0]
                        hour = dlist[1]
                    hour = int(hour.split(":")[0])
                    if day not in date_index_mapper:
                        continue
                    day = date_index_mapper[day]
                    date_idx = (day*24)+hour
                    for ef in ERA5_FEATS:
                        era5_group[ef][date_idx] = era5_data.get(ef, np.nan)
        logger.success("ERA5 weather readings complete!")
