import os, sys
import numpy as np
import h5py
from tqdm import tqdm
from loguru import logger


data = h5py.File(os.path.join(os.path.dirname(__file__), "../all_data.h5"), "r")



non_nan_completeness = {}



OPENSTREETMAP = True
if OPENSTREETMAP:
    for osm_g1 in ["generator:method", "generator:source", "generator:type", "landuse", 'power']:
        for osm_g2 in data[f"stations/99490/open_street_map/{osm_g1}/"].keys():
            non_nan_completeness[f"{osm_g1}/{osm_g2}"] = []

    # Total area calculation
    for sui in tqdm(data["stations"], total=len(data['stations'])):
        for osm_g1 in ["generator:method", "generator:source", "generator:type", "landuse", 'power']:
            for osm_g2 in data[f"stations/{sui}/open_street_map/{osm_g1}/"].keys():
                keyed_bool = (data[f"stations/{sui}/open_street_map/{osm_g1}/{osm_g2}"][:] == True)
                assert keyed_bool.shape == data[f"stations/{sui}/open_street_map/geometry/area"].shape
                keyed_areas = data[f"stations/{sui}/open_street_map/geometry/area"][keyed_bool]
                # Just area
                #non_nan_completeness[f"{osm_g1}/{osm_g2}"].append(np.sum(keyed_areas))
                decayed_distance = np.exp(-0.25 * data[f"stations/{sui}/open_street_map/"]['geometry']['distance_from_station'][:])[keyed_bool]
                scaled_areas = keyed_areas * decayed_distance
                non_nan_completeness[f"{osm_g1}/{osm_g2}"].append(np.sum(scaled_areas))

    ## Present at site, total count
    #for sui in tqdm(data["stations"], total=len(data['stations'])):
    #    for osm_g1 in ["generator:method", "generator:source", "generator:type", "landuse", 'power']:
    #        for osm_g2 in data[f"stations/{sui}/open_street_map/{osm_g1}/"].keys():
    #            non_nan_completeness[f"{osm_g1}/{osm_g2}"].append(sum(data[f"stations/{sui}/open_street_map/{osm_g1}/{osm_g2}"][:] == True))

    #for key in non_nan_completeness:
    #    print(f"{key}: {sum([1 for val in non_nan_completeness[key] if val > 0])}")

    for key in non_nan_completeness:
        print(f"{key}: {sum(non_nan_completeness[key])}")









AIRPOL = False
if AIRPOL:
    # for each station count how many of each feature "count", "max", "min", "mean", "median", "stdev", "q1", "q3" have at least one non-nan value
    non_nan_completeness["count"] = []
    non_nan_completeness["max"] = []
    non_nan_completeness["min"] = []
    non_nan_completeness["mean"] = []
    non_nan_completeness["median"] = []
    non_nan_completeness["stdev"] = []
    non_nan_completeness["q1"] = []
    non_nan_completeness["q3"] = []
    pol = "so2"
    print(pol)
    for suid in tqdm(data["stations"], total=len(data['stations'])):
        feat = data[f"stations/{suid}/waqi_readings/{pol}"]
        # get the number of nan values for "count"
        feat_count = feat[f"count"][:]
        nan_count = sum(np.isnan(feat_count))
        # get the number of nan values for "max"
        feat_max = feat[f"max"][:]
        nan_max = sum(np.isnan(feat_max))
        # repeat this for "min", "mean", "median", "stdev", "q1", "q3"
        feat_min = feat[f"min"][:]
        nan_min = sum(np.isnan(feat_min))

        feat_mean = feat[f"mean"][:]
        nan_mean = sum(np.isnan(feat_mean))

        feat_median = feat[f"median"][:]
        nan_median = sum(np.isnan(feat_median))

        feat_stdev = feat[f"stdev"][:]
        nan_stdev = sum(np.isnan(feat_stdev))

        feat_q1 = feat[f"q1"][:]
        nan_q1 = sum(np.isnan(feat_q1))

        feat_q3 = feat[f"q3"][:]
        nan_q3 = sum(np.isnan(feat_q3))

        # calculate the proporitons of non-nan values for each feature
        non_nan_completeness["count"].append((len(feat_count) - nan_count) / len(feat_count))
        non_nan_completeness["max"].append((len(feat_max) - nan_max) / len(feat_max))
        non_nan_completeness["min"].append((len(feat_min) - nan_min) / len(feat_min))
        non_nan_completeness["mean"].append((len(feat_mean) - nan_mean) / len(feat_mean))
        non_nan_completeness["median"].append((len(feat_median) - nan_median) / len(feat_median))
        non_nan_completeness["stdev"].append((len(feat_stdev) - nan_stdev) / len(feat_stdev))
        non_nan_completeness["q1"].append((len(feat_q1) - nan_q1) / len(feat_q1))
        non_nan_completeness["q3"].append((len(feat_q3) - nan_q3) / len(feat_q3))
        
    # for each of "count", "max", "min", "mean", "median", "stdev", "q1", "q3" print the number of stations that have at least one non-nan value
    for key in non_nan_completeness:
        print(f"{key}: {sum([1 for val in non_nan_completeness[key] if val > 0])}")
        non_nan_completeness["mean"].append((len(feat_mean) - nan_mean) / len(feat_mean))
        non_nan_completeness["median"].append((len(feat_median) - nan_median) / len(feat_median))
        non_nan_completeness["stdev"].append((len(feat_stdev) - nan_stdev) / len(feat_stdev))
        non_nan_completeness["q1"].append((len(feat_q1) - nan_q1) / len(feat_q1))
        non_nan_completeness["q3"].append((len(feat_q3) - nan_q3) / len(feat_q3))
        
    # for each of "count", "max", "min", "mean", "median", "stdev", "q1", "q3" print the number of stations that have at least one non-nan value
    for key in non_nan_completeness:
        print(f"{key}: {sum([1 for val in non_nan_completeness[key] if val > 0])}")
    # also print the mean of the proportions of non-nan values for each feature
    for key in non_nan_completeness:
        print(f"{key}: {100*np.mean(non_nan_completeness[key])}")




WEATHER = False
if WEATHER:
    # initialise non_nan_completeness dictionary for '10m_u', '10m_v', 'skin_temp', 'surface_pressure'
    non_nan_completeness["10m_u"] = []
    non_nan_completeness["10m_v"] = []
    non_nan_completeness["skin_temp"] = []
    non_nan_completeness["surface_pressure"] = []
    for suid in tqdm(data["stations"], total=len(data['stations'])):
        feat = data[f"stations/{suid}/era5_weather"]
        # get the number of nan values for "10m_u"
        feat_10m_u = feat[f"f_10m_u_component_of_wind"][:]
        nan_10m_u = sum(np.isnan(feat_10m_u))
        # get the number of nan values for "10m_v"
        feat_10m_v = feat[f"f_10m_v_component_of_wind"][:]
        nan_10m_v = sum(np.isnan(feat_10m_v))
        # repeat this for "skin_temp", "surface_pressure"
        feat_skin_temp = feat[f"f_skin_temperature"][:]
        nan_skin_temp = sum(np.isnan(feat_skin_temp))

        feat_surface_pressure = feat[f"f_surface_pressure"][:]
        nan_surface_pressure = sum(np.isnan(feat_surface_pressure))

        # calculate the proporitons of non-nan values for each feature
        non_nan_completeness["10m_u"].append((len(feat_10m_u) - nan_10m_u) / len(feat_10m_u))
        non_nan_completeness["10m_v"].append((len(feat_10m_v) - nan_10m_v) / len(feat_10m_v))
        non_nan_completeness["skin_temp"].append((len(feat_skin_temp) - nan_skin_temp) / len(feat_skin_temp))
        non_nan_completeness["surface_pressure"].append((len(feat_surface_pressure) - nan_surface_pressure) / len(feat_surface_pressure))

    # for each of '10m_u', '10m_v', 'skin_temp', 'surface_pressure' print the number of stations that have at least one non-nan value
    for key in non_nan_completeness:
        print(f"{key}: {sum([1 for val in non_nan_completeness[key] if val > 0])}")
    # also print the mean of the proportions of non-nan values for each feature
    for key in non_nan_completeness:
        print(f"{key}: {100*np.mean(non_nan_completeness[key])}")



LANDUSE = False
if LANDUSE:
    # repeat the above for landuse features: tree_cover, shrubland, grassland, cropland, built_up, bare_sparse_vegetation, snow_and_ice, permanent_water_bodies, herbaceous_wetland, mangroves, moss_and_lichen
    feat_dict = {}
    all_feats = []
    feat_dict["tree_cover"] = []
    feat_dict["shrubland"] = []
    feat_dict["grassland"] = []
    feat_dict["cropland"] = []
    feat_dict["built_up"] = []
    feat_dict["bare_sparse_vegetation"] = []
    feat_dict["snow_and_ice"] = []
    feat_dict["permanent_water_bodies"] = []
    feat_dict["herbaceous_wetland"] = []
    feat_dict["mangroves"] = []
    feat_dict["moss_and_lichen"] = []
    
    for suid in tqdm(data["stations"], total=len(data['stations'])):
        feat = data[f"stations/{suid}/landuse"][0]
        all_feats.append(feat.sum())
        # feat.shape = (11)
        # get the amount of tree_cover at feat[0]
        feat_tree_cover = feat[0]
        # get the amount of shrubland at feat[1]
        feat_shrubland = feat[1]
        # repeat this for grassland, cropland, built_up, bare_sparse_vegetation, snow_and_ice, permanent_water_bodies, herbaceous_wetland, mangroves, moss_and_lichen
        feat_grassland = feat[2]
        if not(any(np.isnan(feat))):
            feat_dict["tree_cover"].append(feat_tree_cover/feat.sum())
            feat_dict["shrubland"].append(feat_shrubland/feat.sum())
            feat_dict["grassland"].append(feat_grassland/feat.sum())
            feat_dict["cropland"].append(feat_cropland/feat.sum())
            feat_dict["built_up"].append(feat_built_up/feat.sum())
            feat_dict["bare_sparse_vegetation"].append(feat_bare_sparse_vegetation/feat.sum())
            feat_dict["snow_and_ice"].append(feat_snow_and_ice/feat.sum())
            feat_dict["permanent_water_bodies"].append(feat_permanent_water_bodies/feat.sum())
            feat_dict["herbaceous_wetland"].append(feat_herbaceous_wetland/feat.sum())  
            feat_dict["mangroves"].append(feat_mangroves/feat.sum())
            feat_dict["moss_and_lichen"].append(feat_moss_and_lichen/feat.sum())

    # find the number of nan values for each feature
    for key in feat_dict:
        print(f"{key}: {sum(np.isnan(feat_dict[key]))}")

    for key in feat_dict:
        logger.success(f"{key}: {100*np.mean(feat_dict[key])}")
        logger.warning(f"{key}: {100*np.std(feat_dict[key])}")


    #print(f"tree_cover: {np.mean(feat_dict['tree_cover'])}")
    #print(f"shrubland: {np.mean(feat_dict['shrubland'])}")
    #print(f"grassland: {np.mean(feat_dict['grassland'])}")
    #print(f"cropland: {np.mean(feat_dict['cropland'])}")
    #print(f"built_up: {np.mean(feat_dict['built_up'])}")
    #print(f"bare_sparse_vegetation: {np.mean(feat_dict['bare_sparse_vegetation'])}")
    #print(f"snow_and_ice: {np.mean(feat_dict['snow_and_ice'])}")
    #print(f"permanent_water_bodies: {np.mean(feat_dict['permanent_water_bodies'])}")
    #print(f"herbaceous_wetland: {np.mean(feat_dict['herbaceous_wetland'])}")
    #print(f"mangroves: {np.mean(feat_dict['mangroves'])}")
    #print(f"moss_and_lichen: {np.mean(feat_dict['moss_and_lichen'])}")


        




ELEVATION = False
if ELEVATION:
    all_elevations = []
    for suid in tqdm(data["stations"], total=len(data['stations'])):
        feat = data[f"stations/{suid}/elevation"][0]
        all_elevations.append(feat)
