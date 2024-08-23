import os, sys
import json
from PIL import Image
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

# Local Imports
from utils import *

Image.MAX_IMAGE_PIXELS = None   # These are some damn big images



class LanduseTiffTile():
    def __init__(self, lat: float, lon: float, lat_0: int, lon_0: int, calc_neighbour_tiles: bool):
        """
        lat :- The latitude of the point of interest
        lon :- The longitude of the point of interest
        lat_0 :- Lower left corner's latitude
        lon_0 :- Lower left corner's longitude
            Both must be multiples of 3
        calc_neighbour_tiles :- Weather or not to instantiate internal TiffTiles for neighbours
        """
        self.lat = lat
        self.lon = lon

        # Get the needed constants for radius calculation
        radius_around_station_pixels = RADIUS_AROUND_STATION / 10 # 10m per pixel
        degree_per_pixel = 3/36000
        radius_around_station_degree = degree_per_pixel * radius_around_station_pixels

        assert (lat_0%3) == 0
        assert (lon_0%3) == 0
        self.corners = {
            "lower_left"    : {"lat": lat_0  , "lon": lon_0  },
            "upper_left"    : {"lat": lat_0+3, "lon": lon_0  },
            "lower_right"   : {"lat": lat_0  , "lon": lon_0+3},
            "upper_right"   : {"lat": lat_0+3, "lon": lon_0+3},
        }
        if lat_0 >= 0:
            lat_dir = "N"   # North
        else:
            lat_dir = "S"   # South
        if lon_0 >= 0:
            lon_dir = "E"   # East
        else:
            lon_dir = "W"   # West
        tile_code = f"{lat_dir}{abs(lat_0):02d}{lon_dir}{abs(lon_0):03d}"
        self.tiff_path = os.path.join(LANDUSE_PATH, f"ESA_WorldCover_10m_2021_v200_{tile_code}_Map.tif")

        # Now create the tile object for each neighbouring tile
        self.needed_tiles = {}
        self.north_tile      = None
        self.north_east_tile = None
        self.east_tile       = None
        self.south_east_tile = None
        self.south_tile      = None
        self.south_west_tile = None
        self.west_tile       = None
        self.north_west_tile = None
        if calc_neighbour_tiles:
            self.needed_tiles["center"] = LanduseTiffTile(lat, lon, lat_0, lon_0  , calc_neighbour_tiles=False)
            ################################################
            # North
            self.north_tile         = LanduseTiffTile(lat, lon, lat_0+3, lon_0  , calc_neighbour_tiles=False)
            if abs((lat_0+3)-lat) < radius_around_station_degree:
                self.needed_tiles["north"] = self.north_tile
            # East
            self.east_tile  = LanduseTiffTile(lat, lon, lat_0  , lon_0+3, calc_neighbour_tiles=False)
            if abs((lon_0+3)-lon) < radius_around_station_degree:
                self.needed_tiles["east"] = self.east_tile
            # South
            self.south_tile = LanduseTiffTile(lat, lon, lat_0-3, lon_0  , calc_neighbour_tiles=False)
            if abs((lat_0)-lat) < radius_around_station_degree:
                self.needed_tiles["south"] = self.south_tile
            # West
            self.west_tile  = LanduseTiffTile(lat, lon, lat_0  , lon_0-3, calc_neighbour_tiles=False)
            if abs((lon_0)-lon) < radius_around_station_degree:
                self.needed_tiles["west"] = self.west_tile

            #################################################
            # North East
            self.north_east_tile    = LanduseTiffTile(lat, lon, lat_0+3, lon_0+3, calc_neighbour_tiles=False)
            if ( ( abs( lat_0+3-lat )**2 + abs( lon_0+3-lon  )**2 )**0.5 ) < radius_around_station_degree:
                self.needed_tiles["north_east"] = self.north_east_tile
            # South East
            self.south_east_tile    = LanduseTiffTile(lat, lon, lat_0-3, lon_0+3, calc_neighbour_tiles=False)
            if ( ( abs( lat_0  -lat )**2 + abs( lon_0+3-lon  )**2 )**0.5 ) < radius_around_station_degree:
                self.needed_tiles["south_east"] = self.south_east_tile
            # South West
            self.south_west_tile    = LanduseTiffTile(lat, lon, lat_0-3, lon_0-3, calc_neighbour_tiles=False)
            if ( ( abs( lat_0  -lat )**2 + abs( lon_0  -lon  )**2 )**0.5 ) < radius_around_station_degree:
                self.needed_tiles["south_west"] = self.south_west_tile
            # North West
            self.north_west_tile    = LanduseTiffTile(lat, lon, lat_0+3, lon_0-3, calc_neighbour_tiles=False)
            if ( ( abs( lat_0+3-lat )**2 + abs( lon_0  -lon  )**2 )**0.5 ) < radius_around_station_degree:
                self.needed_tiles["north_west"] = self.north_west_tile

            assert len(self.needed_tiles) <= 4
            for k, t in self.needed_tiles.items():
                self.needed_tiles[k] = {}
                self.needed_tiles[k]["tile"] = t
                self.needed_tiles[k]["path"] = t.tiff_path
                mask = calculate_mask_for_tiff(t) 
                self.needed_tiles[k]["mask"] = mask
    



#def calculate_mask_for_tiff(tiff_tile):
#    lat = tiff_tile.lat
#    lon = tiff_tile.lon
#    lat_llcorner = tiff_tile.corners['lower_left']['lat']
#    lon_llcorner = tiff_tile.corners['lower_left']['lon']
#
#    # Calculate how far along by lat-lon the actual 
#    lat_diff = lat-lat_llcorner
#    lon_diff = lon-lon_llcorner
#    # Now since they are in 3 degree slices, we need to figure out how many pixels in it should be given the 36000 pixels 
#    lat_pix_offset = round((lat_diff/3)*36000)
#    lon_pix_offset = round((lon_diff/3)*36000)
#    edge_boundary = int(RADIUS_AROUND_STATION/10) # each pixel is 10m
#
#    # Slice out an upper bound of the region of interest to speed up calculations
#    tiff_array = np.zeros((36000,36000), dtype=int)
#
#    if lat_pix_offset > 0:
#        roi_lat_high    = 36000
#    else:
#        roi_lat_high    = edge_boundary
#
#    if lon_pix_offset > 0:
#        roi_lon_high    = 36000
#    else:
#        roi_lon_high    = edge_boundary
#
#    roi = tiff_array[
#        0:roi_lat_high,
#        0:roi_lon_high,
#    ]
#    # Each pixel is 10m, mask out the radius outside the landuse radius boundary
#    breakpoint()
#    lat_mesh_range = np.arange(-lat_pix_offset, -lat_pix_offset+roi_lat_high)
#    lon_mesh_range = np.arange(-lon_pix_offset, -lon_pix_offset+roi_lon_high)
#    y, x = np.meshgrid(
#        lat_mesh_range, 
#        lon_mesh_range
#    )
#    # Calculate the distance of each point from the center using the Pythagorean theorem
#    distance = np.sqrt( (y)**2 + (x)**2 )
#    # Create a mask to identify points within the desired circular radius
#    mask = distance <= edge_boundary
#    desired_shape = (36000, 36000)  # For example, pad the tensor to have 4 rows and 5 columns
#    # Calculate the amount of padding required for each dimension
#    pad_width = [(0, max(desired_shape[0] - mask.shape[0], 0)),  # Padding for rows
#                 (0, max(desired_shape[1] - mask.shape[1], 0))]  # Padding for columns
#    # Perform the padding
#    mask = np.pad(mask, pad_width, mode='constant', constant_values=False)
#    return mask



def calculate_mask_for_tiff(tiff_tile):
    lat = tiff_tile.lat
    lon = tiff_tile.lon
    lat_llcorner = tiff_tile.corners['lower_left']['lat']
    lon_llcorner = tiff_tile.corners['lower_left']['lon']

    # Calculate how far along by lat-lon the actual 
    lat_diff = lat-lat_llcorner
    lon_diff = lon-lon_llcorner
    # Now since they are in 3 degree slices, we need to figure out how many pixels in it should be given the 36000 pixels 
    lat_pix_offset = round((lat_diff/3)*36000)
    lon_pix_offset = round((lon_diff/3)*36000)
    edge_boundary = int(RADIUS_AROUND_STATION/10) # each pixel is 10m

    # Each pixel is 10m, mask out the radius outside the landuse radius boundary
    lat_mesh_range = np.arange(-edge_boundary, edge_boundary)
    lon_mesh_range = np.arange(-edge_boundary, edge_boundary)
    y, x = np.meshgrid(
        lat_mesh_range, 
        lon_mesh_range
    )
    # Calculate the distance of each point from the center using the Pythagorean theorem
    distance = np.sqrt( (y)**2 + (x)**2 )
    # Create a mask to identify points within the desired circular radius
    mask = distance <= edge_boundary
    pad_north = 36000-lat_pix_offset-edge_boundary
    pad_south = lat_pix_offset-edge_boundary
    pad_east  = 36000-lon_pix_offset-edge_boundary
    pad_west  = lon_pix_offset-edge_boundary
    if pad_north <= 0:
        mask = torch.from_numpy(mask)
        padded_mask = mask[0:pad_north]
        padded_mask = padded_mask.numpy()
    else:
        to_pad = ((0        , pad_north ),(0        , 0     ))
        padded_mask = np.pad(mask, to_pad, mode="constant", constant_values=False)
    if pad_south <= 0:
        padded_mask = torch.from_numpy(padded_mask)
        padded_mask = padded_mask[abs(pad_south):]
        padded_mask = padded_mask.numpy()
    else:
        to_pad = ((pad_south, 0         ),(0        , 0     ))
        padded_mask = np.pad(padded_mask, to_pad, mode="constant", constant_values=False)

    if pad_east <= 0:
        padded_mask = torch.from_numpy(padded_mask)
        padded_mask = padded_mask[:,0:pad_east]
        padded_mask = padded_mask.numpy()
    else:
        to_pad = ((0        , 0         ),(0        , pad_east))
        padded_mask = np.pad(padded_mask, to_pad, mode="constant", constant_values=False)

    if pad_west <= 0:
        padded_mask = torch.from_numpy(padded_mask)
        padded_mask = padded_mask[:,abs(pad_west):]
        padded_mask = padded_mask.numpy()
    else:
        to_pad = ((0        , 0         ),(pad_west , 0     ))
        padded_mask = np.pad(padded_mask, to_pad, mode="constant", constant_values=False)
    return padded_mask

        

def get_all_landuse_tiffs_needed(lat, lon):
    """
    The naming convention of the map tiffs have the tile from the lower-left corner
    e.g. S48E036 for the tile covering the area from 36E to 39E and 48S to 45S.
    """
    # Calculate the lower left and upper right co-ordinates of the tile the station is in
    landuse_tiffs = os.listdir(LANDUSE_PATH)
    lat_0, lon_0 = int((lat//3)*3), int((lon//3)*3) # Round each number DOWN to its nearest multiple of three
    station_tile = LanduseTiffTile(lat, lon, lat_0, lon_0, calc_neighbour_tiles=True) 
    all_needed_tiles = station_tile.needed_tiles
    return all_needed_tiles



if __name__ == "__main__":
    stations = LOAD_WAQI_STATIONS()
    print("Figure out from co-ords which tiff to load")
    for r_idx, row in enumerate(tqdm(stations, total=len(stations))):
        try:
            lat = row["g"][0]
            lon = row["g"][1]
            needed_tiles = get_all_landuse_tiffs_needed(lat, lon)
            total_landuse = [0 for _ in range(11)]
            for k, v in needed_tiles.items():
                tiff_path = v["path"]
                assert os.path.exists(tiff_path)
                tiff_image = Image.open(tiff_path)
                image_data = np.array(tiff_image)
                tiff_image.close()
                image_data = image_data/10  # values are 10, 20, 30, ... now map them to 1, 2, 3
                image_data = image_data.astype(int)
                #print("img_data", np.bincount(image_data.flatten()))
                assert image_data.shape == (36000, 36000)
                lat_first, lat_last = np.where(np.any(v["mask"], axis=1))[0][0], np.where(np.any(v["mask"], axis=1))[0][-1]
                lon_first, lon_last = np.where(np.any(v["mask"], axis=0))[0][0], np.where(np.any(v["mask"], axis=0))[0][-1]
                mask = v["mask"][lat_first:lat_last, lon_first:lon_last]
                image_data = image_data[lat_first:lat_last, lon_first:lon_last]
                result_array = image_data * mask
                landuse_bincounts = np.bincount(result_array.flatten())
                to_save = landuse_bincounts #np.pad(landuse_bincounts, (0, 11-landuse_bincounts.shape[0]))
                to_save = to_save.tolist()
                for lu_idx, lu in enumerate(to_save):
                    total_landuse[lu_idx] += lu
            print("landuse", sum(total_landuse[1:]))
        except:
            total_landuse = []
        save_path = os.path.join(LANDUSE_FEATURES_PROCESSED_PATH, f"{row['x']}.json")
        with open(save_path, "w") as f:
            json.dump(total_landuse, f)

    # https://worldcover2021.esa.int/data/docs/WorldCover_PUM_V2.0.pdf
    columns = [
        "station_id",
        "tree_cover",               # 10
        "shrubland",                # 20
        "grassland",                # 30
        "cropland",                 # 40
        "built_up",                 # 50
        "bare_sparse_vegetation",   # 60
        "snow_and_ice",             # 70 
        "permanent_water_bodies",   # 80
        "herbaceous_wetland",       # 90
        "mangroves",                # 95
        "moss_and_lichen"           # 100
    ]
    all_landuse_data = []
    for jfile in tqdm(os.listdir(LANDUSE_FEATURES_PROCESSED_PATH), total=len(os.listdir(LANDUSE_FEATURES_PROCESSED_PATH))):
        with open(os.path.join(LANDUSE_FEATURES_PROCESSED_PATH, jfile), 'r') as f:
            ld = json.load(f)
        uid = jfile.split(".json")[0]
        row = [uid]+ld
        all_landuse_data.append(row)
    assert len(all_landuse_data) == 14911
    landuse_df = pd.DataFrame(all_landuse_data, columns=columns)
    landuse_df.to_csv(LANDUSE_FEATURES_PATH)
