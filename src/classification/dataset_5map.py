# PyTorch dataset object for the 5MAP dataset
# Standard imports
import argparse
from collections import OrderedDict
import os

# Third party imports
import h5py
from loguru import logger
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



class Dataset5MAP(Dataset):
    def __init__(self, label_modality: str, pollutant: str, require_other_modalities: bool, weather: bool, elevation: bool, landuse: bool, osm: bool, normalise: bool):
        """
        Initialise the dataset object.
            label_modality: The modality to use as the label
            pollutant:      Name of the pollutant to get data for
            weather:   Whether to include weather dataset
            elevation: Whether to include elevation Dataset
            landuse:   Whether to include landuse Dataset
            osm:       How to include the OSM data
        """
        self.label_modality = label_modality
        self.pollutant = pollutant
        self.weather = weather
        self.elevation = elevation
        self.landuse = landuse
        self.osm = osm
        self.data = h5py.File(os.path.join(os.path.dirname(__file__), "../../all_data.h5"), "r")
        self.relevant_spacetimestamps = self.get_relevant_spacetimestamps(pollutant='co', flatten_dictionary=True, require_other_modalities=require_other_modalities)
        logger.info(f"Number of unique spacetimes for this dataum: {len(self.relevant_spacetimestamps)}")
        # Will be overwritten in __getitem__
        self.ordered_features = OrderedDict({})
        self.ordered_label_features = OrderedDict({})
        self.num_features = 0
        self.num_label_features = 0
        self.normalisation_values = None
        if normalise:
            self.normalisation_values = {
                'pollutant': {
                    'mean': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/pollutant_data_mean.pt")).numpy(),
                    'std': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/pollutant_data_std.pt")).numpy(),
                },
                'weather': {
                    'mean': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/weather_data_mean.pt")).numpy(),
                    'std': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/weather_data_std.pt")).numpy(),
                },
                'elevation': {
                    'mean': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/elevation_data_mean.pt")).numpy(),
                    'std': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/elevation_data_std.pt")).numpy(),
                },
                'landuse': {
                    'mean': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/landuse_data_mean.pt")).numpy(),
                    'std': torch.load(os.path.join(os.path.dirname(__file__), "normalisation/landuse_data_std.pt")).numpy(),
                },
            }
    def _initialise_weather_timestamp_mask(self, station: str) -> np.ndarray:
        """
        Initialise a mask for the weather data, all False
        """
        total_timestamps = len(self.data['stations'][station]['era5_weather']['f_10m_u_component_of_wind'])
        return np.zeros(total_timestamps, dtype=bool).reshape(-1, 24)

    # type hint for a numpy array with integers
    def _check_other_modalities_valid_station(self, station: str, valid_timestamps: np.ndarray) -> list:
        """
        Check if this datum has valid data for all modalities for a single station (excluding OSM data).
        """
        # Elevation
        if np.isnan(self.data['stations'][station]['elevation'][0]):
            return []
        # Landuse 
        if any(np.isnan(self.data['stations'][station]['landuse'][0])):
            return []
        # Weather
        valid_weather_days = []
        for key in self.data['stations'][station]['era5_weather'].keys():
            weather_bool_mask = self._initialise_weather_timestamp_mask(station)
            weather_bool_mask[valid_timestamps, :] = True
            weather = np.array(self.data['stations'][station]['era5_weather'][key]).reshape(-1,24)
            valid_days = np.all(~np.isnan(weather) & weather_bool_mask, axis=1)
            valid_weather_days.append(set(np.where(valid_days)[0]))
        return list(set.intersection(*valid_weather_days))

    def _check_other_modalities_valid_spacetime(self, station: str, timestep: int) -> bool:
        """
        Check if this datum has valid data for all modalities for a single spacetime (excluding OSM data).
        """
        logger.warning("This function is slow and should be used sparingly.")
        # Elevation
        if np.isnan(self.data['stations'][station]['elevation'][0]):
            return False
        # Landuse 
        if any(np.isnan(self.data['stations'][station]['landuse'][0])):
            return False
        # Weather
        for key in self.data['stations'][station]['era5_weather'].keys():
            if any(np.isnan(self.data['stations'][station]['era5_weather'][key][24*timestep:24*(timestep+1)])):
                return False
        return True

    def _check_spacetime_validity(self, station: str, timestep: int, pollutant: str) -> bool:
        """
        Check if the data at a given station and timestep is valid.
        """
        logger.warning("This function is slow and should be used sparingly. For gathering all timestamps for a station, use _get_valid_timestamps_for_station.")
        data = self.data['stations'][station]['waqi_readings'][pollutant]
        for reading_var in data:
            if np.isnan(data[reading_var][timestep]):
                return False
        return True

    def _get_valid_timestamps_for_station(self, station: str, pollutant: str) -> np.ndarray:
        """
        Get all valid timestamps for a given station.
        """
        data = self.data['stations'][station]['waqi_readings'][pollutant]
        reading_vars = np.stack([data[reading_var] for reading_var in data.keys() if reading_var != 'mean'])
        indices = np.where((~np.isnan(reading_vars)).all(axis=0))[0]
        assert (np.where((~np.isnan(reading_vars)).all(axis=0))[0] == np.where((~np.isnan(reading_vars)).any(axis=0))[0]).all()
        return indices

    def get_input_output_features(self):
        if self.num_features == 0 or self.num_label_features == 0:
            raise ValueError("Features have not been initialised yet. Run __getitem__ to initialise them.")
        return self.ordered_features, self.num_features, self.ordered_label_features, self.num_label_features

    def get_relevant_spacetimestamps(self, pollutant: str, flatten_dictionary: bool, require_other_modalities: bool) -> type[dict | list]:
        """
        Get all relevant spacetime data for a given pollutant.
            pollutant: 
                str - The pollutant to get data for
            flatten_dictionary: 
                bool - Whether to flatten the dictionary into a list
            require_other_modalities: 
                bool - Whether the returned stations should be filtered to only those that have data for all modalities (excluding OSM data)
        """
        valid_spacetime_data = {}   # Each station - timestep pair that has valid data
        for station in tqdm(self.data['stations'], desc='Retrieving relevant spacetime entries', total=len(self.data['stations'])):
            valid_timestamps = self._get_valid_timestamps_for_station(station, pollutant)
            if len(valid_timestamps) != 0:
                if require_other_modalities:
                    valid_timestamps = self._check_other_modalities_valid_station(station, valid_timestamps)
                valid_spacetime_data[station] = valid_timestamps
        if flatten_dictionary:
            valid_spacetime_data = [ (station, timestep) for station in valid_spacetime_data for timestep in valid_spacetime_data[station] ]
        return valid_spacetime_data

    def split_dataset(self, train_ratio: float, valid_ratio: float, test_ratio: float, seed: int):
        """
        Split the dataset into train, valid, and test sets.
            train_ratio: 
                float - The ratio of the dataset to use for training
            valid_ratio: 
                float - The ratio of the dataset to use for validation
            test_ratio: 
                float - The ratio of the dataset to use for testing
            seed:
                int - The seed for the random number generator
        """
        assert train_ratio + valid_ratio + test_ratio == 1
        # Spacetime tuples are of the form (station, timestamp)
        # make sure that no station is in both train and test
        np.random.seed(seed)
        stations = np.array([station for station, _ in self.relevant_spacetimestamps])
        unique_stations = np.unique(stations)
        np.random.shuffle(unique_stations)
        train_stations = unique_stations[:int(train_ratio*len(unique_stations))]
        valid_stations = unique_stations[int(train_ratio*len(unique_stations)):int((train_ratio+valid_ratio)*len(unique_stations))]
        test_stations = unique_stations[int((train_ratio+valid_ratio)*len(unique_stations)):]
        train_indices = [idx for idx, (station, _) in enumerate(self.relevant_spacetimestamps) if station in train_stations]
        valid_indices = [idx for idx, (station, _) in enumerate(self.relevant_spacetimestamps) if station in valid_stations]
        test_indices = [idx for idx, (station, _) in enumerate(self.relevant_spacetimestamps) if station in test_stations]
        assert len(set(train_stations).intersection(set(valid_stations))) == 0
        assert len(set(train_stations).intersection(set(test_stations))) == 0
        assert len(set(valid_stations).intersection(set(test_stations))) == 0
        train_dataset = torch.utils.data.Subset(self, train_indices)
        valid_dataset = torch.utils.data.Subset(self, valid_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)
        return train_dataset, valid_dataset, test_dataset

    def __len__(self) -> int:
        return len(self.relevant_spacetimestamps)

    def __getitem__(self, idx):
        station, timestamp = self.relevant_spacetimestamps[idx]
        station = self.data['stations'][station]
        # Pollutant data
        pollutant_data = station['waqi_readings'][self.pollutant]
        pollutant_data = {
            'count': pollutant_data['count'][timestamp],
            'max': pollutant_data['max'][timestamp],
            'median': pollutant_data['median'][timestamp],
            'min': pollutant_data['min'][timestamp],
            'q1': pollutant_data['q1'][timestamp],
            'q3': pollutant_data['q3'][timestamp],
            'stdev': pollutant_data['stdev'][timestamp],
        }
        # Weather data
        weather_data = {}
        if self.weather:
            f_10m_u_component_of_wind = station['era5_weather']['f_10m_u_component_of_wind'][(24*timestamp):(24*(timestamp+1))]
            ## calculate the count, max, median, min, q1, q3, stdev for f_10m_u_component_of_wind
            #f_10m_u_count = np.count_nonzero(~np.isnan(f_10m_u_component_of_wind))
            #f_10m_u_max = np.nanmax(f_10m_u_component_of_wind)
            #f_10m_u_median = np.nanmedian(f_10m_u_component_of_wind)
            #f_10m_u_min = np.nanmin(f_10m_u_component_of_wind)
            #f_10m_u_q1 = np.nanpercentile(f_10m_u_component_of_wind, 25)
            #f_10m_u_q3 = np.nanpercentile(f_10m_u_component_of_wind, 75)
            #f_10m_u_stdev = np.nanstd(f_10m_u_component_of_wind)
            for i in range(len(f_10m_u_component_of_wind)):
                weather_data[f'f_10m_u_component_of_wind_{i}'] = f_10m_u_component_of_wind[i]
            f_10m_v_component_of_wind = station['era5_weather']['f_10m_v_component_of_wind'][(24*timestamp):(24*(timestamp+1))]
            for i in range(len(f_10m_v_component_of_wind)):
                weather_data[f'f_10m_v_component_of_wind_{i}'] = f_10m_v_component_of_wind[i]
            f_skin_temperature = station['era5_weather']['f_skin_temperature'][(24*timestamp):(24*(timestamp+1))]
            for i in range(len(f_skin_temperature)):
                weather_data[f'f_skin_temperature_{i}'] = f_skin_temperature[i]
            f_surface_pressure = station['era5_weather']['f_surface_pressure'][(24*timestamp):(24*(timestamp+1))]
            for i in range(len(f_surface_pressure)):
                weather_data[f'f_surface_pressure_{i}'] = f_surface_pressure[i]
        # Elevation data
        elevation_data = {}
        if self.elevation:
            elevation_data = {
                'elevation': np.array([station['elevation'][0]]),
            }
        # Landuse data
        landuse_data = {}
        if self.landuse:
            lu_data = station['landuse'][0]
            landuse_data['tree_cover'] = lu_data[0]
            landuse_data['shrubland'] = lu_data[1]
            landuse_data['grassland'] = lu_data[2]
            landuse_data['cropland'] = lu_data[3]
            landuse_data['built_up'] = lu_data[4]
            landuse_data['bare_sparse_vegetation'] = lu_data[5]
            landuse_data['snow_and_ice'] = lu_data[6]
            landuse_data['permanent_water_bodies'] = lu_data[7]
            landuse_data['herbaceous_wetland'] = lu_data[8]
            landuse_data['mangroves'] = lu_data[9]
            landuse_data['moss_and_lichen'] = lu_data[10]
        # OSM data
        osm_data = {}
        if self.osm:
            raise NotImplementedError("Deal with OSM data later")
        # Save the order of features for later use
        if len(self.ordered_features) == 0:
            self.ordered_features['pollutant'] = list(pollutant_data.keys())
            self.ordered_features['weather'] = list(weather_data.keys())
            self.ordered_features['elevation'] = list(elevation_data.keys())
            self.ordered_features['landuse'] = list(landuse_data.keys())
            self.ordered_features['osm'] = list(osm_data.keys())
            self.ordered_label_features[self.label_modality] = self.ordered_features[self.label_modality]
            del self.ordered_features[self.label_modality]
            self.num_features = sum([len(self.ordered_features[key]) for key in self.ordered_features])
            self.num_label_features = len(self.ordered_label_features[self.label_modality])
        return_data = []
        return_labels = []
        return_pollutant = np.array(list(pollutant_data.values()))
        return_weather = np.array(list(weather_data.values()))
        return_elevation = elevation_data['elevation']
        return_landuse = np.array(list(landuse_data.values()))
        return_osm = np.array(list(osm_data.values()))
        if self.normalisation_values is not None:
            if self.pollutant:
                return_pollutant = (return_pollutant - self.normalisation_values['pollutant']['mean']) / self.normalisation_values['pollutant']['std']
            if self.weather:
                return_weather = (return_weather - self.normalisation_values['weather']['mean']) / self.normalisation_values['weather']['std']
            if self.elevation:
                return_elevation = (return_elevation - self.normalisation_values['elevation']['mean']) / self.normalisation_values['elevation']['std']
            if self.landuse:
                return_landuse = (return_landuse - self.normalisation_values['landuse']['mean']) / self.normalisation_values['landuse']['std']
        # Pollutant
        if self.label_modality == 'pollutant':
            return_labels.append(return_pollutant)
        else:
            return_data.append(return_pollutant)
        # Weather
        if self.label_modality == 'weather':
            return_labels.append(return_weather)
        else:
            return_data.append(return_weather)
        # Elevation
        if self.label_modality == 'elevation':
            return_labels.append(return_elevation)
        else:
            return_data.append(return_elevation)
        # Landuse
        if self.label_modality == 'landuse':
            return_labels.append(return_landuse)
        else:
            return_data.append(return_landuse)
        # OSM
        if self.label_modality == 'osm':
            return_labels.append(return_osm)
        else:
            return_data.append(return_osm)
        breakpoint()
        return_data = torch.tensor(np.concatenate(return_data))
        return_labels = torch.tensor(np.concatenate(return_labels))
        return return_data, return_labels



if __name__ == "__main__":
    # Argument parsing
    args = argparse.ArgumentParser()
    dset_args = args.add_argument_group("Dataset arguments")
    dset_args.add_argument("--pollutant", type=str, choices=['co', 'no2', 'o3', 'pm1', 'pm10', 'pm25', 'so2'])
    dset_args.add_argument("--require_other_modalities", action='store_true')
    dset_args.add_argument("--weather", action='store_true')
    dset_args.add_argument("--elevation", action='store_true')
    dset_args.add_argument("--landuse", action='store_true')
    dset_args.add_argument("--osm", action='store_true')
    dset_args.add_argument("--label_modality", type=str, choices=['pollutant', 'weather', 'elevation', 'landuse', 'osm'])
    dset_args.add_argument("--normalise", action='store_true')
    args = args.parse_args()
    logger.info(args)
    if not args.require_other_modalities:
        assert not args.weather
        assert not args.elevation
        assert not args.landuse
        assert not args.osm
    assert getattr(args, args.label_modality), f"""The label modality {args.label_modality} must be included in the dataset"""

    # Create the dataset
    dataset = Dataset5MAP(
        label_modality=args.label_modality,
        pollutant=args.pollutant,
        require_other_modalities=args.require_other_modalities,
        weather=args.weather,
        elevation=args.elevation,
        landuse=args.landuse,
        osm=args.osm,
        normalise=args.normalise,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    # For each index of both labels and data, store each example in order to calculate the mean and std
    running_data = []
    running_labels = []
    for data_idx, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        data, labels = data
        running_data.append(data)
        running_labels.append(labels)
    running_data = torch.cat(running_data)
    running_labels = torch.cat(running_labels)

    # Only run this is pollutant is label modality and everything else except OSM is included
    if args.label_modality == 'pollutant' and args.weather and args.elevation and args.landuse:
        need_to_save_normalisation = False
        pol_mean_path = os.path.join(os.path.dirname(__file__), "normalisation/pollutant_data_mean.pt")
        pol_std_path = os.path.join(os.path.dirname(__file__), "normalisation/pollutant_data_std.pt")
        wea_mean_path = os.path.join(os.path.dirname(__file__), "normalisation/weather_data_mean.pt")
        wea_std_path = os.path.join(os.path.dirname(__file__), "normalisation/weather_data_std.pt")
        ele_mean_path = os.path.join(os.path.dirname(__file__), "normalisation/elevation_data_mean.pt")
        ele_std_path = os.path.join(os.path.dirname(__file__), "normalisation/elevation_data_std.pt")
        lan_mean_path = os.path.join(os.path.dirname(__file__), "normalisation/landuse_data_mean.pt")
        lan_std_path = os.path.join(os.path.dirname(__file__), "normalisation/landuse_data_std.pt")
        if any(
            not os.path.exists(path) for path in [pol_mean_path, pol_std_path, wea_mean_path, wea_std_path, ele_mean_path, ele_std_path, lan_mean_path, lan_std_path]
        ):
            need_to_save_normalisation = True
            
        if need_to_save_normalisation:
            # Pollutant data
            pollutant_data = running_labels
            pollutant_data_mean = pollutant_data.mean(dim=0)
            pollutant_data_std = pollutant_data.std(dim=0)
    
            # Weather data
            weather_data = running_data[:, 0:96]
            weather_data_mean = weather_data.mean(dim=0)
            weather_data_std = weather_data.std(dim=0)
    
            # Elevation data
            elevation_data = running_data[:, 96:97]
            elevation_data_mean = elevation_data.mean(dim=0)
            elevation_data_std = elevation_data.std(dim=0)
    
            # Landuse data
            landuse_data = running_data[:, 97:]
            landuse_data_mean = landuse_data.mean(dim=0)
            landuse_data_std = landuse_data.std(dim=0)
    
            # Save these tensors for later loading in normalisation
            torch.save(pollutant_data_mean, pol_mean_path)
            torch.save(pollutant_data_std, pol_std_path)
            torch.save(weather_data_mean, wea_mean_path)
            torch.save(weather_data_std, wea_std_path)
            torch.save(elevation_data_mean, ele_mean_path)
            torch.save(elevation_data_std, ele_std_path)
            torch.save(landuse_data_mean, lan_mean_path)
            torch.save(landuse_data_std, lan_std_path)
    
