# standard imports
import argparse

# third party imports
import lightgbm as lgb
from loguru import logger
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import torch
from tqdm import tqdm

# local imports
import dataset_5map

def parse_args():
    args = argparse.ArgumentParser(description="5MAP running script")
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
    assert getattr(args, args.label_modality), f"Label modality {args.label_modality} is not present. Activate it to use as a label."
    return args


def main(args):
    # Create the dataset
    dataset = dataset_5map.Dataset5MAP(
        label_modality=args.label_modality,
        pollutant=args.pollutant,
        require_other_modalities=args.require_other_modalities,
        weather=args.weather,
        elevation=args.elevation,
        landuse=args.landuse,
        osm=args.osm,
        normalise=args.normalise,
    )
    dataset.__getitem__(0)  # Needed to initialise input and output features
    input_features, num_input_features, label_features, num_label_features = dataset.get_input_output_features()
    train_dataset, valid_dataset, test_dataset = dataset.split_dataset(train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, seed=42)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    # Train
    all_train_data = []
    all_train_labels = []
    logger.info("Compiling the train data into numpy array for shallow classifier")
    for data_idx, data in enumerate(tqdm(train_dataloader)):
        data, labels = data
        all_train_data.append(data)
        all_train_labels.append(labels)
    # Valid
    all_valid_data = []
    all_valid_labels = []
    logger.info("Compiling the valid data into numpy array for shallow classifier")
    for data_idx, data in enumerate(tqdm(valid_dataloader)):
        data, labels = data
        all_valid_data.append(data)
        all_valid_labels.append(labels)
    # Test
    all_test_data = []
    all_test_labels = []
    logger.info("Compiling the test data into numpy array for shallow classifier")
    for data_idx, data in enumerate(tqdm(test_dataloader)):
        data, labels = data
        all_test_data.append(data)
        all_test_labels.append(labels)
    all_train_data = torch.cat(all_train_data, dim=0).numpy()
    all_train_labels = torch.cat(all_train_labels, dim=0).numpy()
    all_valid_data = torch.cat(all_valid_data, dim=0).numpy()
    all_valid_labels = torch.cat(all_valid_labels, dim=0).numpy()
    all_test_data = torch.cat(all_test_data, dim=0).numpy()
    all_test_labels = torch.cat(all_test_labels, dim=0).numpy()

    all_train_data = np.concatenate([all_train_data, all_valid_data], axis=0)
    all_train_labels = np.concatenate([all_train_labels, all_valid_labels], axis=0)

    # Train the model
    base_model = lgb.LGBMRegressor()
    model = MultiOutputRegressor(base_model)
    model.fit(all_train_data, all_train_labels)
    test_pred = model.predict(all_test_data)
    mse = mean_squared_error(all_test_labels, test_pred)
    logger.success(f"Mean Squared Error: {mse}")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
