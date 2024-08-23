import os, sys
from tqdm import tqdm
import pandas as pd

from utils import *

aosm = LOAD_OSM_FEATURES()

aosm = [osm["osm_geoms"].drop(columns=set(osm["osm_geoms"].columns)-set(OSM_GREENLIST_COLUMNS)) for osm in aosm if osm['osm_geoms'] is not None]
df = pd.concat(aosm, ignore_index=True)
breakpoint()
osm_tags = [k for k in OSM_TAGS.keys()] + [v for v in OSM_TAGS.values()]
# Initialize an empty list to store the counts
non_nan_counts = []

# Iterate through each column and count non-NaN values
for column in df.columns:
    non_nan_count = df[column].count()  # Count non-NaN values in the column
    non_nan_counts.append((column, non_nan_count))  # Append column name and count to the list

# Sort the list of counts by the count value
non_nan_counts.sort(key=lambda x: x[1], reverse=True)

# Display the counts
for column, count in non_nan_counts:
    print(f"Column '{column}': {count} non-NaN values")

##################
# The only shared column is "geometry"

print(len(df))
breakpoint()
for osm in tqdm(aosm):
    geoms = osm['osm_geoms']
