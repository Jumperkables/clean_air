# Plt a series of lat-lon co-ordinates on a high resolution map of the world, rendered as a scalable PDF file.
# Usage: python plot_latlon_on_map.py
# Input: latlon.txt
# Output: latlon.pdf

import matplotlib.pyplot as plt
import os, sys
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from mpl_toolkits.basemap import Basemap

# Read the lat-lon co-ordinates from the input file
data = h5py.File(os.path.join(os.path.dirname(__file__), "../all_data.h5"), "r")
lats = [ data[f"stations/{suid}/geolocation/"][:][0] for suid in data['stations'].keys() ]
lons = [ data[f"stations/{suid}/geolocation/"][:][1] for suid in data['stations'].keys() ]
# create a DataFrame from the lat-lon co-ordinates
df = pd.DataFrame({'lat': lats, 'lon': lons})
# Create a GeoDataFrame from the lat-lon co-ordinates
geometry = [Point(xy) for xy in zip(df.lon, df.lat)]




# Step 1: Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf_points = gpd.GeoDataFrame(geometry=geometry)

# Step 4: Plot the map and the points using cartopy
fig = plt.figure(figsize=(20, 20))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()

# Add features for better visualization
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
#ax.add_feature(cfeature.LAKES, alpha=0.5)
#ax.add_feature(cfeature.RIVERS)

# Plot the world map
world.plot(ax=ax, edgecolor='black', facecolor='none')#, transform=ccrs.PlateCarree())

# Plot the points
gdf_points.plot(ax=ax, color='red', markersize=5)#, transform=ccrs.PlateCarree())

plt.show()

## Step 4: Plot the map and the points
#fig, ax = plt.subplots(figsize=(20, 20))
#world.plot(ax=ax, color='lightgrey')  # Plot the world map
#gdf_points.plot(ax=ax, color='red', markersize=5)  # Plot the points
#
#plt.show()

## Create a high resolution map of the world
#fig, ax = plt.subplots(figsize=(20, 20))
#m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180, resolution='i')
#m.drawcoastlines()
#m.drawcountries()
#m.drawmapboundary(fill_color='aqua')
#m.fillcontinents(color='lightgreen', lake_color='aqua')
#m.drawparallels(np.arange(-90, 90, 30))
#m.drawmeridians(np.arange(-180, 180, 60))
#
## Plot the lat-lon co-ordinates on the map
#gdf = gpd.GeoDataFrame(df, geometry=geometry)
#gdf.plot(ax=ax, marker='o', color='red', markersize=50)
#
## Save the map as a scalable PDF file
#plt.savefig('latlon.pdf', format='pdf', dpi=300)
#plt.show()
#
## End of script
