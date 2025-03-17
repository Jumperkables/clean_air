# 5MAP: 5 Modalities Aligned for Air Pollution
Augmenting air pollution readings with 4 other modalities:
![comp_table](figs/5map_vs_others.png)

## QR Code For This Repo
![qr](figs/qr.png)

## **1. Air Pollution:** 
- [WAQI](https://waqi.info/)
    * `PM-25`
    * `PM-10`
    * `PM-1`
    * `o3`
    * `no2`
    * `so2`
    * `co`
- Daily, Over 3+ years
- ~15,000 sites worldwide
![map](figs/world_map.png)
#### My Thoughts
- Pollution and healthcare have been paired passions of mine since I was 11 and became 'climate aware' 
- Web scraping practice
- [`src/data_processing/get_aqicn_readings.py`](src/data_processing/get_aqicn_readings.py)
 

## **2. Weather:**
- ERA5 re-analysis from [copernius](https://pypi.org/project/cdsapi/)
    * `100m_u_component_of_wind`
    * `100m_v_component_of_wind`
    * `10m_u_component_of_wind`
    * `10m_v_component_of_wind`
    * `2m_temperature`
    * `skin_temperature`
    * `surface_pressure`
    * `total_precipitation`
- At each EXACT lat-lon geo-cords for the above ~15,000 sites
- Hourly
####  My Thoughts
- Practiced efficiency tricks for processing geogrphical data (longitudinal `.grib`)
    - **Efficient** to access data down `time` dimension
    - **Hard** to access data across the spatial dimension (lat-lons for each 15,000 site)
    - `import xarray`
    - `xr.set_options(parallel=True)` 
    - `xr.open_dataset(engine="cfgrib")`
    - Efficient `np` indexing
    - 200 days `->` 4 days running
- [`src/data_processing/era_*.py`](src/data_processing/era5_process_final.py)


## **3. Landuse:** 
- [Worldcover 2021 ESA](https://worldcover2021.esa.int/download)
    * `tree_cover`
    * `shrubland`
    * `grassland`
    * `cropland`
    * `built_up`
    * `bare_sparse_vegetation`
    * `snow_and_ice`
    * `permanent_water_bodies`
    * `herbaceous_wetland`
    * `mangroves`
    * `moss_and_lichen`
- Gathered statistics in a 15km radius around each site (from tiff)
![landuse_tab](figs/landuse_tab.png)
#### My Thoughts
- Challenging but satisfying method of collating these
    1. Large satellite images divide the world up in squares (`.tiff`)
    2. If a site is on the edge of one, the radius may overlap in up to **4** tiles at once
    3. Satisfying geometric reasoning required that was fun to code
- [`src/data_processing/landuse.py`](src/data_processing/landuse.py)


## **4. Elevation:**
- The height above sea level at each ~15,000 site
#### My Thoughts
- Pretty simple to be honest
- [`src/data_processing/elevation.py`](src/data_processing/elevation.py)


## **5. OpenStreetMap:**
- Semi-structured list of things related to air-pollution
```python
tags = {
    "power": True,
    "man_made": ["petroleum_well", "monitoring_station"],
    "landuse": ["industrial", "highway", "motorway", "quarry", "depot", "farm"],
}
```
- Return features are as follows
![osm_tab](figs/osm_table.png)
![all_tab](figs/all_tab.png)
#### My Thoughts
- Extremely rich semi-structured data
- Significantly more complicated than comparable publically available dataset
- Great to have practice with

# Final Thoughts
I have many other more ML heavy projects that were a close second
- End-to-end Multimodal Skin Cancer Project
    * NDA and non-compete covers code, paper, and project details for another year
    * Not allowed to speak about this one `:(`
- Neurolinguistic Word Norm LLM Finetuning
    * [Initial Project for my Thesis](https://github.com/Jumperkables/a_vs_c)
    * [LLM Finetuning Repo](https://github.com/Jumperkables/llm_wordnorms)
    * Satisfying getting LLM to run on 11GB GPU
    * Didn't push my skillset as much as 5MAP
- Finetuning a TTS Home Assistant
    * [Repo](https://github.com/Jumperkables/sigmarvis)
    * Successful, but not as big in scope
- Archaeology Artefact Detection
    * [Repo](https://github.com/Jumperkables/archaeology_classifier)
    * [Nature's Scientific Report paper](https://www.nature.com/articles/s41598-022-15965-2)
    * Didn't push my skillset as far as 5MAP
- Does Generative Next-Frame pretraining Instill Knowledge of Physics?
    * [Repo](https://github.com/Visual-modelling)
    * AMAZING to work on, fundamental ML research
    * Led this collaboration project
- Beyond Left and Right: News Topic Clustering Startup
    * Lots of LLM tricks to make work
    * Gave me decent backend experience (used `Django` and `GraphQL`)
    * Gave startup a go but it didn't work out (9/10 fail this is ok)
- And 2 more research projects from my PhD


# Next Steps
- I have basic classification results and framework set up
    * [`src/classification/`](src/classification/)
- Use this to put a 5MAP classifier on an embedded Device (Nvidia Jetson) as proof of concept
- Shopping List
| Component | Brand | Approx Price |
- | Jetson Board | NVIDIA Jetson Orin Nano (4GB) | £150 |
- | Storage | SanDisk Extreme MicroSD (32GB) | £15 |
- | Air Pollution Sensor | Plantower PMS5003 | £40 |
- | Weather Sensor | Bosch BME680 | £25 |
- | GPS Module | u-blox NEO-6M | £30 |
- | Internet Connectivity	Waveshare | 4G LTE (SIM7600G-H) | £70 |
- | WiFi Adapter (Optional) | TP-Link Archer T3U | £20 |
- | External GPS Antenna | Generic | £10 |
- | Power Supply | Anker PowerCore+ PD 26800mAh | £80 |
- | Enclosure | Generic ABS/Aluminum Box | £30 |
- Learning List
    * C++ [my progress here](https://github.com/Jumperkables/learning/tree/main/c%2B%2B)
    * CUDA
    * TensorRT