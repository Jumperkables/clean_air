#!/bin/bash
source ../venv/bin/activate
for i in {1..31}; do
    i=$(printf '%02d' "$i")
    fname="day-$i.json"
    if [ -f "/home/jumperkables/clean_air/data/ERA5Weather/$fname" ]
    then
        echo "$fname exists"
    else
        python era5_process_final.py -d "$i"
    fi
done
