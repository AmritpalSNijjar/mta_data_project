#!/bin/bash

python preprocess_hourly_data.py -f "$1"

read user_input

if [[ "$user_input" = "EXIT" ]]; then
	echo "Exiting."
	rm df_cache.csv
fi
