#!/usr/bin/env bash
mkdir -p profiling
filename=$(date +"%Y_%m_%d_%H%M%S")
python -m cProfile -s cumulative -o profiling/"$filename".stats "$*"
