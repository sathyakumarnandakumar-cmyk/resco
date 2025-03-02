#!/usr/bin/env bash
mkdir -p profiling
filename=$(date +"%Y_%m_%d_%H%M%S")
python -m scalene --html --outfile profiling/"$filename".html "$*"
