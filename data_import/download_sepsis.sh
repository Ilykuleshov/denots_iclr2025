#!/bin/bash

mkdir /mnt/data/raw/sepsis/
cd /mnt/data/raw/sepsis/

# Kaggle is much faster than the per-file wget
kaggle datasets download salikhussaini49/prediction-of-sepsis -p .
unzip -q ./prediction-of-sepsis.zip