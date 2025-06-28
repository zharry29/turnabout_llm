#!/usr/bin/env bash

set -e

echo "Creating csv..."
python create_csv.py

echo "Visualizing..."
# python visualize.py

echo "Done!"