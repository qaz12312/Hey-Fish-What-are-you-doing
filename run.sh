#!/bin/bash

# cd pwd
cd `dirname $0`

echo "Loading all croods files....."
ls *.csv
cp /*.csv testCSV
echo "**********************************************************"
echo "Convert data....."
/usr/bin/python3.6 convertData.py
echo "**********************************************************"
echo "Training data....."
/usr/bin/python3.6 training.py
echo "**********************************************************"

exit 0