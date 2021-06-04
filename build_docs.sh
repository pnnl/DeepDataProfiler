#!/bin/bash

rm -rf docs/build
rm -rf docs/source/classes
rm -rf docs/source/algorithms 

sphinx-apidoc -o docs/source/classes deep_data_profiler/classes
sphinx-apidoc -o docs/source/algorithms deep_data_profiler/algorithms
sphinx-build -b html docs/source docs/build
