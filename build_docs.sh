#!/bin/bash

rm -rf docs/build
rm -rf docs/source/classes
rm -rf docs/source/utils 

sphinx-apidoc -o docs/source/classes deep_data_profiler/classes
sphinx-apidoc -o docs/source/utils deep_data_profiler/utils
sphinx-build -b html docs/source docs/build