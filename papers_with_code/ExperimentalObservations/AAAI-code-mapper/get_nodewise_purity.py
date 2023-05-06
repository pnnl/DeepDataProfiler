#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 05:51:15 2021

"""
import re
import os
import sys
import json
from glob import glob

if __name__ == '__main__':
    mapper_version = sys.argv[1]
    mapper_dir = f'../mapper_graphs/{mapper_version}/'
    purity_dir = f'../datasets/nodewise_purity/'
    mapper_filename = mapper_dir + sys.argv[2]
    mapper_layer = sys.argv[3]

    if not os.path.exists(purity_dir):
        os.makedirs(purity_dir)
    
    num_classes = 10
    num_class_pts = 5000
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    if os.path.exists(os.path.join(purity_dir, f'{mapper_version}.json')):
        with open(os.path.join(purity_dir, f'{mapper_version}.json')) as f:
            purity_dict = json.load(f)
    else:
        purity_dict = {}
        
    with open(mapper_filename) as f:
        mapper = json.load(f)
    nodes = mapper['nodes']
    links = mapper['edges']
            
    purity = 0
    for key in nodes:
        node = nodes[key]
        purity += 1 / len(nodes[key]['categorical_cols_summary']['label'].keys())
    purity = purity / len(list(nodes.keys()))
    purity_dict[mapper_layer] = purity
        
    with open(os.path.join(purity_dir, f'{mapper_version}.json'), 'w') as f:
        f.write(json.dumps(purity_dict, indent=4))