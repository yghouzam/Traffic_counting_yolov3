#! /usr/bin/env python3
# coding: utf-8

import configparser
import os
import ast


class Config:
    def __init__(self):
        self.config = ConfigFile()

    def parse_counters(self):
        counters = []
        for counter in self.config['COUNTERS']:
            coords = ast.literal_eval(self.config['COUNTERS'][counter])
            counters.append((coords['x1'], coords['y1'], coords['x2'], coords['y2']))
        return counters

    def parse_czones(self):
        czones = []
        ids_seen = []
        for czone in self.config['CONTROL_ZONE']:
            czone_dict = {}
            czone_dict_p = ast.literal_eval(self.config['CONTROL_ZONE'][czone])

            czid = czone_dict_p['id']
            assert (czid not in ids_seen), 'Control zone id {} is not unique'.format(czid)
            czone_dict['id'] = czid
            ids_seen.append(czid)

            czone_dict['speed_limit'] = czone_dict_p['speed_limit']
            czone_dict['cz_distance'] = czone_dict_p['cz_distance']
            czone_dict['start'] = ((czone_dict_p['start']['x1'], czone_dict_p['start']['y1']),
                                   (czone_dict_p['start']['x2'], czone_dict_p['start']['y2']))
            czone_dict['exit'] = ((czone_dict_p['exit']['x3'], czone_dict_p['exit']['y3']),
                                  (czone_dict_p['exit']['x4'], czone_dict_p['exit']['y4']))
            czones.append(czone_dict)
        return czones


class ConfigFile:
    def __new__(self, ini_file="config.ini"):
        ini_dir = os.path.dirname(os.path.realpath(__file__))
        ini_path = os.path.join(ini_dir, ini_file)
        config = configparser.ConfigParser()

        if not os.path.exists(ini_path):
            raise IOError("File {} does not exist".format(ini_path))

        config.read(ini_path)
        return config
