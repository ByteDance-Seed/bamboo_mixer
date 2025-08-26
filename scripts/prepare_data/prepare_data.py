# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, wait

import numpy as np

from formula_design.data import DatasetConfig, MonoDataset, MXDataset
from formula_design.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description="process data and save to pkl")
parser.add_argument("--conf", type=str, help='config yaml file')
parser.add_argument("--num_workers",
                    type=int,
                    default=1,
                    help='parallel processes')
parser.add_argument("--data_type",
                    type=str,
                    default="formula",
                    help='formula or mono')

args = parser.parse_args()

with open("./smiles_dict/smiles_dict.json", "r") as f:
    smiles_dict = json.load(f)


def normalize_entries(entries, properties, config):
    """
    Normalize the specified properties in the entries.

    :param entries: List of dictionaries containing the data
    :param properties: List of property keys in the entries to normalize
    :return: List of normalized entries
    """

    # Initialize min and max values for each property
    min_values = {prop: float('inf') for prop in properties}
    max_values = {prop: float('-inf') for prop in properties}

    # Find min and max values across all entries for each property

    for i, entry in enumerate(entries):
        for prop in properties:
            value = entry[prop]
            flag = True

            if prop + "_mask" in entry.keys():
                if entry[prop + "_mask"] == False:
                    flag = False

            if flag:
                if value < min_values[prop]:
                    min_values[prop] = value
                if value > max_values[prop]:
                    max_values[prop] = value

    for prop in properties:
        if "_T" in prop:
            min_values[prop] = 0

    for entry in entries:
        for prop in properties:
            min_val = min_values[prop]
            max_val = max_values[prop]
            if max_val - min_val == 0:
                entry[prop] = 0.5
            else:
                entry[prop] = (entry[prop] - min_val) / (max_val - min_val)

    data_to_save = {"min_values": min_values, "max_values": max_values}

    # Save the min and max values to a JSON file
    with open(os.path.join(config._config['save_dir'], "min_max_values.json"),
              "w") as json_file:
        json.dump(data_to_save, json_file, indent=4)

    return entries


def preprocessing_mono(config):

    with open(config._config['json_path'], "r") as f:
        data = json.load(f)

    for molecule in data:

        assert "name" in molecule
        if "smiles" in molecule:
            pass
        elif molecule["name"] in smiles_dict:
            molecule["smiles"] = smiles_dict[molecule["name"]]
        else:
            raise ValueError(f"Missing smiles for {molecule['name']}")

        assert "temperature" in molecule
        ## Normalization
        # The original unit of temperature is celsius
        molecule['temperature'] = (molecule['temperature'] + 273.15) / 373.15
        if not os.path.exists(config._config['save_dir']):
            os.makedirs(config._config['save_dir'])

        with open(
                os.path.join(config._config['save_dir'],
                             "processed_data.json"), "w") as f:
            json.dump(data, f, indent=4)

    return config


def preprocessing_formula(config):

    with open(config._config['json_path'], "r") as f:
        data = json.load(f)

    for formula in data:

        solv_ratios = 0

        for solv in formula["solvents"]:
            assert "name" in solv
            assert "molar_ratio" in solv
            if "smiles" in solv:
                pass
            elif solv["name"] in smiles_dict:
                solv["smiles"] = smiles_dict[solv["name"]]
            else:
                raise ValueError(f"Missing smiles for {solv['name']}")
            solv_ratios += solv["molar_ratio"]

        if solv_ratios != 1:
            for solv in formula["solvents"]:
                solv["molar_ratio"] /= solv_ratios

        salt_ratios = 0
        for salt in formula["salts"]:
            assert "name" in salt
            assert "molar_ratio" in salt
            if "smiles" in salt:
                pass
            elif salt["name"].replace("Li", "").replace("LI",
                                                        "") in smiles_dict:
                salt["smiles"] = smiles_dict[salt["name"].replace(
                    "Li", "").replace("LI", "")]
            else:
                raise ValueError(f"Missing smiles for {salt['name']}")
            salt_ratios += salt["molar_ratio"]

        if salt_ratios != 1:
            for salt in formula["salts"]:
                salt["molar_ratio"] /= salt_ratios

        assert formula["salt_molar_ratio"] < 1
        assert "temperature" in formula

        ## Normalization
        # The original unit of temperature is celsius
        formula['temperature'] = (formula['temperature'] + 273.15) / 373.15
        # The original unit of conductivity is mS/cm
        if "conductivity" in formula:
            formula['conductivity'] = (np.log(formula['conductivity']) +
                                       4.6002) / 8.3344
        if "conductivity_NE" in formula:
            formula['conductivity_NE'] = (np.log(formula['conductivity_NE']) +
                                          4.6002) / 8.3344
        if "conductivity_mistry" in formula:
            formula['conductivity_mistry'] = (
                np.log(formula['conductivity_mistry']) + 4.6002) / 8.3344

    if not os.path.exists(config._config['save_dir']):
        os.makedirs(config._config['save_dir'])

    with open(os.path.join(config._config['save_dir'], "processed_data.json"),
              "w") as f:
        json.dump(data, f, indent=4)

    return config


if __name__ == "__main__":

    config = DatasetConfig(args.conf)

    logger.info(str(config))

    if args.data_type == "formula":

        config = preprocessing_formula(config)

    elif args.data_type == "mono":

        config = preprocessing_mono(config)

    else:
        raise ValueError(f"Unknown data type {args.data_type}")

    logger.info('Finished preprocessing data')

    with ProcessPoolExecutor(args.num_workers) as pool:
        futs = []

        if args.data_type == "formula":
            futs.append(pool.submit(MXDataset.process, args.conf))

        elif args.data_type == "mono":
            futs.append(pool.submit(MonoDataset.process, args.conf))

        else:
            raise ValueError(f"Unknown data type {args.data_type}")

        wait(futs, return_when='FIRST_EXCEPTION')

        for fut in futs:
            if fut.exception() is not None:
                raise fut.exception()
