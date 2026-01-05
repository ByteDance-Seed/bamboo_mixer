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

import numpy as np

from bamboo_mixer.data import DatasetConfig, FormulaDataset, MonoDataset
from bamboo_mixer.utils import get_data_file_path, setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description="process data and save to pkl")
parser.add_argument("--conf", type=str, help='config yaml file')
parser.add_argument("--num_workers", type=int, default=1, help='parallel processes')
parser.add_argument("--data_type", type=str, default="formula", help='formula or mono')

args = parser.parse_args()

with open(get_data_file_path('smiles_dict.json', 'bamboo_mixer.smiles_dict'), "r") as f:
    smiles_dict = json.load(f)


def normalize_entries(entries: list, properties: list, config: DatasetConfig):
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

    for _, entry in enumerate(entries):
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
            if np.isclose(max_val - min_val, 0):
                entry[prop] = 0.5
            else:
                entry[prop] = (entry[prop] - min_val) / (max_val - min_val)

    data_to_save = {"min_values": min_values, "max_values": max_values}

    # Save the min and max values to a JSON file
    with open(os.path.join(config._config['save_dir'], "min_max_values.json"), "w") as json_file:
        json.dump(data_to_save, json_file, indent=4)

    return entries


def preprocessing_mono(config: DatasetConfig):

    with open(config._config['json_path'], "r") as f:
        data = json.load(f)

    try:
        pretrain: bool = config._config["kwargs"]["pretrain"]
    except KeyError:
        logger.info('kwargs: pretrain not defined in config.yaml, default to False')
        pretrain: bool = False

    for molecule in data:

        assert "name" in molecule
        if "smiles" in molecule:
            pass
        elif molecule["name"] in smiles_dict:
            molecule["smiles"] = smiles_dict[molecule["name"]]
            logger.info(f"loaded smiles{molecule['smiles']} for {molecule['name']} from smiles_dict")
        else:
            raise ValueError(f"Missing smiles for {molecule['name']}")

        if not pretrain:
            assert "temperature" in molecule
            ## Normalization
            # The original unit of temperature is celsius
            molecule['temperature'] = (molecule['temperature'] + 273.15) / 373.15

    if pretrain:
        properties = [x for x in data[0].keys() if x not in {"name", "smiles"} and "mask" not in x]
        data = normalize_entries(data, properties, config)

    with open(os.path.join(config._config['save_dir'], "processed_data.json"), "w") as f:
        json.dump(data, f, indent=4)

    return config


def preprocessing_formula(config: DatasetConfig):

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
            elif salt["name"].replace("Li", "").replace("LI", "") in smiles_dict:
                salt["smiles"] = smiles_dict[salt["name"].replace("Li", "").replace("LI", "")]
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
        # you may change the normalize constants based on your data
        for k, v in formula.items():
            if k.startswith('conductivity') and k != 'conductivity_mask':
                formula[k] = (np.log(v) + 4.6002) / 8.3344
        # if "conductivity" in formula:
        #     formula['conductivity'] = (np.log(formula['conductivity']) + 4.6002) / 8.3344
        # if "conductivity_ne" in formula:
        #     formula['conductivity_ne'] = (np.log(formula['conductivity_ne']) + 4.6002) / 8.3344
        # if "conductivity_pred" in formula:
        #     formula['conductivity_pred'] = (np.log(formula['conductivity_pred']) + 4.6002) / 8.3344

    with open(os.path.join(config._config['save_dir'], "processed_data.json"), "w") as f:
        json.dump(data, f, indent=4)

    return config


if __name__ == "__main__":

    config = DatasetConfig(args.conf)

    logger.info(str(config))

    save_dir = config._config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f'save_dir {save_dir} created')
    else:
        logger.info(f'save_dir {save_dir} exists, may overwrite existing data')

    if args.data_type == "formula":

        config = preprocessing_formula(config)

    elif args.data_type == "mono":

        config = preprocessing_mono(config)

    else:
        raise ValueError(f"Unknown data type {args.data_type}")

    logger.info('Finished preprocessing data')

    if args.data_type == "formula":
        FormulaDataset.process(args.conf)
    elif args.data_type == "mono":
        MonoDataset.process(args.conf)
    else:
        raise ValueError(f"Unknown data type {args.data_type}")
