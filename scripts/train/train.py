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
import os

from bamboo_mixer.train import FormulaTrainer, MonoTrainer
from bamboo_mixer.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='train local')
parser.add_argument('--conf', type=str, default='config.yaml')
parser.add_argument('--task', type=str, default="formula")
args = parser.parse_args()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'

    if args.task == "formula":

        trainer = FormulaTrainer(args.conf)

    elif args.task == "mono":

        trainer = MonoTrainer(args.conf)

    else:
        raise ValueError(f"Unknown task {args.task}")

    trainer.train_loop()

    logger.info("Training finished!")


if __name__ == "__main__":
    main()
