set -xe

pip3 install -U pre-commit yapf pylint
pre-commit install
git lfs install