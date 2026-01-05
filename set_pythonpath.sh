# source this script to set PYTHONPATH 

PROJECTPATH=$(git rev-parse --show-toplevel)
export PYTHONPATH=$PROJECTPATH:$PYTHONPATH
echo $PYTHONPATH
