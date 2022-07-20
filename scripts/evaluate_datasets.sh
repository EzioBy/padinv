#!/bin/bash

# Help information.
if [[ $# -lt 5 || ${*: -1} == "-h" || ${*: -1} == "--help" ]]; then
    echo "This script evaluates the similarity between two datasets."
    echo
    echo "Usage: $0 GPUS SRC_DATASET TGT_DATASET RESOLUTION METRICS"
    echo
    echo "Note: More than one metric should be separated by comma." \
         "Also, all metrics assume using all samples from both the source and" \
         "target datasets."
    echo
    echo "Example: $0 1 ~/data/rendered_images.zip ~/data/real_images.zip 512" \
         "fid,kid"
    echo
    exit 0
fi

# Get an available port for launching distributed training.
# Credit to https://superuser.com/a/1293762.
export DEFAULT_FREE_PORT
DEFAULT_FREE_PORT=$(comm -23 <(seq 49152 65535 | sort) \
                    <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) \
                    | shuf | head -n 1)

GPUS=$1
SRC_DATASET=$2
TGT_DATASET=$3
RESOLUTION=$4
PORT=${PORT:-$DEFAULT_FREE_PORT}

# Parse metrics to test.
METRICS=$5
TEST_FID="false"
TEST_KID="false"
if [[ ${METRICS} == "all" ]]; then
    TEST_FID="true"
    TEST_KID="true"
else
    array=(${METRICS//,/ })
    for var in ${array[@]}; do
        if [[ ${var} == "fid" ]]; then
            TEST_FID="true"
        fi
        if [[ ${var} == "kid" ]]; then
            TEST_KID="true"
        fi
    done
fi

# Detect `python3` command.
# This workaround addresses a common issue:
#   `python` points to python2, which is deprecated.
export PYTHONS
export RVAL

PYTHONS=$(compgen -c | grep "^python3$")

# `$?` is a built-in variable in bash, which is the exit status of the most
# recently-executed command; by convention, 0 means success and anything else
# indicates failure.
RVAL=$?

if [ $RVAL -eq 0 ]; then  # if `python3` exist
    PYTHON="python3"
else
    PYTHON="python"
fi

${PYTHON} -m torch.distributed.launch \
    --nproc_per_node=${GPUS} \
    --master_port=${PORT} \
    ./eval_dataset_similarity.py \
        --src_dataset ${SRC_DATASET} \
        --tgt_dataset ${TGT_DATASET} \
        --resolution ${RESOLUTION} \
        --src_num -1 \
        --tgt_num 50000 \
        --test_fid ${TEST_FID} \
        --test_kid ${TEST_KID}
