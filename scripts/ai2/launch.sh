#!/bin/bash

# Default values
GPUS=8
SHARED_MEMORY="250GiB"
CLUSTER="jupiter"
REPLICAS=1
DESCRIPTION="vidS2R"
EXP_SCRIPT_PATH=""

# Function to log with timestamps
log() {
    printf "\033[34m%s\033[0m %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

# Function to print help message
print_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --gpus <number>            Number of GPUs to use (default: 8)"
    echo "  --shared_memory <size>     Size of shared memory (default: 250GiB)"
    echo "  --cluster <name>           Cluster name (default: jupiter)"
    echo "  --replicas <number>        Number of replicas (default: 1)"
    echo "  --description <text>       Description of the experiment (default: ov_train)"
    echo "  --script <path>            Path to the experiment script (REQUIRED)"
    echo "  --help                     Display this help message"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpus)
            GPUS="$2"
            shift; shift ;;
        --shared_memory)
            SHARED_MEMORY="$2"
            shift; shift ;;
        --cluster)
            CLUSTER="$2"
            shift; shift ;;
        --replicas)
            REPLICAS="$2"
            shift; shift ;;
        --description)
            DESCRIPTION="$2"
            shift; shift ;;
        --script)
            EXP_SCRIPT_PATH="$2"
            shift; shift ;;
        --help)
            print_help
            exit 0 ;;
        *)
            log "Unknown argument: $1"
            exit 1 ;;
    esac
done

# Check for required arguments
if [[ -z "$EXP_SCRIPT_PATH" ]]; then
    log "Error: --exp_script_path is required."
    exit 1
elif [[ ! -f "$EXP_SCRIPT_PATH" ]]; then
    log "Error: File not found: $EXP_SCRIPT_PATH"
    exit 1
fi


# Map short cluster names to full cluster names
case $CLUSTER in
    saturn)
        # A100_80GB
        cluster_fullname="ai2/saturn-cirrascale"
        ;;
    jupiter)
        # H100
        cluster_fullname="ai2/jupiter-cirrascale-2"
        ;;
    ceres)
        # H100
        cluster_fullname="ai2/ceres-cirrascale"
        ;;
    neptune)
        cluster_fullname="ai2/neptune-cirrascale"
        ;;
    *)
        log "Error: Unknown cluster name: $CLUSTER"
        exit 1
        ;;
esac

# get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
YAML_PATH="$DIR/beaker_exp_base.yaml"

# create extended description: cluster_RxG + desc + script basename (no extension)
script_basename=$(basename -- "$EXP_SCRIPT_PATH")
script_basename_no_ext="${script_basename%.*}"
DESCRIPTION="${DESCRIPTION}_${script_basename_no_ext}"
# "names may contain letters, numbers, the characters -_. and may not start with -"
DESCRIPTION="${DESCRIPTION//[^a-zA-Z0-9._-]/_}"
extended_desc="${CLUSTER}_${REPLICAS}x${GPUS}_${DESCRIPTION}"

# transform script path to relative to project root
fullpath=$(realpath $EXP_SCRIPT_PATH)
# keep everything after "LLaVA-NeXT"
relativepath=${fullpath#*LLaVA-NeXT/}
# PROJ_ROOT="/data/weka/ellisb/LLaVA-NeXT"


################# Submit Job #################
# set environment variables
export REPLICAS
export GPUS
export SHARED_MEMORY
export CLUSTER=$cluster_fullname
export DESCRIPTION
export EXTENDED_DESCRIPTION=$extended_desc
export EXP_SCRIPT_PATH=$relativepath

# submit the job
CMD="beaker experiment create $YAML_PATH"

log "Submitting job"
echo "> $CMD"
echo "Variables:"
echo "  - replicas: $REPLICAS"
echo "  - gpus: $GPUS"
echo "  - shared_memory: $SHARED_MEMORY"
echo "  - cluster: $CLUSTER"
echo "  - description: $DESCRIPTION"
echo "  - extended_description: $EXTENDED_DESCRIPTION"
echo "  - exp_script_path: $EXP_SCRIPT_PATH"
echo ""

eval $CMD

if [ $? -ne 0 ]; then
    log "Error submitting job"
    exit 1
fi

log "Job submitted successfully"