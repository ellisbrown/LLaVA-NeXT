#!/bin/bash

# Default values
GPUS=8
SHARED_MEMORY="250GiB"
CLUSTERS="80gb"
NODES=1
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
    echo "  --clusters <name>          Cluster name or combination (default: 80gb)"
    echo "                             Examples:"
    echo "                               jupiter"
    echo "                               jupiter+saturn"
    echo "                               80GB"
    echo "                               all"
    echo "  --nodes <number>           Number of nodes (default: 1)"
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
        --clusters)
            CLUSTERS="$2"
            shift; shift ;;
        --nodes)
            NODES="$2"
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


# ------------------------------------------------------------------------------
# A simple function that maps a short cluster name to its full name.
# Returns an empty string if the short name is unknown.
# ------------------------------------------------------------------------------
get_cluster_fullname() {
    case "$1" in
        # NVIDIA H100 80GB
        jupiter)  echo "ai2/jupiter-cirrascale-2" ;;
        ceres)    echo "ai2/ceres-cirrascale" ;;
        # NVIDIA A100 80GB
        saturn)   echo "ai2/saturn-cirrascale" ;;
        # NVIDIA L40 48GB
        neptune)  echo "ai2/neptune-cirrascale" ;;
        *)        echo "" ;;
    esac
}

# ------------------------------------------------------------------------------
# parse_clusters():
# Accepts a single cluster name, plus-separated cluster names (e.g. "jupiter+saturn"),
# or the special keyword "all" to get *all* known clusters.
# Returns an array of cluster_fullnames via stdout.
# ------------------------------------------------------------------------------
parse_clusters() {
    local input="$1"
    local cluster_fullname
    local cluster_array=()

    if [[ "$input" == "all" ]]; then
        # Add all known clusters. Adjust as needed.
        cluster_array+=($(get_cluster_fullname "jupiter"))
        cluster_array+=($(get_cluster_fullname "ceres"))
        cluster_array+=($(get_cluster_fullname "saturn"))
        cluster_array+=($(get_cluster_fullname "neptune"))
    elif [[ "$input" == "80gb" ]]; then
        # Add all 80GB clusters (A100 + H100)
        cluster_array+=($(get_cluster_fullname "jupiter"))
        cluster_array+=($(get_cluster_fullname "ceres"))
        cluster_array+=($(get_cluster_fullname "saturn"))
    else
        # Split on "+" sign for multiple clusters
        IFS='+' read -ra splitted <<< "$input"
        for c in "${splitted[@]}"; do
            cluster_fullname="$(get_cluster_fullname "$c")"
            if [[ -z "$cluster_fullname" ]]; then
                log "Invalid cluster: $c"
                exit 1
            fi
            cluster_array+=("$cluster_fullname")
        done
    fi

    # Echo them space-separated so we can capture in an array later
    echo "${cluster_array[@]}"
}

# ------------------------------------------------------------------------------
# Parse the user-provided cluster(s) into an array
# ------------------------------------------------------------------------------
read_clusters=($(parse_clusters "$CLUSTERS"))
[[ ${#read_clusters[@]} -eq 0 ]] && {
    log "No valid cluster(s) specified."
    exit 1
}

clusters_yaml=""
for c in "${read_clusters[@]}"; do
    clusters_yaml+="'${c}',"
done

# get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
YAML_PATH="$DIR/beaker_exp_base.yaml"

# create extended description: cluster_RxG + desc + script basename (no extension)
script_basename=$(basename -- "$EXP_SCRIPT_PATH")
script_basename_no_ext="${script_basename%.*}"
DESCRIPTION="${DESCRIPTION}_${script_basename_no_ext}"
# "names may contain letters, numbers, the characters -_. and may not start with -"
DESCRIPTION="${DESCRIPTION//[^a-zA-Z0-9._-]/_}"
extended_desc="${CLUSTERS}_${NODES}x${GPUS}_${DESCRIPTION}"

# transform script path to relative to project root
fullpath=$(realpath $EXP_SCRIPT_PATH)
# keep everything after "LLaVA-NeXT"
relativepath=${fullpath#*LLaVA-NeXT/}
# PROJ_ROOT="/data/weka/ellisb/LLaVA-NeXT"


################# Submit Job #################
# set environment variables
export NODES
export GPUS
export SHARED_MEMORY
export CLUSTERS=$clusters_yaml
export DESCRIPTION
export EXTENDED_DESCRIPTION=$extended_desc
export EXP_SCRIPT_PATH=$relativepath

# submit the job
CMD="beaker experiment create $YAML_PATH"

log "Submitting job"
echo "> $CMD"
echo "Variables:"
echo "  - nodes: $NODES"
echo "  - gpus: $GPUS"
echo "  - shared_memory: $SHARED_MEMORY"
echo "  - clusters: $CLUSTERS"
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