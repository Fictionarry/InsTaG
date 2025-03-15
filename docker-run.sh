#!/bin/bash

# Script to help run InsTaG commands inside the Docker container

# Ensure script exits on error
set -e

# Function to print usage information
print_usage() {
  echo "Usage: ./docker-run.sh COMMAND [ARGS]"
  echo ""
  echo "Available commands:"
  echo "  build                      - Build the Docker image"
  echo "  build-sapiens              - Build the Sapiens Docker image"
  echo "  shell                      - Open a shell in the container"
  echo "  sapiens-shell              - Open a shell in the Sapiens container"
  echo "  prepare                    - Run the prepare.sh script inside the container"
  echo "  prepare-sapiens            - Run the prepare_sapiens.sh script"
  echo "  download-easyportrait-model - Download the EasyPortrait model"
  echo "  convert-bfm                - Convert Basel Face Model (requires manual download first)"
  echo "  pretrain ARGS              - Run pretrain_con.sh with arguments (data source, output dir, gpu)"
  echo "  train ARGS                 - Run train_df_few.sh with arguments (data source, output dir, gpu)"
  echo "  process VIDEO_PATH         - Process a video using data_utils/process.py"
  echo "  teeth-mask PATH            - Generate teeth masks for a given person directory"
  echo "  extract-au PATH            - Extract facial Action Units for a person using OpenFace"
  echo "  extract-audio-features PATH TYPE - Extract audio features (types: deepspeech, wav2vec, hubert, ave)"
  echo "  run-sapiens PATH           - Generate Sapiens geometry priors for a given person"
  echo "  synthesize ARGS            - Run synthesize_fuse.py with arguments"
  echo ""
  echo "Examples:"
  echo "  ./docker-run.sh build"
  echo "  ./docker-run.sh shell"
  echo "  ./docker-run.sh pretrain data/pretrain output/pretrain_model 0"
  echo "  ./docker-run.sh train data/alice output/alice_model 0"
  echo "  ./docker-run.sh run-sapiens data/alice"
  echo "  ./docker-run.sh extract-audio-features data/alice/audio.wav wav2vec"
}

# Check if there are any arguments
if [ $# -eq 0 ]; then
  print_usage
  exit 1
fi

# Parse command
COMMAND=$1
shift

case $COMMAND in
  build)
    echo "Building Docker image..."
    docker-compose build instag
    ;;
    
  build-sapiens)
    echo "Building Sapiens Docker image..."
    docker-compose build sapiens
    ;;
    
  shell)
    echo "Opening shell in container..."
    docker-compose run --rm instag bash
    ;;
    
  sapiens-shell)
    echo "Opening shell in Sapiens container..."
    docker-compose run --rm sapiens bash
    ;;
    
  prepare)
    echo "Running prepare.sh in container..."
    docker-compose run --rm instag bash scripts/prepare.sh
    ;;
    
  download-easyportrait-model)
    echo "Downloading EasyPortrait model..."
    docker-compose run --rm instag bash -c "mkdir -p data_utils/easyportrait && \
      wget -O data_utils/easyportrait/fpn-fp-512.pth \
      https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/ep_models_v2/fpn-fp-512.pth || \
      echo 'Failed to download. Please check URL in training document and download manually.'"
    ;;
    
  convert-bfm)
    echo "Converting Basel Face Model..."
    if [ ! -f "data_utils/face_tracking/3DMM/01_MorphableModel.mat" ]; then
      echo "Error: Basel Face Model file not found."
      echo "Please download it from https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads"
      echo "and place it at data_utils/face_tracking/3DMM/01_MorphableModel.mat"
      exit 1
    fi
    docker-compose run --rm instag bash -c "cd data_utils/face_tracking && python convert_BFM.py"
    ;;
    
  prepare-sapiens)
    echo "Running prepare_sapiens.sh in Sapiens container..."
    docker-compose run --rm sapiens bash scripts/prepare_sapiens.sh
    ;;
    
  pretrain)
    if [ $# -lt 3 ]; then
      echo "Error: pretrain requires at least 3 arguments: data source, output dir, gpu"
      print_usage
      exit 1
    fi
    
    DATA_SOURCE=$1
    OUTPUT_DIR=$2
    GPU=$3
    shift 3
    
    echo "Running pretrain_con.sh with data from $DATA_SOURCE, output to $OUTPUT_DIR, gpu $GPU"
    docker-compose run --rm instag bash scripts/pretrain_con.sh $DATA_SOURCE $OUTPUT_DIR $GPU $@
    ;;
    
  train)
    if [ $# -lt 3 ]; then
      echo "Error: train requires at least 3 arguments: data source, output dir, gpu"
      print_usage
      exit 1
    fi
    
    DATA_SOURCE=$1
    OUTPUT_DIR=$2
    GPU=$3
    shift 3
    
    echo "Running train_df_few.sh with data from $DATA_SOURCE, output to $OUTPUT_DIR, gpu $GPU"
    docker-compose run --rm instag bash scripts/train_df_few.sh $DATA_SOURCE $OUTPUT_DIR $GPU $@
    ;;
    
  process)
    if [ $# -lt 1 ]; then
      echo "Error: process requires a video path"
      print_usage
      exit 1
    fi
    
    VIDEO_PATH=$1
    shift
    
    echo "Processing video at $VIDEO_PATH"
    docker-compose run --rm instag python data_utils/process.py $VIDEO_PATH $@
    ;;
    
  teeth-mask)
    if [ $# -lt 1 ]; then
      echo "Error: teeth-mask requires a path"
      print_usage
      exit 1
    fi
    
    PERSON_PATH=$1
    
    echo "Generating teeth masks for $PERSON_PATH"
    docker-compose run --rm instag bash -c "export PYTHONPATH=./data_utils/easyportrait && python data_utils/easyportrait/create_teeth_mask.py $PERSON_PATH"
    ;;
    
  extract-au)
    if [ $# -lt 1 ]; then
      echo "Error: extract-au requires a person path"
      print_usage
      exit 1
    fi
    
    PERSON_PATH=$1
    
    echo "Extracting facial Action Units for $PERSON_PATH using OpenFace..."
    docker-compose run --rm instag bash -c "mkdir -p $PERSON_PATH/au && \
      FeatureExtraction -fdir $PERSON_PATH/frames -out_dir $PERSON_PATH/au -aus && \
      cp $PERSON_PATH/au/*.csv $PERSON_PATH/au.csv"
    ;;
    
  extract-audio-features)
    if [ $# -lt 2 ]; then
      echo "Error: extract-audio-features requires an audio path and feature type"
      print_usage
      exit 1
    fi
    
    AUDIO_PATH=$1
    FEATURE_TYPE=$2
    
    echo "Extracting $FEATURE_TYPE audio features from $AUDIO_PATH"
    case $FEATURE_TYPE in
      deepspeech)
        docker-compose run --rm instag python data_utils/deepspeech_features/extract_ds_features.py --input $AUDIO_PATH
        ;;
      wav2vec)
        docker-compose run --rm instag python data_utils/wav2vec.py $AUDIO_PATH
        ;;
      hubert)
        docker-compose run --rm instag python data_utils/hubert.py $AUDIO_PATH
        ;;
      ave)
        echo "AVE features are processed on-the-fly during training/inference with --audio_extractor ave"
        ;;
      *)
        echo "Unknown feature type. Supported types: deepspeech, wav2vec, hubert, ave"
        exit 1
        ;;
    esac
    ;;
    
  run-sapiens)
    if [ $# -lt 1 ]; then
      echo "Error: run-sapiens requires a path"
      print_usage
      exit 1
    fi
    
    PERSON_PATH=$1
    
    echo "Generating Sapiens geometry priors for $PERSON_PATH using the Sapiens container"
    docker-compose run --rm sapiens bash data_utils/sapiens/run.sh $PERSON_PATH
    ;;
    
  synthesize)
    if [ $# -lt 2 ]; then
      echo "Error: synthesize requires at least -S and -M arguments"
      print_usage
      exit 1
    fi
    
    echo "Running synthesize_fuse.py with arguments: $@"
    docker-compose run --rm instag python synthesize_fuse.py $@
    ;;
    
  *)
    echo "Unknown command: $COMMAND"
    print_usage
    exit 1
    ;;
esac 