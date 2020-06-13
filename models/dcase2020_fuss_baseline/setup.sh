#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Define ROOT_DIR variable which will hold all downloaded/generated model data.
# Uncomment next line and change to the directory for storing model data.
#ROOT_DIR=/mnt/hgfs/share/
ROOT_DIR=/home/yinghaorong/dcase

if [ x${ROOT_DIR} == x ]; then
  echo "Please define ROOT_DIR variable inside `dirname $0`/setup.sh."
  exit 1
fi

# Download directory to download data into.
# This can be set to /tmp if desired.
DOWNLOAD_DIR=${ROOT_DIR}/download

# The following should not be changed:

# Main directory for model data (e.g. trained model weights, eval results).
MODEL_DIR=${ROOT_DIR}/dcase2020_fuss

# Directory for pretrained baseline model (reverberant data).
BASELINE_MODEL_DIR=${MODEL_DIR}/2020-05-13_22-16-33
# Directory for pretrained baseline model (non-reverberant data).
BASELINE_DRY_MODEL_DIR=${MODEL_DIR}/fuss_desed_baseline_dry_2_inference

BASELINE_MODEL_URL="https://zenodo.org/record/3743844/files/FUSS_baseline_model.tar.gz"
BASELINE_DRY_MODEL_URL="https://zenodo.org/record/3743844/files/FUSS_baseline_dry_model.tar.gz"
