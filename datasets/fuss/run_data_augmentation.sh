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


# TOP LEVEL VARIABLES ARE DEFINED IN setup.sh

SCRIPT_PATH=`dirname $0`

source ${SCRIPT_PATH}/setup.sh

bash ${SCRIPT_PATH}/get_raw_data.sh ${DOWNLOAD_DIR} ${RAW_DATA_DIR} ${FSD_DATA_URL} ${RIR_DATA_URL}
#get_raw_data：download fsd、 rir data
bash ${SCRIPT_PATH}/install_dependencies.sh

bash ${SCRIPT_PATH}/run_scaper.sh ${RAW_DATA_DIR} ${AUG_DATA_DIR} ${RANDOM_SEED} ${NUM_TRAIN} ${NUM_VAL}
# 返回fsd数据集下的所有wav文件的绝对路径
bash ${SCRIPT_PATH}/run_reverb_and_mix.sh ${RAW_DATA_DIR} ${AUG_DATA_DIR} ${RANDOM_SEED}
# return :read_mix_info:Read a mix_info_file to form REVERB_MIX_DIR/mix_info.txt
#rerurn :
echo Done!
echo Check directory ${AUG_DATA_DIR}_${RANDOM_SEED} to access the augmented
echo train/validation data.


