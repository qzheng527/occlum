#!/usr/bin/env bash
###############################################
# File Name: run_kube.sh
# Author: Liang Jiang
# mail: jiangliang0811@gmail.com
# Created Time: Sat 25 Mar 2023 05:51:30 PM CST
# Description: 
###############################################

set -e
set -x

script=$1
output_dir=$2
resume_timestamp=$3
if [ $resume_timestamp"x" == "x" ]; then
  resume_timestamp=$(date "+%Y%m%d-%H%M%S")
  resume_from_checkpoint=false
else
  resume_from_checkpoint=true
fi
output_dir=$output_dir/`basename $script`/$resume_timestamp

sh $script $output_dir $resume_from_checkpoint
