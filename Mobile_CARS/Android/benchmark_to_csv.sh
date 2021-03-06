#!/bin/bash

OUT_FILE="performances.csv"

declare -a FILES=("mobile_log" "neumf_log" "ecam_neumf_log")
declare -a DEVICES=("mobile" "neumf" "ecam neumf")

echo "model,warmup,init,inference" > $OUT_FILE

for i in "${!FILES[@]}"
do
    dev_i=${DEVICES[i]}
    dev_core_i="$dev_i,"   # concat device name and number of core, this variable will be passed to AWK

    # write to a CSV file with columns device, cores, warmup, init, inference
    awk -F 'in us: ' '{print $2}' ${FILES[i]} | awk -F 'Overall' '{print $1}' | sed 's/Warmup: //g' | sed 's/Init: //g' | 
                                                sed 's/Inference: //g' | sed 's/ *//g' | 
                                                awk -v var="$dev_core_i" '$0=var $0' >> $OUT_FILE
done