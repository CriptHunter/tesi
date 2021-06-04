#!/bin/bash

BENCHMARK_APP="org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
MODEL_PATHS=("/data/local/tmp/classifier.tflite" "/data/local/tmp/NeuMF.tflite" "/data/local/tmp/ECAM_NeuMF.tflite")
MODEL_NAMES=("mobile_log" "neumf_log" "ecam_neumf_log")

N_THREADS=1
N_RUNS=100
USE_GPU="true"
N_BENCHMARK=10

adb logcat -c # clear log cat

for i in "${!MODEL_NAMES[@]}"
do
    ARGS='"--graph='${MODEL_PATHS[i]}' --num_threads='${N_THREADS}' --num_runs='${N_RUNS}' --use_gpu='${USE_GPU}'"'
    echo $ARGS

    for k in $( eval echo {1..$N_BENCHMARK} )
    do
        echo -ne "Running test $k/$N_BENCHMARK\r"
        adb shell am start -S -n $BENCHMARK_APP --es args $ARGS > /dev/null 2>&1 # run benchmark app on Android device
        sleep 60  # wait otherwise it doesn't print anything in the logcat...
    done
    adb logcat -d | grep --line-buffered "Average inference" > ${MODEL_NAMES[i]} # write logcat on file
    adb logcat -c # clear log cat
done
