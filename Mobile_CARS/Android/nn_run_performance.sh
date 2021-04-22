#!/bin/bash

BENCHMARK_APP="org.tensorflow.lite.benchmark/.BenchmarkModelActivity"
MODEL_PATH="/data/local/tmp/NeuMF.tflite"
N_THREADS=1
N_RUNS=1000
USE_GPU="true"
N_BENCHMARK=2

ARGS='"--graph='${MODEL_PATH}' --num_threads='${N_THREADS}' --num_runs='${N_RUNS}' --use_gpu='${USE_GPU}'"'

echo $ARGS

adb logcat -c # clear log cat

for i in $( eval echo {1..$N_BENCHMARK} )
do
    echo -ne "Running test $i/$N_BENCHMARK\r"
    adb shell am start -S -n $BENCHMARK_APP --es args $ARGS > /dev/null 2>&1 # run benchmark app on Android device
    sleep 5  # wait otherwise it doesn't print anything in the logcat...
done

adb logcat -d | grep --line-buffered "Average inference" > neumf_log # write logcat on file
