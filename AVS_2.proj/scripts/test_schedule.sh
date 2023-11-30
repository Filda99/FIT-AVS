#!/bin/bash

num_runs=10

data_files=$(ls ./data/*.pts)
data_files_count=$(ls ./data/*.pts | wc -l)
total_runs=$(($data_files_count * $num_runs))

schedules=("static" "guided" "dynamic" "dynamic,8" "dynamic,16" "dynamic,32" "dynamic,64")

for schedule in "${schedules[@]}"; do
  sed -i -E "s/^(#pragma.* schedule)\(.*\)(.*)$/\1($schedule)\2/" ./parallel_builder/loop_mesh_builder.cpp
  cd build && make -j > /dev/null && cd ..
  
  ms_total_count=0

  echo "--------------------------- $schedule ---------------------------"

  for file in $data_files; do
    ms_data_file_count=0
    
    for i in $(seq 1 $num_runs); do
      output=$(./build/PMC --builder loop $file)

      ms_elapsed=$(echo $output | sed -n 's/.*Elapsed Time: \([0-9]*\) ms.*/\1/p')
      ms_total_count=$(($ms_total_count + $ms_elapsed))
      ms_data_file_count=$(($ms_data_file_count + $ms_elapsed))
    done

    ms_data_file_average=$(($ms_data_file_count / $num_runs))
    echo "$file average over $num_runs runs - $ms_data_file_average ms"

  done

  ms_average=$(($ms_total_count / $total_runs))
  echo "Average across all the data files - $ms_average ms"
  echo "---------------------------------------------------------------"

done
