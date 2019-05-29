#!/bin/bash
# run processes and store pids in array
counter=0
for i in $@; do
     webots --mode=fast --batch --minimize $i &
     pids[counter]=$!
     ((counter++))
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done