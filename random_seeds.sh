#!/bin/bash
declare -a arr=("2452435" "4363462" "134123156", "5456736", "12326236")

for seed in "${arr[@]}"
do
    command="python run.py --model random --seed $seed --do_property --epochs 1 --log_file log_random_$seed.txt --plot_fp results_$seed.png"
    echo "$command"
    eval "$command"
done
