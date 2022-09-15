#!/usr/bin/env bash

DISEASE=../full_data/va_disease_outcome_training.csv
POP=../full_data/va_person.csv
GRAPH=../full_data/va_population_network.csv

N_JOBS=10

mkdir -p ../sub_data/logistic_regression/train/
mkdir -p ../sub_data/logistic_regression/eval/

for (( PART=0; PART < N_JOBS; PART++ ))
do
    now=$(date +"%T")
    echo "Preprocess time : $now"
    time python make_subset_data.py $GRAPH $DISEASE $POP  ../full_data/logistic_regression/train/train_$PART.csv --pid_partition $PART --n_jobs $N_JOBS &
    time python make_subset_data.py $GRAPH $DISEASE $POP  ../full_data/logistic_regression/eval/eval_$PART.csv --min-date 50 --is-eval --pid_partition $PART --n_jobs $N_JOBS
done

#python logistic_regression.py --training_dir ../full_data/logistic_regression/train/ --eval_dir ../full_data/logistic_regression/eval/ \
#    --pop $POP --eval-labels ../full_data/va_disease_outcome_target.csv
