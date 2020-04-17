#!/bin/bash

if [ "$1" == "entailment" ]; then
    output_dir="models/entail-gen"
else if [ "$1" == "contradiction" ]; then
    output_dir="models/contra-gen"
else
    echo "invalid argument(s)"
    exit 1
fi

echo 'Training in progress...'
python3 train.py --output_dir=$output_dir --model_type=$1

echo "model saved to $output_dir"