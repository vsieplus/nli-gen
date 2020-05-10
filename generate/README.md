# generate

This directory contains code for the entailment and contradiction generation models. Both have the same architecture, consisting of encoder and decoder RNNs using the LSTM unit as the recurrent unit. Word-by-word attention is also implemented as well. Generation is done using greedy decoding.

## Data

We use the Stanford Natural Language Inference (SNLI) Corpus to train, via `torchtext`. For each model, we filter the data to pairs of the correct type (i.e. 'entailment' vs. 'contradiction')

## Obtaining and Evaluating Models

To train the generation models, call

`bash run_training.sh MODEL_TYPE`

where `MODEL_TYPE` is `entailment` or `contradiction`. The resulting models will be saved to `models/entail-gen` and `models/contra-gen` respectively.

Similarly to evaluate the models, call

`python eval.py --model=MODEL_TYPE --contexts=PATH_TO_CONTEXTS`

where `PATH_TO_CONTEXTS` is a path to a file containing generation contexts, one per line. For entailment, a provided context should entail the generated sentence. Conversely for contradiction, a provided context should contradict the generated sentence. Results will be stored in `results/`.
