# NLI-Gen

This project explores the use of LSTM RNNs for Natural Language 
entailment and contradiction generation.

Given an input sentence A, we wish to generate output sentences B and C,
which A entails and contradicts, respectively. Informally, we say that A entails
B if when A is true, it follows that B is true. For example, "The pen is on the table"
entails that "There is something on the table." We say two sentences are contradictory
when they cannot both be true at the same time. Note that while contradiction is
symmetric, entailment is not in general.

The model is built using the Pytorch library. Code for models can be found in
`generate/`. Refer to `writeup/` for more details.