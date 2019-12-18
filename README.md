# NLI-Gen

This project explores the use of deep learning methods to assist Natural
Language Inference (NLI) with entailment and contradiction generation.

There are two tasks to be specified.

## (1) Entailment and Contradiction Generation

Given an input sentence A, we wish to generate output sentences b1, b2, ...,
and c1, c2, ... which A entails and contradicts. In addition, we are also
interested in in the converse task of given an input sentence A, generating 
output sentences d1, d2, … which entail A. Note that the analogous task for
contradiction is identical to the original, since contradiction between two 
sentences is symmetric.

## (2) Natural Language Inference

Given a hypothesis sentence H, we aim to determine whether A entails,
contradicts, or is neutral to H. In theory, this classification process will be
aided by the results of the generation processes described above. In particular,
if the generation works well we can aid the direct inference of A to H by the
inferences of Bi, Cj, Dk to H, and potentially vice versa. For instance, if we
assume that A entails B1 and B1 entails H, we expect A to entail H. The 
specifics of these interactions are specified further within the particular
implementation, and depends on the nature of each inference, and how well the
generation seems to be doing.

As such, there are two separate models as part of this project - one for
generation, and one for inference. For further implementation details, refer
to `report.pdf`.
