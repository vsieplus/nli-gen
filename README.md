# NLI-Gen

This project explores the use of deep learning methods to assist Natural
Language Inference (NLI) with entailment and contradiction generation.

There are two tasks to be specified.

## (1) Entailment and Contradiction Generation

Given an input sentence A, we wish to generate output sentences B<sub>1</sub>, B<sub>2</sub>, 
..., and C<sub>1</sub>, C<sub>2</sub>, ... which A entails and contradicts. In addition, we are also
interested in in the converse task of given an input sentence A, generating 
output sentences D<sub>1</sub>, D<sub>2</sub>, … which entail A. Note that the analogous task for
contradiction is identical to the original, since contradiction between two 
sentences is symmetric.

## (2) Natural Language Inference

Given an input sentence A and hypothesis sentence H, we aim to determine whether A entails,
contradicts, or is neutral to H. In theory, this classification process will be
aided by the results of the generation processes described above. In particular,
if the generation works well we can aid the direct inference of A to H by the
inferences of B<sub>i</sub>, C<sub>j</sub>, D<sub>k</sub> to H, and potentially vice versa. For instance, if we
assume that A entails B<sub>1</sub> and B<sub>1</sub> entails H, we expect A to entail H. The 
specifics of these interactions are specified further within the particular
implementation, and depends on the nature of each inference, and how well the
generation seems to be doing.

As such, there are two separate models as part of this project - one for
generation, and one for inference - both are built using the Pytorch library.
For further implementation details, refer to `writeup/writeup.pdf`.
