# Evaluation script for the generation model

import data
import argparse
import gen_encoder
import gen_decoder

import argparse



# Function to perform evaluation/write out results of model for a given batch
def test_batch(batch, encoder, decoder, df):
    curr_batch_size = batch.batch_size
    result_dicts = [{} for _ in range(curr_batch_size)]

    for b in range(curr_batch_size):
        result_dicts[b]["premise"] = ""
        for w in range(batch.premise[b, :].size(1)):
            word_idx = batch.premise[b,w]
            result_dicts[b]["premise"] += pp.inputs.iots[word_idx] + " "

    # Feed input through encoder, store packed output + context
    encoder_outputs, (hn, cn) = encoder(batch.premise,
        encoder.initHidden(curr_batch_size), encoder.initCell(curr_batch_size), 
        curr_batch_size)

    # Decoder setup -> forward propogation
    decoder_input = torch.tensor(pp.inputs.init_token, dtype = torch.long).view(-1,1)
    decoder_input = torch.tensor([decoder_input] * curr_batch_size)

    context = (hn, cn)
    decoder_hidden = context[0].squeeze()
    decoder_cell = context[1].squeeze()

    # Feed actual target token as input to next timestep
    for timestep in range(min(processing.MAX_SENT_LEN, target_length)):
        decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(
            decoder_input, decoder_hidden, decoder_cell, 
            encoder_outputs[continue_idxs, :])

        continue_idxs = []

        # Continue/print generation for data in batch that have not passed <EOS> yet,
        for b in range(curr_batch_size):
            curr_target_idx = batch.hypothesis[b, timestep][0]

            # If current target index 
            if curr_target_idx != 0:
                continue_idxs.append(b)

        # Input to next timestep are argmax indices of decoder output
        decoder_input = torch.argmax(decoder_output[continue_idxs, :], dim = -1)

        # Detokenize (to text) and write results to file
        for b in range(curr_batch_size):
            result_dicts[b]["hypothesis"] = ""
            if b in continue_idx:
                word_b = pp.inputs.iots[decoder_input[b,0]]        
                result_dicts[b]["hypothesis"] += word_b + " "

    rows_list.extend(result_dicts)

rows_list_entail = []
for batch_num, batch in enumerate(pp.test_iter_entail):
    test_batch(batch, rows_list_entail)

rows_list_contradict = []
for batch_num, batch in enumerate(pp.test_iter_contradict):
    test_batch(batch, rows_list_contradict)

df_entail = pd.DataFrame(rows_list_entail, columns = ("premise", "hypothesis"))
df_contradict = pd.DataFrame(rows_list_contradict, columns = ("premise", "hypothesis"))
df_entail.to_csv("model_test_outputs_entailments.tsv", sep = "\t")
df_contradict.to_csv("model_test_outputs_contradictions.tsv", sep = "\t")

def main():
    device = torch.device("cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--contexts", type=str)

    args = parser.parse_args()

    print("Loading model")
    encoder = gen_encoder.Encoder(data.inputs.vocab.max_size, data.HIDDEN_SIZE,
        embeddings = data.GLOVE_VECS_200D)
    decoder = gen_decoder.Decoder(pp.inputs.vocab.max_size, data.HIDDEN_SIZE, 
        embeddings = data.GLOVE_VECS_200D)

    model = torch.load(PATH, map_location=device)
    encoder.load_state_dict(model['encoder_state_dict'])
    decoder.load_state_dict(model['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    print("Starting Evaluation")

    encoder.eval()
    decoder.eval()

    with open(args.contexts, "r") as f:
        gen_contexts = f.read().splitlines() 


if __name__ == "__main__":
    main()