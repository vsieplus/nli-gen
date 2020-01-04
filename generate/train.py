# Script for training the generation model, and providing user functionality
# to interact with it afterwards

import preprocessing

import time
import math

###############################################################################
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
###############################################################################

torch.manual_seed(321)

# Hyperparams
NUM_EPOCHS = 1
HIDDEN_SIZE = 256
LEARNING_RATE = 0.005
BATCH_SIZE = 16

# Negative likelihood loss, with SGD
loss_F = nn.NLLLoss()

optimizer = 

for epoch 
