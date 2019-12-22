# Helper class to construct a "Language" object, to track dictionaries of words

# Start/End of Sentence symbols
SOS = "SOS"
SOS_token = 0
EOS = "EOS"
EOS_token = 1

class Lang:
    # Constructor - takes in name of language
    def __init__(self, name):
        self.name = name

        # Dictionaries to track words/indices
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: SOS, 1: EOS}
        
        self.num_words = 2

    # Adds all words in the given sentence
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    # Add a word to the dictionaries, or if already exists, increase count
    def addWord(self, word):
        if not word in self.word2index:
            self.index2word[self.num_words] = word
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.num_words += 1
        else:
            self.word2count[word] += 1

