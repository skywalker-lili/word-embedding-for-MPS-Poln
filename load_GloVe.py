from collections import defaultdict

def load_GloVe(gloveFile):
    """
    Return the word-vector mapping in a defaault dictionary
    , default option is None.
    * input type: string, a path to GloVe txt file
    * output type: dictionary, mapping between words and vecotrs
    """
    print("Loading GloVe")
    model = defaultdict(lambda: None)
    with open(gloveFile,'r') as f:
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = embedding
    print("Done,", len(model), "words loaded!")
    return model
