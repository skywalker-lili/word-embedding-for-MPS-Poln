import numpy

def sent2vec(sent, w2v, return_word_miss = False):
    """
    Tranform a sentence to a vector by averaging each word vectors
    """
    sent = sent.split() # split into a list of words
    new_sent = []
    word_miss = 0 # number of words not in the word2vec model
    
    if len(sent) < 1:
        raise Exception("The sentence should contain at least one word!")
    
    # Replace each word with vectrors
    for word in sent:
        try:
            new_sent.append(w2v[word.lower()]) # lower case word
        except:
            word_miss += 1
            continue
    
    # Average the vectors
    new_sent = numpy.array(new_sent)
    new_sent_avg = numpy.mean(new_sent, axis=0)
    
    if not return_word_miss:
        return new_sent_avg
    else:
        return new_sent_avg, word_miss
