# encoding: utf-8
# Helper functions mainly for read and write
import csv, json, pickle, math, time, re
from sklearn.externals import joblib
from xml.etree import ElementTree

# Save the classifier and tfidf_transformer
def save_model(path,clf,clf_name,vectorizer=None,vectorizer_name=None):
    import os
    from sklearn.externals import joblib
    
    cwd = os.getcwd()
    os.chdir(path)
    
    clf_msg = joblib.dump(clf, clf_name+".pk1")
    if vectorizer != None and vectorizer_name != None:
        vectorizer_msg = joblib.dump(vectorizer, vectorizer_name+".pk1")
    
    os.chdir(cwd)

# Function to load classifier and vectorizer
def load_model(path, clf_name, vectorizer_name):
    """
    Load the classifiers and word transformer
    Input:
        path: a string describing the absolute path to the directory that stores pk1 files of the model
    Output:
        clf_sub, clf_main: they are two model object used 
        tfidf_transformer
    """
    clf = joblib.load(path + "/"+clf_name+".pk1")
    tfidf_transformer = joblib.load(path + "/"+vectorizer_name+".pk1")
    
    return clf, tfidf_transformer

def load_model_mainAndSub(path, vectorizer_name, main_clf_name, cates_tree_name):
    """
    Rebuild the main and sub tree structure and load all models
    Return x vectorizer, main classifier and a dictionary containing sub classifiers
    """
    import os
    from sklearn.externals import joblib
    cwd = os.getcwd()
    os.chdir(path)

    # Load the tree structure of categories
    with open(cates_tree_name, "r") as infile:
        cates_tree = json.load(infile)
    main_cates = cates_tree.keys()
    
    # Load the vectorizer and main classifier
    vectorizer = joblib.load(vectorizer_name+".pk1")
    main_clf = joblib.load(main_clf_name+".pk1")
    
    # Load sub classifiers
    sub_clfs = {}
    for main_cate in main_cates:
        if main_cate == "Software":
            continue
        sub_clfs[main_cate] = joblib.load(main_cate+".pk1")

    os.chdir(cwd)
    return vectorizer, main_clf, sub_clfs

# Function to transform SQL data lists to X and Y
def data_toXY(data, truncated_cate = []):
    """
    Function to transform SQL data lists to X and Y
    """
    X, Y = [], []
#     X_vali, Y_vali = [],[]
#     X_test, Y_test = [],[]

    unicode_skip = 0
    for i, d in enumerate(data):
        id = d[0]
        try:    
            title = unicode(d[1].lower())
            desc = unicode(d[2].lower())
            cate = unicode(d[3].lower())
        except:
            unicode_skip += 1
            continue
        x = desc + (title+" ")*3# create an instance of x: 3 titles plus description
        x = x.replace("\r", "") # remove line changer "\r"

        # put firs 60 % in training
        if (i % 5) in [0,1,2,3,4]:
            X.append(x) 
            Y.append(cate)
#         # 20% in validation
#         elif (i % 5) in [3,4]:
#             X_vali.append(x)
#             Y_vali.append(cate)
#         # 20% in test, so far no need
#         else:
#             X_test.append(x)
#             Y_test.append(cate)

    print "# of instance ignored during transformation:", unicode_skip
    print "# of training samples:", len(X)
    
    # Remove data of certain categories 
    print("The category wants to remove is {}".format(truncated_cate))
    X_clean = []
    Y_clean = []
    for i, y in enumerate(Y):
        if y not in truncated_cate:
            X_clean.append(X[i])
            Y_clean.append(Y[i])
    print("After removing certain categories, the length of X and Y is {}, {}".format(len(X_clean), len(Y_clean)))
    
    return X_clean, Y_clean

# Save a list of list of strings into csv
def save_csv(path, data, delimiter = ","):
    with open(path, "wb") as f:
        writer = csv.writer(f, delimiter = delimiter)
        for d in data:
            writer.writerow(d)

# Load csv formatted data
def load_csv(path, has_header = False, delimiter = ","):
    with open(path, "rb") as f:
        reader = csv.reader(f, delimiter = delimiter)
        res = [row for row in reader]
        if has_header == True:
            res = res[1:]
        return res

# Load csv formatted verified data
def load_verified(path, no_header = True, delimiter = ","):
    with open(path, "rb") as f:
        reader = csv.reader(f, delimiter = delimiter)
        res = [row for row in reader]
        if no_header == True:
            res = res[1:]
        return res

def verified_toXY(verified):
    """
    Transform verified data to X and Y pairs
    """
    X, Y = [], []
    cleaner = re.compile('<.*?>')
    for case in verified:
        title = re.sub(cleaner, "", case[2])
        desc = re.sub(cleaner, "", case[3])
        cate = re.sub(cleaner, "", case[17])
        X.append(" ".join([desc, title, title, title]))
        Y.append(cate)
    
    return X, Y

# Function to extract the weight of a feature given certain class
def get_weight(weights, class_name, feature_name,classes,features):
    try:
        i = classes.index(class_name)
    except valueError:
        raise ValueError("Doesn't have this class: {}".format(class_name))
    
    try:
        j = features.index(feature_name)
    except ValueError:
        raise ValueError("Doesn't have this feature: {}".format(feature_name))
        
    return weights[i,j]

# A fucntion that outputs top 3 results
def topK_predictor(x, k, clf, vectorizer):
    """
    Predict the top K categories for a given x
    """
    x = vectorizer.transform([x])
    decisions = clf.decision_function(x)[0]
    topK_indice = sorted([i for i in range(len(decisions))], \
                         key=lambda j: -decisions[j])[:k]
    topK_cates = [clf.classes_[i] for i in topK_indice]
    
    return topK_cates

# Remove non-alphabetical characters
def only_alphabeta(string):
    """
    Concatenate consecutive spaces and tabs and newlines into one space;
    Remove non-alphabetical characters;
    To lowercases;
    """
    no_dash = string.replace("-", " ")
    no_slash = string.replace("/", " ")
    only_oneSpace = re.sub("\s+", " ", no_slash)
    only_en = re.sub("[^a-zA-Z ]", "", only_oneSpace)
    return only_en.lower()

# Only keep the stem of words
def stem_word(string):
    """
    Stem the plural words into singular. 
    Remove last "s", "es", "ies". 
    Transform "men" to "man" and "women" to "woman"
    """
    string = list(string)
    
    if len(string) > 1 and string[-1] == "s":
        string.pop()

    if len(string) > 1 and string[-1] == "e":
        string.pop()
    
    if len(string) > 1 and string[-1] == "i":
        string.pop()

    if len(string) > 2 and string[-1] == "n" and string[-2] == "e":
        string[-2] = "a"

    if len(string) < 2:
        return ""
    else:
        return "".join(string)

# Clean a list of strings
def clean_input(strings):
    """
    Clean the list of string using the same standard as cleaning the training data
    """
    cleaned = []
    for string in strings:
        only_aB = only_alphabeta(string)
        stemmed = [stem_word(word) for word in only_aB.split()]
        cleaned.append(" ".join(stemmed))
    
    return cleaned


# Clean the data use the same standard as in building Doc2Vec
import string
from nltk.corpus import stopwords
printables = set(string.printable)
stop_words = set(stopwords.words("english"))
def clean_forDoc2Vec(s, 
                     printables = printables,
                     stop_words = stop_words):
    """
    Clean the given string using the standards for building Doc2Vec model
    """
    # Retain only printables characters (removes non-eng chars)
    s = filter(lambda x: x in printables, s)
    
    # All to lower cases
    s = s.lower()
    
    # Replace connectors and punctuations into space
    pattern = "[\.\,\:\-\?\!\_\'\/]"
    s = re.sub(pattern, " ", s)
    
    # Remove brackets <>, [], () and contents in them
    pattern = "[\(\[\<].*?[\)\]\>]"
    s = re.sub(pattern, "", s)
    
    # Remove all \n, \t space-like characters
    pattern = "[\n\t]"
    s = re.sub(pattern, " ", s)
    
    # Remove all numbers
    pattern = "\d"
    s = re.sub(pattern, "", s)
    
    # Remove stop words
    exclusions = {"he", "him", "his", "she", "her"}
    stop_words = set(filter(lambda x: x not in exclusions, stop_words))
    splited = s.split()
    s = " ".join([token for token in splited\
                       if token not in stop_words])
    
    # Remove any words that is less than 2 characters long
    least_length = 2
    splited = s.split()
    s = " ".join([token for token in splited \
                       if (len(token)>=least_length or token=="he")])
    
    # Replace all multiple spaces with one space
    pattern = "\s\s+"
    s = re.sub(pattern, " ", s)
    
    return s
