import numpy as np
import pandas as pd
import re
import string
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#load model
with open('static/model/model.pickle', 'rb') as f:
    model = pickle.load(f)

#load stopwords
with open('static/model/corpora/stopwords/english', 'r') as file:
    sw = file.read().splitlines()

#load tokens
vocab = pd.read_csv('static/model/vocabulary.tex', header=None)
tokens = vocab[0].tolist()   

def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
        return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE) for x in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuation)
    data["tweet"] = data["tweet"].str.replace('\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data['tweet']

def vectorizer(ds):
    vectorized_list = []

    for sentance in ds:
        sentance_list = np.zeros(len(tokens))

        for i in range(len(tokens)):
            if tokens[i] in sentance.split():
                sentance_list[i] =1

        vectorized_list.append(sentance_list)

    vectorized_list_new = np.asarray(vectorized_list, dtype= np.float32)

    return vectorized_list_new

def get_prediction(vectorizer_txt):
    prediction = model.predict(vectorizer_txt)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'