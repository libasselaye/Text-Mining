# fonctions permettant de visualiser facilement les vecteurs documents
# des options permettent de limiter (ou non) le nombre de lignes/colonnes affichées
import pandas as pd
import numpy 
import math

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import spacy
nlp = spacy.load('fr_core_news_lg')

import gensim
from gensim.utils import simple_preprocess

def top_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = numpy.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids if row[i]>0]
    df = pd.DataFrame(top_feats)
    if len(top_feats) > 0:     
        df.columns = ['feature', 'score']
    return df

#cette fonction permetra d'afficher les top_n mots avec le score le plus eleve du document entré en parametre
def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top features in specific document (matrix row) '''
    row = numpy.squeeze(Xtr[row_id].toarray())
    return top_feats(row, features, top_n)

# fonction calculant le cosinus entre deux vecteurs
def cosinus(i, j):
    num = numpy.dot(i, j)
    den = math.sqrt(sum(i*i))*math.sqrt(sum(j*j))
    if (den>0):    
        return (num/den)
    else:
        return 0
    
# fonction qui crée un dictionnaire associant le cosinus à chaque document puis le trie de manière décroissante
def search(q, D, n_docs):
    cc = {i: cosinus(D[i, :], q) for i in range(n_docs)}
    cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
    return cc

                 
#fonction permettant de calculer le centre d'inertie d'un ensemble de vecteurs mots
def centre(d,dim):
    m = numpy.zeros(shape=(1,dim))
    nbw = 0
    for w in d:
        try:
            v = nlp.vocab.vectors[nlp.vocab.strings[str(w)]]
            m = numpy.append(m, v.reshape((1,dim)), axis=0)     
            nbw += 1
        except:
            pass
    seuil = True
    if nbw>0:
        return (nbw, numpy.sum(m, axis=0)/nbw) # la normalisation est inutile si on utilise le cosinus
    else:
        return (0, m)
    
# fonction qui génère les listes de mots (token) à partir des textes
def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
        
#fonction permettant de retirer les mots-outils
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
