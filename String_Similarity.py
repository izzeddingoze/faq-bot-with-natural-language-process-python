import nltk
from nltk.tokenize import *
from nltk.corpus import stopwords
import string
from TurkishStemmer import TurkishStemmer
import math

stemmer = TurkishStemmer()
stop = set(stopwords.words("Turkish"))
stop.clear()
stop.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','şey',
                 'mi','misin','mısın','musun','müsün','ile','mu','miyim',
                 'miyiz','misiniz','gibi','sonra','önce','için','de', 'niçin',
                 'şu', 'ya', 'hep', 'bu', 'çok', 'ise', 'mü', 'daha', 'hiç',
                 'her','acaba', 'az', 'aslında',
                 'da', 'ile', 'sanki', 'ki'])



def clean_stop_words(sentence):
    tmp_sentence_array=[i for i in word_tokenize(sentence.lower()) if i not in stop]
    sentence_array=[]
    [sentence_array.append(stemmer.stem(y)) for y in tmp_sentence_array]
    return sentence_array

def jakard_similarity_rate(sentence1, sentence2):

    array_of_sentence1=clean_stop_words(sentence1)
    array_of_sentence2= clean_stop_words(sentence2)
    tmp_combine_array=array_of_sentence1+array_of_sentence2

    combine_array=[]
    [combine_array.append(y) for y in tmp_combine_array if y not in combine_array]

    kesisim=(len(array_of_sentence1) + len(array_of_sentence2)) - len(combine_array)
    return  kesisim / len(combine_array)

def vector_of_document (all_document):
    tmp_sentence=""
    for x in range(0, len(all_document)):
        tmp_sentence=tmp_sentence + all_document[x]+" "

    tmp_vector=clean_stop_words(tmp_sentence)
    vector_of_doc=[]
    [vector_of_doc.append(y) for y in tmp_vector if y not in vector_of_doc]
    return vector_of_doc

def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
    return dot_product/magnitude

def tf_vector(vector_of_doc,document):
    vector=[]
    count=0
    doc_array = clean_stop_words(document)
    for x in range(0, len(vector_of_doc)):
        for y in range(0, len(doc_array)):
            if (vector_of_doc[x] == doc_array[y]):
                count=count+1
        vector.append(count)
        count=0
    return vector

def idf_vector_of_document(vector_of_doc,tf_vectors):
    count=0
    idf_vector=[]
    for x in range(0, len(vector_of_doc)):
        for y in range(0, len(tf_vectors)):
            if(tf_vectors[y][x]>=1):
                count=count+1;
        tmp=len(tf_vectors)/count
        idf_vector.append(1 + math.log(tmp,math.e))
        count=0

    return idf_vector

def normalize_tf_vectors(tf_vectors,normalize_val):

    for x in range(0, len(tf_vectors)):
        for y in range(0, len(tf_vectors[0])):
            tf_vectors[x][y]=tf_vectors[x][y]/normalize_val

    return tf_vectors

def tf_idf_vector(tf_vector, idf_vector):

    tf_idf_vector=[]
    for x in range(0,len(idf_vector)):
        tf_idf_vector.append(tf_vector[x]*idf_vector[x])
    return tf_idf_vector

def jensen_shanon_rate(vector1, vector2):
    tmp_vector1=[]
    tmp_vector2=[]
    for x in range(0, len(vector1)):
        if(vector1[x]==0 and vector2[x]==0):
            continue
        else:
            tmp_vector1.append(vector1[x]+1)
            tmp_vector2.append(vector2[x]+1)
    sum1=sum(tmp_vector1)
    sum2=sum(tmp_vector2)
    rate=0
    for x in range(0, len(tmp_vector1)):
        tmp1 = tmp_vector1[x] / sum1
        tmp2 = tmp_vector2[x] / sum2
        if (vector1[x] == 0 and vector2[x] == 0):
            continue
        else:
            rate += ((math.log(((tmp1) / ((tmp1 + tmp2) / 2))) * tmp1) + (math.log(((tmp2) / ((tmp1 + tmp2) / 2))) * tmp2)) / 2;

    return  rate








