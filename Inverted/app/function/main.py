import PyPDF2
import docx2txt
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import math
import string
import re
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import nltk

def get_content_file(directory):
    extensions = ['pdf','docx','txt']
    document_contents = []
    id_doc = 1

    for filename in os.listdir(directory):
        path = directory+'\\'+filename
        text = ""
        if(filename[len(filename)-3:] == 'pdf'):
            pdf_file = open(path,'rb')
            pdf_reader = PyPDF2.PdfFileReader(pdf_file)
            num_pages = pdf_reader.numPages
            for i in range(num_pages):
                page = pdf_reader.getPage(i)
                text += page.extractText()
        elif(filename[len(filename)-4:] == 'docx'):
            text = docx2txt.process(path)
        elif(filename[len(filename)-3:] == 'txt'):
        	f = open(path,'r')
        	for t in f.readlines():
        		text += ' '+t

#         document = Document(id_doc, filename, text, path)
        document_contents.append((id_doc, filename, text, path))
        id_doc += 1
    return document_contents

def preprocessing(document):
    tokens = []
    document = re.sub(r'^https?:\/\/.*[\r\n]*', '', document, flags=re.MULTILINE)
    
    document = document.lower()
    document = ''.join([word for word in document if  not word.isdigit()])
    
    for token in nltk.word_tokenize(document):
            tokens.append(token)
    
    punc = string.punctuation
    tokens =  [token.strip(punc) for token in tokens]
    
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    
    stemmer = PorterStemmer()
    for i in range(0, len(tokens)):
        if (tokens[i] != stemmer.stem(tokens[i])):
            tokens[i] = stemmer.stem(tokens[i])
            
    return tokens

all_document = get_content_file('D:/Kuliah/Semester 8/INRE/Proyek/example')
N = len(all_document)

tokens = []
for i in range(N):
    tokens.append(preprocessing(all_document[i][2]))

data = []
for i in range(N):
    for w in tokens[i]:
        data.append(w)

new_sentence = ' '.join([w for w in data])

for w in CountVectorizer().build_tokenizer()(new_sentence):
    data.append(w)
    
data = set(data)

def preprosQuery(query):

    queri=[]
    spl = query.split()
    for i in range(len(spl)):
        if not spl[i].isdigit():
            queri.append(spl[i])
            
    # define punctuation  
    punc = string.punctuation

    # remove punctuation from the string  
    no_punc =  [token.strip(punc) for token in queri]
    # display the unpunctuated string  

    lower=[]
    for i in range(len(no_punc)):
        lower.append(no_punc[i].lower())
        
    stop = []
    stop_words = set(stopwords.words('english'))
    for i in range(len(lower)):
        if lower[i] not in stop_words:
                stop.append(lower[i])

    stem = []
    stemmer = PorterStemmer()
    for i in range(len(stop)):
        stem.append(stemmer.stem(stop[i]))
        
    join_word = ' '.join([w for w in stem])
    n = len(stem)

    return join_word,n

# join_word, n = preprosQuery(query)

def generate_ngrams(data, n):
    ngram=[]
    result = []
    
    #menampilkan hasil n-gram per dokumen
    for i in range(len(data)):
        sequences = [data[i][j:] for j in range(n)]
        temp = zip(*sequences)
        lst = list(temp)
        result.append([" ".join(lst) for lst in lst])
    
    #menggabungkan n-gram semua dokumen dalam bentuk array
    for i in range(len(result)):
        for j in range(len(result[i])):
            ngram.append(result[i][j])
            
    return ngram, result

# ngram, ngram_doc = generate_ngrams(tokens, n)


def countDF(N,ngram,query):
    df = []

    for i in range(N):
        count = 0
        for j in range(len(ngram[i])):
            if query == ngram[i][j]:
                count+=1
        df.append(count)
        
    return df

def countIDF(N,document,query):
    # idf
    df = countDF(N,document,query)
    idf = []
    for i in range(N):
        try:
            idf.append(math.log10(N/df[i]))
        except ZeroDivisionError:
            idf.append(str(0))
            
    return idf

def countWTD(N,document,query,ngram):
    #w(t, d)
    #t = term
    #d = document
    hasil = []
    l = []
    # weight=[]
    idf = countIDF(N,ngram,query)
    for i in range(N):
        wtd = {}
        tf = ngram[i].count(query) # menghitung nilai tf
        if tf != 0:
            score = math.log10(tf) #log10(tf(t,d))
            score+=1 # 1 + log(tf(t,d))
            score*=idf[i] #tf * idf
            
            idx = document[i][0]
            title = document[i][1] # filename
            path = document[i][3]

            # weight.append((idx,title,path,score))
            # dic[idx] = score
            wtd['docno'] = idx
            wtd['title'] = title
            wtd['path'] = path
            wtd['score'] = score # [i+1] = defenisi nomor dokumen; score = wtd
            
            l.append(wtd)
            
    hasil.append(l)
    return hasil


def main_function(query):
    join_word, n = preprosQuery(query)
    ngram, ngram_doc = generate_ngrams(tokens, n)
    # indexing ngram
    n_gram_index = {}
    for ngram_token in ngram:
        doc_no = []
        for i in range(N):
            if(ngram_token in ngram_doc[i]):
                doc_no.append(all_document[i][0])
        n_gram_index[ngram_token] = doc_no

    weight = countWTD(N,all_document,join_word,ngram_doc)
    hasil = []
    hasil.append(sorted(weight[0], key = lambda x : x['score'], reverse = True))

    return hasil