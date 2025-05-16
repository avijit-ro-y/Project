import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("train.csv")
small_df=df.sample(37000,random_state=2) #take only 30k data and create a new dataframe as there are not sufficient RAM
small_df = small_df.fillna('')

#----------------------------------preprocessing----------------------------------
def preprocessing(q):
    q=str(q).lower().strip()
    
    q=q.replace('%','percent')
    q=q.replace('$','dollar')
    q=q.replace('@','at')
    q=q.replace('[math]','')
    q=q.replace(',000,000,000','b')
    q=q.replace(',000,000','m')
    q=re.sub(r'([0-9]+)000000000',r'\1b',q)
    q=re.sub(r'([0-9]+)000000',r'\1m',q)
    q=re.sub(r'([0-9]+)000',r'\1k',q)

    contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'd've": "he would have",
        "he'll": "he shall",
        "he'll've": "he shall have",
        "he's": "he has",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how has",
        "I'd": "I had",
        "I'd've": "I would have",
        "I'll": "I shall",
        "I'll've": "I shall have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it had",
        "it'd've": "it would have",
        "it'll": "it shall",
        "it'll've": "it shall have",
        "it's": "it has",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she had",
        "she'd've": "she would have",
        "she'll": "she shall",
        "she'll've": "she shall have",
        "she's": "she has",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that has",
        "there'd": "there had",
        "there'd've": "there would have",
        "there's": "there has",
        "they'd": "they had",
        "they'd've": "they would have",
        "they'll": "they shall",
        "they'll've": "they shall have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we had",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what shall",
        "what'll've": "what shall have",
        "what're": "what are",
        "what's": "what has",
        "what've": "what have",
        "when's": "when has",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where has",
        "where've": "where have",
        "who'll": "who shall",
        "who'll've": "who shall have",
        "who's": "who has",
        "who've": "who have",
        "why's": "why has",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you had",
        "you'd've": "you would have",
        "you'll": "you shall",
        "you'll've": "you shall have",
        "you're": "you are",
        "you've": "you have"
    }
    q_decontracted=[]
    for word in q.split():
        if word in contractions:
            word=contractions[word]
        q_decontracted.append(word)
    q=' '.join(q_decontracted)
    q=q.replace('ve','have')
    q=q.replace("n't",'not')
    q=q.replace('re','are')
    q=q.replace('ll','will')
    
    #html tag
    q=BeautifulSoup(q)
    q=q.get_text()
    
    #punctuation
    pattern=re.compile('\W')
    q=re.sub(pattern,' ',q).strip()
    return q

small_df['question1']=small_df['question1'].apply(preprocessing) #create new collumn
small_df['question2']=small_df['question2'].apply(preprocessing)

#--------------------------------Feature Engineering--------------------------------
small_df['q1_len']=small_df['question1'].str.len() #create new collumn
small_df['q2_len']=small_df['question2'].str.len()

small_df['q1_num_words']=small_df['question1'].apply(lambda row: len(row.split(" "))) #create new collumn
small_df['q2_num_words']=small_df['question2'].apply(lambda row: len(row.split(" ")))

def common_word(row):
    w1=set(map(lambda word : word.lower().strip(),row['question1'].split(" ")))
    w2=set(map(lambda word : word.lower().strip(),row['question2'].split(" ")))
    return len(w1 & w2)
small_df['word_common']=small_df.apply(common_word,axis=1)

def total_word(row):
    w1=set(map(lambda word : word.lower().strip(),row['question1'].split(" ")))
    w2=set(map(lambda word : word.lower().strip(),row['question2'].split(" ")))
    return len(w1) + len(w2)
small_df['word_total']=small_df.apply(total_word,axis=1)

small_df['word_share']=round(small_df['word_common']/small_df['word_total'],2)

#---------------------------Advance Feature Engineering----------------------------
from nltk.corpus import stopwords

def fetch_token_feature(q1,q2):
    SAFE_DIV=0.0001
    STOP_WORDS=stopwords.words("english")
    token_feature=[0.0]*8
    
    q1_token=q1.split()  #converting the sentence into token
    q2_token=q2.split() #converting the sentence into token
    if len(q1_token)==0 or len(q2_token)==0:
        return token_feature
    
    #get nonstop words in question
    q1_words=set([word for word in q1_token if word not in STOP_WORDS])
    q2_words=set([word for word in q2_token if word not in STOP_WORDS])

    #get the stopwords in question
    q1_stopwords=set([word for word in q1_token if word in STOP_WORDS])
    q2_stopwords=set([word for word in q2_token if word in STOP_WORDS])
    
    #get common nonstopwords from question pair
    common_word_count=len(q1_words.intersection(q2_words))
    
    #get common stopwords from question pair
    common_stopword_count=len(q1_stopwords.intersection(q2_words))

    #get common token from question pair
    common_token_count=len(set(q1_token).intersection(set(q2_words)))
    
    token_feature[0]=common_word_count/(min(len(q1_words),len(q2_words))+SAFE_DIV)
    token_feature[1]=common_word_count/(max(len(q1_words),len(q2_words))+SAFE_DIV)
    token_feature[2]=common_stopword_count/(min(len(q1_stopwords),len(q2_stopwords))+SAFE_DIV)
    token_feature[3]=common_stopword_count/(max(len(q1_stopwords),len(q2_stopwords))+SAFE_DIV)
    token_feature[4]=common_token_count/(min(len(q1_token),len(q2_token))+SAFE_DIV)
    token_feature[5]=common_token_count/(max(len(q1_token),len(q2_token))+SAFE_DIV)

    #last word of both question is same or not
    token_feature[6]=int(q1_token[-1]==q2_token[-1])
    
    #first word of both question is same or not
    token_feature[7]=int(q1_token[0]==q2_token[0])
    
    return token_feature

token_features=small_df.apply(lambda row: fetch_token_feature(row['question1'], row['question2']), axis=1)
token_features = list(token_features)

small_df['cwc_min']=list(map(lambda x:x[0],token_features))
small_df['cwc_max']=list(map(lambda x:x[1],token_features))
small_df['csc_min']=list(map(lambda x:x[2],token_features))
small_df['csc_max']=list(map(lambda x:x[3],token_features))
small_df['ctc_min']=list(map(lambda x:x[4],token_features))
small_df['ctc_max']=list(map(lambda x:x[5],token_features))
small_df['last_word']=list(map(lambda x:x[6],token_features))
small_df['first_word']=list(map(lambda x:x[7],token_features))

import distance
def fetch_length_feature(q1,q2):
    length_feature=[0.0]*3
    
    q1_token=q1.split()  #converting the sentence into token
    q2_token=q2.split() #converting the sentence into token
    if len(q1_token)==0 or len(q2_token)==0:
        return length_feature
    
    #absulate length feature
    length_feature[0]=abs(len(q1_token)-len(q2_token))
    
    #average token length of both question
    length_feature[1]=(len(q1_token)+len(q2_token))/2
    
    strs=list(distance.lcsubstrings(q1,q2))
    length_feature[2]=len(strs[0])/(min(len(q1),len(q2))+1)
    return length_feature

length_features = small_df.apply(lambda row: fetch_length_feature(row['question1'], row['question2']), axis=1)
length_features= list(length_features)

small_df['abs_len_deff']=list(map(lambda x:x[0],length_features))
small_df['mean_len']=list(map(lambda x:x[1],length_features))
small_df['longest_substr_ratio']=list(map(lambda x:x[2],length_features))

from fuzzywuzzy import fuzz
def fetch_fuzzy_feature(q1,q2):
    fuzzy_feature=[0.0]*4
    
    #fuzz_ratio
    fuzzy_feature[0]=fuzz.QRatio(q1,q2)
    
    #fuzz_partial_ratio
    fuzzy_feature[1]=fuzz.partial_ratio(q1,q2)

    #token_sort_ratio
    fuzzy_feature[2]=fuzz.token_sort_ratio(q1,q2)
    
    #token_set_ratio
    fuzzy_feature[3]=fuzz.token_set_ratio(q1,q2)
    
    return fuzzy_feature

fuzzy_features = small_df.apply(lambda row: fetch_fuzzy_feature(row['question1'], row['question2']), axis=1)
fuzzy_features = list(fuzzy_features)

small_df['fuzzy_ratio']=list(map(lambda x:x[0],fuzzy_features))
small_df['fuzz_partial_ratio']=list(map(lambda x:x[1],fuzzy_features))
small_df['token_sort_ratio']=list(map(lambda x:x[2],fuzzy_features))
small_df['token_set_ratio']=list(map(lambda x:x[3],fuzzy_features))


ques_df=small_df[['question1','question2']] #take 2 collumn and again create a new dataframe
ques_df = ques_df.fillna('') #for handelling the null value

final_df=small_df.drop(columns=['id','qid1','qid2','question1','question2'])

##------------------------------------Modelling-------------------------------------
from sklearn.feature_extraction.text import CountVectorizer #applyint BOW
total_ques=list(ques_df['question1'])+list(ques_df['question2']) #sum the 2 column and put them in a list
cv=CountVectorizer(max_features=9500)
q1_array,q2_array=np.vsplit(cv.fit_transform(total_ques).toarray(),2) #convert the text data into vector and then again split them into 2 section and store them in 2 variable
temp_df1=pd.DataFrame(q1_array,index=ques_df.index) #convert them dataframe with the original index
temp_df2=pd.DataFrame(q2_array,index=ques_df.index)
temp_df=pd.concat([temp_df1,temp_df2],axis=1) #again build new dataframe of 2 column
final_df=pd.concat([final_df,temp_df],axis=1)


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(final_df.iloc[:,1:].values,final_df.iloc[:,0].values,test_size=0.2)

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
xgb=XGBClassifier()
xgb.fit(xtrain,ytrain)
ypred=xgb.predict(xtest)
accuracy=accuracy_score(ytest,ypred)

def test_common_word(q1,q2):
    w1=set(map(lambda word:word.lower().strip(),q1.split(" ")))
    w2=set(map(lambda word:word.lower().strip(),q2.split(" ")))
    return len(w1 & w2)

def test_total_word(q1,q2):
    w1=set(map(lambda word:word.lower().strip(),q1.split(" ")))
    w2=set(map(lambda word:word.lower().strip(),q2.split(" ")))
    return (len(w1) + len(w2)) 

#---------------------------------------main---------------------------------------
def query_point_creator(q1,q2):
    input_query=[]
    
    #preprocessing
    q1=preprocessing(q1) #preprocessing of q1
    q2=preprocessing(q2) #preprocessing of q2
    
    #fetch basic feature
    input_query.append(len(q1))
    input_query.append(len(q2))
    
    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))
    
    input_query.append(test_common_word(q1,q2))
    input_query.append(test_total_word(q1,q2))
    input_query.append(round(test_common_word(q1,q2)/test_total_word(q1,q2),2))
    
    #fetch token feature
    token_feature=fetch_token_feature(q1,q2)
    input_query.extend(token_feature)
    
    #fetch length feature
    length_feature=fetch_length_feature(q1,q2)
    input_query.extend(length_feature)
    
    #fetch fuzzy feature
    fuzzy_feature=fetch_fuzzy_feature(q1,q2)
    input_query.extend(fuzzy_feature)
    
    #bow feature for q1
    q1_bow=cv.transform([q1]).toarray()
    
    #bow feature for q2
    q2_bow=cv.transform([q2]).toarray()
    
    return np.hstack((np.array(input_query).reshape(1,22),q1_bow,q2_bow))


