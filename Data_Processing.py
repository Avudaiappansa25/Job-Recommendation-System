import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD, NMF
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


def tokenize_stem(series):

    tokenizer =TreebankWordTokenizer()
    stemmer = PorterStemmer()
    series = series.apply(lambda x: x.replace("\n", ' '))
    series = series.apply(lambda x: tokenizer.tokenize(x))
    series = series.apply(lambda x: [stemmer.stem(w) for w in x])
    series = series.apply(lambda x: ' '.join(x))
    return series

def display_topics(model, feature_names, no_top_words, topic_names=None):
    '''
    displays topics and returns list of toppics
    '''

    topic_list = []
    for i, topic in enumerate(model.components_):
        if not topic_names or not topic_names[i]:
            print("\nTopic ", i)
        else:
            print("\nTopic: '",topic_names[i],"'")

        print(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_list.append(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
    return model.components_, topic_list

def return_topics(series, num_topics, no_top_words, model, vectorizer):
    '''
    returns document_topic matrix and topic modeling model
    '''
    #turn job into series
    series = tokenize_stem(series)
    #transform series into corpus
    ex_label = [e[:30]+"..." for e in series]
    #set vectorizer ngrams = (2,2)
    vec = vectorizer(stop_words = 'english')

    doc_word = vec.fit_transform(series)

    #build model
    def_model = model(num_topics)
    def_model = def_model.fit(doc_word)
    doc_topic = def_model.transform(doc_word)
    #print('model components: ', def_model.components_[0].shape)
    #print('doc_topic', doc_topic[0])
    model_components, topic_list = display_topics(def_model, vec.get_feature_names(), no_top_words)
    return def_model.components_, doc_topic, def_model, vec, topic_list#, topics


def process_data():
    '''
    uses the functions above to read in files, model, and return a topic_document dataframe
    '''
    #read in jobs file and get descriptions
    df = pd.read_csv('D:/Job_Recommender/ML_Project.csv')
    #df = df[df.keyword!='marketing']
    jobs_df = pd.DataFrame(zip(df['Job Description'], df['keyword']), columns = ['Description', 'Job'])

    array, doc, topic_model, vec, topic_list  = return_topics(jobs_df['Description'],20, 10, TruncatedSVD, TfidfVectorizer)

    topic_df = pd.DataFrame(doc)
    topic_df.columns = ['Topic ' + str(i+1) for i in range(len(topic_df.columns)) ]

    topic_df['job'] = jobs_df.Job
    #topic_df.to_csv('D:/Job_Recommender/topic_df2.csv')
    resume="Data enthusisast\n who is keen to work and do progress with capabilities of machine learning"
    resume=list(resume)
    resume = pd.Series(resume)
    doc = tokenize_stem(resume)
    doc = vec.transform(doc)
    #topic_model_name = 'D:/Job_Recommender/topic_model1.sav'
    #pickle.dump(topic_model, open(topic_model_name, 'wb'))
    vec_name = 'D:/Job_Recommender/job_vec1.sav'
    pickle.dump(vec, open(vec_name, 'wb'))
    return topic_df, topic_model, vec, topic_list
def predictive_modeling(df):
    '''
    fits, optimizes, and predicts job class based on topic modeling corpus
    '''
    X,y = df.iloc[:,0:-1], df.iloc[:, -1]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)

    param_grid = {'n_estimators': [100,200,300,400,500], 'max_depth': [3,7,9, 11]}
    #search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
    #search.fit(X_tr, y_tr)
    #bp = search.best_params_
    #print(bp)
    #rfc = RandomForestClassifier(n_estimators = bp['n_estimators'], max_depth = bp['max_depth'])
    rfc = RandomForestClassifier(n_estimators =3500, max_depth = 10)
    rfc.fit(X_tr, y_tr)
    print('acc: ', np.mean(cross_val_score(rfc, X_tr, y_tr, scoring = 'accuracy', cv=5)))
    print('test_acc: ', accuracy_score(y_te, rfc.predict(X_te)))
    print(rfc.predict(X_te))
    return rfc