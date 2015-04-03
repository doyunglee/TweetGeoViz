# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:37:10 2015

@author: doyung
"""

import pymongo
import thinkstats2
import thinkplot
import datetime
import numpy as np
import statsmodels.api as sm
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from random import choice


def tweeting():
    all_words_vect = TfidfVectorizer()
    
    words_to_avoid = ['the','my','you','co','http','to']
    
    tweets_arr = np.empty([1,2], dtype='object')
    hypothesis_arr = np.empty([1,2], dtype='object')
    wide_area_tweets_arr = np.array([])

    words_dict = {}
    wide_area_words_dict = {}    
    
    client = MongoClient()
    db = client['twitter']
    collection = db['ControlTweets']
    #tlt is tweet latitude, which is around 31, tln is longitude, which is around -91
    start = datetime.datetime(2014,7,1,0,0,0,0)
    end = datetime.datetime(2014,11,15,0,0,0,0)
    
    #this finds all the tweets under the selected criteria. results is all the tweets in the outbreak zone. wide_area_results is the area around it.
    results = collection.find({'cc': 'US' , 'tlt': {"$gt": 40, "$lt": 41 }, 'tln': {"$gt": -74.5, "$lt": -73.5 }, 'cr': {'$gt': start, '$lt': end}}, limit=10000)
    wide_area_results = collection.find({'cc': 'US' , 'tlt': {"$gt": 39, "$lt": 42 }, 'tln': {"$gt": -75.5, "$lt": -72.5 }, 'cr': {'$gt': start, '$lt': end}}, limit=10000)    


    #going through the results and putting the texts of the tweets into an array.    
    for result in results:
        if "" in result['t']:
            tweets_arr = np.vstack((tweets_arr, np.array([(result['t'].encode('utf-8'), 'a')], dtype='object')))
            #westnile_times_arr.append((result['cr']-datetime.datetime(1970,1,1)).total_seconds())            
    for result in wide_area_results:
        if "" in result["t"] and result["t"] not in tweets_arr[:,0]:
            tweets_arr = np.vstack((tweets_arr, np.array([(result['t'].encode('utf-8'), 'b')], dtype='object')))
    
    local_tweets_arr = tweets_arr[tweets_arr[:,1] == 'a']
    wide_area_tweets_arr = tweets_arr[tweets_arr[:,1] == 'b']    
    
    #these are matricies with the array of tweet texts.
    all_words_vect.fit(local_tweets_arr[:,0])
    all_words_vect.fit(wide_area_tweets_arr[:,0])
    local_words_count = all_words_vect.transform(local_tweets_arr[:,0])
    wide_area_words_count = all_words_vect.transform(wide_area_tweets_arr[:,0])
    
    #print zip(all_words_vect.get_feature_names(), np.asarray(all_words_count.sum(axis=0).ravel())) #finds total number of times a word is shown
    #print all_words_vect.get_feature_names()
    
    local_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), np.asarray(wide_area_words_count.sum(axis=0).ravel())[0]/wide_area_words_count.shape[0]))
    
    wide_area_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), np.asarray(local_words_count.sum(axis=0).ravel())[0]/local_words_count.shape[0]))
    
    local_tweets_avg = local_word_avg[:,1]
    wide_area_tweets_avg = wide_area_word_avg[:,1]
        
    diff_tweets_avg = np.absolute(np.subtract(local_tweets_avg.astype(float),wide_area_tweets_avg.astype(float)))
    
    diff_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), diff_tweets_avg))

    sorted_dif_word_avg = diff_word_avg[diff_word_avg[:,1].argsort()]
    for i in sorted_dif_word_avg:
        if float(i[1]) >  0.001:       
            print i
        #print sorted_dif_word_avg[i][1]
    pdf = thinkstats2.EstimatedPdf(diff_tweets_avg)
    thinkplot.Pdf(pdf)
    thinkplot.Show(xlabel='Difference in Mean TF-IDF', ylabel="Proablity Density (in %)", title="Probablily Density of Mean Difference in TF-IDF")
    

    control_val = diff_tweets_avg[5]
    
    count = 0
    for i in range(1000):
        for row in np.nditer(tweets_arr[:,0], flags=['refs_ok']):
            hypothesis_arr = np.vstack((hypothesis_arr, np.array([str(row), choice(['a','b'])], dtype='object')))
        
        local_tweets_arr = hypothesis_arr[hypothesis_arr[:,1] == 'a']
        wide_area_tweets_arr = hypothesis_arr[hypothesis_arr[:,1] == 'b']
        
        #these are matricies with the array of tweet texts.
        all_words_vect.fit(local_tweets_arr[:,0])
        all_words_vect.fit(wide_area_tweets_arr[:,0])
        local_words_count = all_words_vect.transform(local_tweets_arr[:,0])
        wide_area_words_count = all_words_vect.transform(wide_area_tweets_arr[:,0])
        
        #print zip(all_words_vect.get_feature_names(), np.asarray(all_words_count.sum(axis=0).ravel())) #finds total number of times a word is shown
        #print all_words_vect.get_feature_names()
        
        local_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), np.asarray(wide_area_words_count.sum(axis=0).ravel())[0]/wide_area_words_count.shape[0]))
        
        wide_area_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), np.asarray(local_words_count.sum(axis=0).ravel())[0]/local_words_count.shape[0]))
        
        local_tweets_avg = local_word_avg[:,1]
        wide_area_tweets_avg = wide_area_word_avg[:,1]
            
        diff_tweets_avg = np.absolute(np.subtract(local_tweets_avg.astype(float),wide_area_tweets_avg.astype(float)))
        test_val = diff_tweets_avg[0]
        
        if test_val>=control_val:
            count +=1
             
    p = count/1000;
    print p
    #print sorted(wide_area_words_dict.items(), key=lambda x:x[1])
    
    #cdf = thinkstats2.Cdf(westnile_times_arr)
    #print westnile_times_arr[:]
    #print len(westnile_times_arr)
    
    #hist = thinkstats2.Hist(tweetlength_arr)    
    #thinkplot.Cdf(cdf, label = 'freq')
    #thinkplot.Hist(hist, label = 'tweet length')
    #print len(cdf.ps)    
    #print len(westnile_times_arr)
    #model = sm.OLS(westnile_times_arr, cdf.ps)
    #res = model.fit()
    #print res.summary()
    #thinkplot.Show()

if __name__ == '__main__':
    tweeting()

