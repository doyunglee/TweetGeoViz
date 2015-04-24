# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:37:10 2015

@author: doyung
"""

import pymongo
import thinkstats2
import thinkplot
import datetime
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from random import choice


def tweeting(epi,r1,r2,tI,tF):
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
    #start = tI
    #end = tF\

    start = datetime.datetime(2014,5,1,0,0,0,0)
    end = datetime.datetime(2014,6,30,0,0,0,0)
    #this finds all the tweets under the selected criteria. results is all the tweets in the outbreak zone. wide_area_results is the area around it.
    #results = collection.find({'cc': 'US' , 'tlt': {"$gt": epi-r1, "$lt": epi+r1 }, 'tln': {"$gt": epi-r1, "$lt": epi+r1 }, 'cr': {'$gt': start, '$lt': end}}, limit=10000)
    #wide_area_results = collection.find({'cc': 'US' , 'tlt': {"$gt": epi-r2, "$lt": epi+r2}, 'tln': {"$gt": epi-r2, "$lt": epi+r2 }, 'cr': {'$gt': start, '$lt': end}}, limit=10000)

    results = collection.find({'cc': 'US' , 'tlt': {"$gt": float(epi[0])-r2, "$lt": float(epi[0])+r2}, 'tln': {"$gt": float(epi[2])-r2, "$lt": float(epi[2])+r2 }, 'cr': {'$gt': start, '$lt': end}}, limit=10000)

    results = pd.DataFrame(list(results));

    local_tweets_df = results[(epi-r1<results.tlt<epi+r1)and(epi-r1<results.tln<epi+r1)];
    wide_tweets_df = results[not((epi-r1<results.tlt<epi+r1)and(epi-r1<results.tln<epi+r1))];

    local_tweets_arr = np.asarray(local_tweets_df.t)
    local_tweets_arr = np.hstack((local_tweets_arr, np.zeros((a.shape[0], 1), dtype='object')))
    local_tweets_arr[:,1] = 'a'

    wide_area_tweets_arr = np.asarray(wide_tweets_df.t)
    wide_area_tweets_arr = np.hstack((wide_tweets_arr, np.zeros((a.shape[0], 1), dtype='object')))
    wide_area_tweets_arr[:,1] = 'b'

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

    #diff_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), diff_tweets_avg))
    diff_word_avg = np.rec.fromarrays((all_words_vect.get_feature_names(),diff_tweets_avg), names=('features', 'diffs'));
    sorted_dif_word_avg = diff_word_avg[diff_word_avg[:,1].argsort()]
    print sorted_dif_word_avg
    return sorted_dif_word_avg
    #for i in sorted_dif_word_avg:
        #if float(i[1]) >  0.001:
            #print i
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
    tweeting([31, 100], 1,1,0,0)

