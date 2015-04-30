# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:37:10 2015

@author: Doyung and Austin
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

    limit = 10000
    all_words_vect = TfidfVectorizer(stop_words="english", ngram_range=(1,3), max_df=.50, min_df=20/limit)


    tweets_arr = np.empty([1,2], dtype='object')
    hypothesis_arr = np.empty([1,2], dtype='object')
    wide_area_tweets_arr = np.array([])

    words_dict = {}
    wide_area_words_dict = {}

    client = MongoClient()
    db = client['twitter']
    collection = db['ControlTweets']
    #tlt is tweet latitude, which is around 31, tln is longitude, which is around -91
    start = tI
    end = tF

    results = collection.find({'cc': 'US' , 'tlt': {"$gt": float(epi[0])-r2, "$lt": float(epi[0])+r2}, 'tln': {"$gt": float(epi[1])-r2, "$lt": float(epi[1])+r2 }, 'cr': {'$gt': start, '$lt': end}}, limit=10000)

    results = pd.DataFrame(list(results));

    local_tweets_df = results.query('(@epi[0]-@r1<tlt<@epi[0]+@r1) and (@epi[1]-@r1<tln<@epi[1]+@r1)');
    local_tweets_df['loc'] = 'a'
    wide_tweets_df = results.query('not((@epi[0]-@r1<tlt<@epi[0]+@r1) and (@epi[1]-@r1<tln<@epi[1]+@r1))');
    wide_tweets_df['loc'] = 'b'

    local_tweets_arr = local_tweets_df.as_matrix(['t', 'loc']);
    wide_area_tweets_arr = wide_tweets_df.as_matrix(['t', 'loc']);


    #these are matricies with the array of tweet texts.
    all_words_vect.fit(local_tweets_arr[:,0])
    all_words_vect.fit(wide_area_tweets_arr[:,0])
    local_words_count = all_words_vect.transform(local_tweets_arr[:,0])
    wide_area_words_count = all_words_vect.transform(wide_area_tweets_arr[:,0])

    local_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), np.asarray(wide_area_words_count.sum(axis=0).ravel())[0]/wide_area_words_count.shape[0]))

    wide_area_word_avg = np.asarray(zip(all_words_vect.get_feature_names(), np.asarray(local_words_count.sum(axis=0).ravel())[0]/local_words_count.shape[0]))

    local_tweets_avg = local_word_avg[:,1]
    wide_area_tweets_avg = wide_area_word_avg[:,1]

    diff_tweets_avg = np.absolute(np.subtract(local_tweets_avg.astype(float),wide_area_tweets_avg.astype(float)))

    diff_word_avg_df = pd.DataFrame({'features': all_words_vect.get_feature_names(), 'diffs': diff_tweets_avg})

    sorted_diff_word_avg_df = diff_word_avg_df.sort(['diffs'], ascending=False)
    print sorted_diff_word_avg_df
    #return sorted_diff_word_avg_df
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
    tweeting([40.5, -74],1,3,datetime.datetime(2014,7,1,0,0,0,0),datetime.datetime(2014,11,15,0,0,0,0) )

