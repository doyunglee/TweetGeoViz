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


    local_tweets_avg = np.asarray(local_words_count.sum(axis=0).ravel())[0]/local_words_count.shape[0]

    wide_area_tweets_avg = np.asarray(wide_area_words_count.sum(axis=0).ravel())[0]/wide_area_words_count.shape[0]

    expected_avg = (local_tweets_avg + wide_area_tweets_avg)/2

    chi2s = (((local_tweets_avg-expected_avg)**2)/expected_avg)+(((wide_area_tweets_avg-expected_avg)**2)/expected_avg)

    diff_tweets_avg = np.absolute(np.subtract(local_tweets_avg.astype(float),wide_area_tweets_avg.astype(float)))

    final_df = pd.DataFrame({'features': all_words_vect.get_feature_names(), 'diffs': diff_tweets_avg, 'chi2s':chi2s})

    final_df  = final_df.sort(['chi2s'], ascending=False)
    print final_df

if __name__ == '__main__':
    tweeting([37, 122],1,3,datetime.datetime(2014,5,20,0,0,0,0),datetime.datetime(2014,6,30,0,0,0,0) )

