# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:37:10 2015

@author: doyung
"""

import pymongo
from pymongo import MongoClient

def tweeting():
    client = MongoClient()
    db = client['twitter']
    collection = db['ControlTweets']
    collection.find({'t'})


if __name__ == '__main__':
    tweeting()

