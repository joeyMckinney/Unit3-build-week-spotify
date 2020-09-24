import logging
import random

from fastapi import APIRouter
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator
import joblib
import json

log = logging.getLogger(__name__)
router = APIRouter()

knn = joblib.load('app/api/knn.joblib')
dataset = pd.read_csv('app/api/herokuspotify.csv')

def getfeatures(id,df):
  """Gets Features From track for Neural Network Model"""
  features = df.copy()
  if id in (features['track_id'].values):
    features = features[features['track_id']==id]
    features = features[['danceability','energy','key','loudness','mode', 
                         'speechiness', 'acousticness','instrumentalness',
                         'liveness' , 'valence','tempo']]
    return features
  else:
    raise Exception('Track ID Not Found')


@router.get('/predict/{track_id}')
async def predict(track_id: str):
    song_f = getfeatures(track_id, dataset)
    predtion = knn.kneighbors(song_f)
    arr = predtion[1][0]

    data = np.array([])
    for i,_ in enumerate(arr):
        data = np.append(data,dataset['track_id'][dataset.index==arr[i]])


    print(data)
    return {
        'song1': data[0],
        'song2': data[1],
        'song3': data[2],
        'song4': data[3],
        'song5': data[4],
        'song6': data[5],
        'song7': data[6],
        'song8': data[7],
        'song9': data[8],
        'song10': data[9]
    }