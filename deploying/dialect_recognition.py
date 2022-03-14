import tensorflow as tf
from joblib import load
import pickle
import regex as re
import numpy as np
from string import punctuation

class Dialect_Recognition:
    def __init__(self) -> None:
        self.nn_model_path = "../models/model0.477.h5"
        self.cnb_model_path = "../models/cnb_054.joblib"
        
        self.nn_model = tf.keras.models.load_model(self.nn_model_path)
        self.cnb_model = load(self.cnb_model_path)
        
        with open("../models/word2idx.pickle","rb") as file:
            self.word2idx = pickle.load(file)
            
        with open("../models/labels_dict.pickle","rb") as file:
            self.labels_dict = pickle.load(file)
            
    def preprocess(self,text):
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)

        latin_pattern = "[A-z0-9]"

        punct_pattern = "["+punctuation+"،ـ#؛]"
        noise = re.compile(""" ّ    | # Tashdid
                                َ    | # Fatha
                                ً    | # Tanwin Fath
                                ُ    | # Damma
                                ٌ    | # Tanwin Damm
                                ِ    | # Kasra
                                ٍ    | # Tanwin Kasr
                                ْ    | # Sukun
                                ـ     # Tatwil/Kashida
                            """, re.VERBOSE)
        
        patterns = [emoji_pattern,latin_pattern,punct_pattern,noise]
        

        text = re.sub("[إأٱآا]", "ا", text)
        text = re.sub("ى", "ي", text)
        text = re.sub("ؤ", "ء", text)
        text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)
        text = re.compile(r'([\u0621-\u064A])\1{1,}').sub(r'\1',text)
        text = re.sub("_"," ",text)
        text = re.sub("[^\u0621-\u064A^\s]"," ",text)
        text = re.sub("\s{2,}"," ",text)
        
        for pattern in patterns:
            text = re.sub(pattern,"",text)
        
        text = text.strip()
        
        return text
    
    def predict_nn(self,text):
        text = self.preprocess(text)
        text = [self.word2idx[w] for w in text.split()]
        if(len(text) < 4):
            return "Text must be longer than 4 words"

        prediction = self.nn_model.predict([text])
        
        idx = np.argmax(prediction)

        return self.labels_dict[idx]
    
    def predict_cnb(self,text: str):
        text = self.preprocess(text)
        if(len(text.split()) < 4):
            return "Text must be 4 words or longer"
        prediction = self.cnb_model.predict([text])
        return self.labels_dict[prediction[0]]