import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Loader:
    def __init__(self):
        self.buys = pd.read_csv("yoochoose-data/yoochoose-buys.dat", header=None, low_memory=False)
        self.clicks = pd.read_csv("yoochoose-data/yoochoose-clicks.dat", header=None, low_memory=False)
        # self.test = pd.read_csv("yoochoose-data/yoochoose-test.dat", header=None, low_memory=False)
        self.sample = 1

        self.preprocess_data()
        self.sampling()

    def preprocess_data(self):
        self.buys.columns = ['session_id', 'timestamp', 'item_id', 'price', 'quantity']
        self.clicks.columns = ['session_id', 'timestamp', 'item_id', 'category']
        # self.test.columns = ['session_id', 'timestamp', 'item_id', 'category']

        item_encoder = LabelEncoder()
        self.clicks['item_id'] = item_encoder.fit_transform(self.clicks.item_id)

    def sampling(self):
        if self.sample == 1:
            sampled_session_id = np.random.choice(self.clicks.session_id.unique(), 1000000, replace=False)
            self.clicks = self.clicks.loc[self.clicks.session_id.isin(sampled_session_id)]
            self.clicks['label'] = self.clicks.session_id.isin(self.buys.session_id)




