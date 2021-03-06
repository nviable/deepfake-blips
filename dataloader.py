#%%
from __future__ import print_function, division
import os, pickle, math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# class that holds all the data
# then go to the dataloader

#%%
class BlipDatasetLoader:
    '''
    Returns 
    # batch Xv(batch_size, frames, 384, 512, 3)
    # batch Xa(batch_size, frames, 1025, 2)
    # batch Y(batch_size, frames, 1)
    '''
    def __init__(self, batch_size=16, frames=10, test=0.25, val=0.25, only_vid=False):
        super().__init__()
        self.batch_size = batch_size
        self.frames = frames
        p = shuffle(list(Path('./data/picklejar').glob('*.pickle')))
        tr_size = math.floor(len(p) * (1 - (test + val)))
        te_size = tr_size + math.floor(len(p) * test)
        self.train = p[:tr_size]
        self.test = p[tr_size:te_size]
        self.val = p[te_size:]
        self.only_vid = only_vid
        print("="*10)
        print("Data split into {} | {} | {}".format( len(self.train), len(self.test), len(self.val) ))
        print("="*10)

    def gen(self, train=True, val=False, no_timesteps=False):
        t_gen = self.get_timewindow(train, val, no_timesteps)
        while True:
            batch_Xa, batch_Xv, batch_Y = [],[], []

            while len(batch_Xa) < self.batch_size:
                Xv, Xa, Y = next(t_gen)
                batch_Xa.append(Xa)
                batch_Xv.append(Xv)
                batch_Y.append(Y)
            # print('spitting out: {} | {} | {}'.format(np.shape(batch_Xv), np.shape(batch_Xa), np.shape(batch_Y)))
            if self.only_vid:
                yield np.array(batch_Xv), np.array(batch_Y)
            else:
                yield [np.array(batch_Xv), np.array(batch_Xa)], np.array(batch_Y)
            # yield np.array(batch_Xv), np.array(batch_Y)
            # yield ({'vid': np.array(batch_Xv), 'aud': np.array(batch_Xa)}, np.array(batch_Y))
            # batch Xv(batch_size, frames, 384, 512, 3)
            # batch Xa(batch_size, frames, 1025, 2)
            # batch Y(batch_size, frames, 1)

    def get_timewindow(self, train=True, val=False,  no_timesteps=False):
        Xa_b, Xv_b, Y = [],[],[]
        data_pipe = self.train
        if not train:
            if not val:
                data_pipe = self.test
            else:
                data_pipe = self.val
        for f_name in data_pipe:
            f = open(f_name, 'rb')
            p = pickle.load(f)

            # print('Opened ', f_name)
            for s in p:
                if no_timesteps:
                    yield (np.array(s[0]), np.reshape(s[1], (25, 41, 2)), np.array([0,1] if s[2] else [1,0]))
                    # yield (np.array(s[0]), np.array(s[2]))
                else: 
                    Xv_b.append(s[0])
                    Xa_b.append(np.reshape(s[1], (25, 41, 2)))
                    Y.append(s[2])
                    if(len(Y) == self.frames):
                        yield (np.array(Xv_b), np.array(Xa_b), np.array([0,1] if s[2] else [1,0]))
                        Xa_b, Xv_b, Y = [],[],[]


#%%
'''

#########

def saniat_generator(folder_path, batch_size=64, time_window=20):
	batch_img_X = []
	batch_sound_X = []
	batch_Y = []

	for files in folder_path:
		batch_img_X.append(files[0])
		batch_sound_X.append(files[1])
		batch_Y.append()

		if(len(bat) == batch_size)
			yield(batch)

		batch = []

	raise Stop

##########



class WrappedDataLoader:
    def __init__(self, batch_size=16, frames=10):
        super().__init__()
        self.batch_size = batch_size
        self.frames = frames
        self.pkl_files = shuffle(list(Path('./data/picklejar').glob('*.pickle')))

    def get_batch(self):
        X_img = []
        X_aud = []
        Y = []
        i = 0

        while(True):
            print('adding element {} to batch'.format(i))
            f = open(str(self.pkl_files[i]), 'rb')
            data = pickle.load(f)
            data = data[:math.floor(len(data)/self.frames)*self.frames] #cutoff at multiples of self.frames
            if(i == len(self.pkl_files)):
                i=0
            i+=1

            for j in range(self.batch_size):
                
                x_v = TensorDataset(data[:][:][0], data[:][:][2])
                x_a = TensorDataset(data[:][:][1], data[:][:][2])

                
                X_img.append(np.shape(np.moveaxis(x_img, 2, 0)))
                X_aud.append(x_aud)
                Y.append(y)    
                print(np.shape(X_img))
                yield X_img, X_aud, Y


# pickle_in = open("data/picklejar/sx374-video-mstk0.pickle","rb")
# example_dict = pickle.load(pickle_in)
'''
