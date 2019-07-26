#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path, argparse, subprocess, random, numpy as np, pandas as pd, tensorflow as tf
from keras.backend import tensorflow_backend
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, PReLU, BatchNormalization, Dropout, Input, Layer
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras import regularizers, backend as K
import load_ffri2018_D
mal_api = None


class CustomVariationalLayer(Layer):
    def custom_rmse_loss(self, x):
        global mal_api
        return K.sqrt(K.mean(K.sum(K.square((x + 1.0) / 2.0 - mal_api), axis=1))) * 0.05
    def call(self, inputs):
        x = inputs
        loss = self.custom_rmse_loss(x)
        self.add_loss(loss, inputs=inputs)
        return x
    
    
def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1) *0.95

    
def set_trainable(model, trainable=False):
    model.trainable = trainable
    try:
        layers = model.layers
    except:
        return
    for layer in layers:
        set_trainable(layer, trainable)

        
def generator_model(api_dim):
    reg = regularizers.l2(0.00001)
    rand_init = RandomNormal(stddev=0.02)
    input_layer = Input((api_dim, ))
    x = Dense(512, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(input_layer)
    x = PReLU()(x)
    x = Dense(256, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    x = Dense(api_dim, activation='tanh', kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = CustomVariationalLayer()(x) 
    return Model(input_layer, x, name='generator')
    

def discriminator_model(api_dim):
    reg = regularizers.l2(0.00001)
    rand_init = RandomNormal(stddev=0.02) 
    input_layer = Input((api_dim, ))
    x = Dense(64, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(input_layer)
    x = PReLU()(x)
    x = Dense(128, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dense(256, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dense(512, activation=None, kernel_initializer=rand_init, kernel_regularizer=reg, bias_regularizer=reg)(x)
    x = BatchNormalization(axis=-1)(x)
    x = PReLU()(x)
    x = Dense(1, activation='sigmoid', kernel_initializer=RandomNormal(stddev=0.02), kernel_regularizer=reg, bias_regularizer=reg)(x)
    return Model(input_layer, x, name='discriminator')


def GAN(api_dim):
    d = discriminator_model(api_dim)
    d_opt = Adam(lr=1e-5, beta_1=0.1)
    d.compile(optimizer=d_opt, loss='binary_crossentropy', metrics=['acc'])
    
    g = generator_model(api_dim)
    g_opt = Adam(lr=2e-4, beta_1=0.5)
    g.compile(optimizer=g_opt, loss=binary_crossentropy, metrics=['acc'])
    
    set_trainable(d, False)
    d_on_g = Sequential([g, d])
    d_on_g.compile(optimizer=g_opt, loss=binary_crossentropy, metrics=['acc'])
    return g, d, d_on_g


def train(c_folder, m_file):
    print('Make api_list...')
    folder_list = [c_folder]
    api_list_area = '555*.txt'
    api_list = load_ffri2018_D.make_api_list(folder_list, api_list_area)
    
    print('Loading the dataset...')
    malware_api = load_ffri2018_D.make_used_api_dataframe_with_malware_file(m_file, api_list)
    #cleanware_api = load_ffri2018_D.make_used_api_dataframe(folder_list, ['5*.txt'], malware_api.columns)
    cleanware_api = load_ffri2018_D.make_used_api_dataframe(folder_list, [api_list_area], malware_api.columns)
    X_malware = malware_api.drop('label', axis=1)
    y_malware = malware_api['label']
    X_cleanware = cleanware_api.drop('label', axis=1)
    y_cleanware = cleanware_api['label']
    X_malware_ndarray = X_malware.values
    X_cleanware_ndarray = X_cleanware.values 
    
    print('Save api list...'+'discriminator_api_list_'+api_list_area)
    with open('discriminator_api_list_'+api_list_area, 'w') as f:
        for api in malware_api.columns:
            f.write(api+'\n')
           
    print('Model definitions...')
    api_dim = len(X_malware.columns)
    global mal_api
    mal_api = np.asarray(X_malware_ndarray[0], dtype=np.float32)
    g, d, d_on_g = GAN(api_dim)
   
    print('Training...')
    BATCH_SIZE = 26   # a, b, c, ... , z 
    MAX_EPOCH = 26   # a, b, c, ... , z 
    CREATE_FILE_NUM = 200
    for epoch in range(MAX_EPOCH):
        for batch in range(BATCH_SIZE):
            # Generate
            Z = np.asarray(np.random.randint(-1, 1, (CREATE_FILE_NUM, api_dim)), dtype=np.float32)
            X_gen = (g.predict(Z) + 1.0) / 2.0
            for file_num in range(CREATE_FILE_NUM):
                for i in range(api_dim):
                    X_gen[file_num][i] = int(np.round(X_gen[file_num][i])) | X_malware_ndarray[0][i]
                    
            # Save generated data
            save_gen_data = 'gen/'+chr(epoch+97)   # a, b, c, ... , z 
            if not os.path.exists(save_gen_data):
                os.mkdir(save_gen_data)
            save_gen_data += '/'+chr(batch+97)   # a, b, c, ... , z 
            if not os.path.exists(save_gen_data):
                os.mkdir(save_gen_data)
            for file_num in range(CREATE_FILE_NUM):
                with open(save_gen_data+'/gen_data_'+str(file_num).zfill(2)+'.txt', 'w') as f:
                    for i, api in enumerate(X_malware.columns):
                        if X_gen[file_num][i] == 1:
                            f.write(api+' Hint\n')
                            
            # Labeling by blackboxD
            cmd = 'python blackboxDiscriminator.py --mode predict --algo MLP --folder '+save_gen_data
            cmd_result = subprocess.check_output(cmd.split()).decode('utf-8')
            print(cmd_result)
            
            y_gen = []
            randm = random.randrange(len(X_cleanware_ndarray))
            X_gen_clean = X_cleanware_ndarray[randm:randm+1]
            y_gen_clean = [0]
            
            for i, line in enumerate(cmd_result.split('\n')):
                if 'is cleanware' in line:
                    y_gen.append(0)
                    X_gen_clean = np.concatenate([X_gen_clean, [X_gen[i]]], axis=0)
                    y_gen_clean.append(0)
                elif 'is malware' in line:
                    y_gen.append(1)
            X_gen = np.concatenate([X_gen, X_malware_ndarray], axis=0)
            y_gen.append(1)
            
            # Save result
            sum_gen_clean = 0
            for i in range(len(X_gen_clean)-1):
                sum_gen_clean += np.sum(X_gen_clean[i+1] == 1)
            with open('discriminator_result_'+api_list_area, 'a') as f:
                f.write(str(26*epoch + batch)+' '+str(len(y_gen_clean)-1)+' '+str(sum_gen_clean/len(X_gen_clean)-1)+'\n')
                           
            # Training
            gen_loss = 0
            gen_acc = 0
            dis_loss = 0
            dis_acc = 0
            repet_num = 10
            for i in range(repet_num):
                # Discriminator training
                set_trainable(d, True)
                dis_loss_now, dis_acc_now = d.train_on_batch(X_gen * 2.0 - 1.0, np.array(y_gen))
                set_trainable(d, False)
                dis_loss += dis_loss_now
                dis_acc  += dis_acc_now

                # Generator training
                gen_loss_now, gen_acc_now = d_on_g.train_on_batch(X_gen_clean * 2.0 - 1.0, np.array(y_gen_clean))
                gen_loss += gen_loss_now
                gen_acc  += gen_acc_now
                print('gen_loss = {0:.3f}, dis_loss = {1:.3f}, gen_acc = {2:.3f}, dis_acc = {3:.3f}'
                      .format(gen_loss_now, dis_loss_now, gen_acc_now, dis_acc_now))
                
            # Loss and accuracy
            print('Generator training loss:         {0}'.format(gen_loss / repet_num))
            print('Discriminator training loss:     {0}'.format(dis_loss / repet_num))
            print('Generator training accuracy:     {0}'.format(gen_acc / repet_num))
            print('Discriminator training accuracy: {0}'.format(dis_acc / repet_num))

            # Save weight
            g.save_weights('generator.h5')
            d.save_weights('discriminator.h5')
            
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_folder", type=str, help="cleanware folder_path")
    parser.add_argument("--m_file", type=str, help="malware file_path")
    args = parser.parse_args()
    return args
               
    
if __name__ == '__main__':
    # Set random seeds
    np.random.seed(2)
    tf.set_random_seed(2)
    
    args = get_args()
    train(args.c_folder, args.m_file)
    
    