'''
Created on 2018. 12. 8.

@author: hong
'''
''''
과적합 조기 종료시키기
'''
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

np.random.seed(3)

# 1. 데이터셋 준비하기

# 훈련셋과 시험셋 로딩
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print("길이 : " , len(X_train))
print("길이 : " , len(X_train[59999:]))
print("값 : " , (X_train[59999]))