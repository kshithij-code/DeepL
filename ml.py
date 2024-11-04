import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from keras.layers import Dense,Input
from keras.models import Sequential
from keras.optimizers import SGD,Adam
import numpy as np
import matplotlib.pyplot as plt

x=np.random.rand(100,1)


y=2*x+0.0006



model=Sequential()
model.add(Input((1,)))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001),loss="mean_squared_error")
model.summary()

model.fit(x=x,y=y,epochs=500)
print(y_pred:=model.predict(x))

plt.scatter(x=x,y=y,label="original data")
plt.plot(x,y_pred,label="predicted data")
plt.show()
