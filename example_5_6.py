from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# create model
model = Sequential()
model.add(Dense(128, input_dim=784, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
#compile model
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


mnist = input_data.read_data_sets("/tf/data/mnist/", one_hot=True)
X = mnist.train.images
Y = mnist.train.labels
# Fit the model

model.fit(X, Y, epochs=1, batch_size=10)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
