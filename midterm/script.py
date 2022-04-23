import tensorflow as tf, matplotlib.pyplot as plt
from keras.preprocessing import sequence
from sklearn.metrics import classification_report

top_words = 25000
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=top_words)
review_length = 100
x_train = sequence.pad_sequences(x_train, maxlen=review_length)
x_test = sequence.pad_sequences(x_test, maxlen=review_length)


def build_model():
	input_layer = tf.keras.layers.Input((review_length,))
	embedding = tf.keras.layers.Embedding(top_words, 32, input_length=review_length)(input_layer)
	layer = tf.keras.layers.LSTM(100)(embedding)
	output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(layer)
	lstm = tf.keras.Model(inputs=input_layer, outputs=output_layer)
	lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	return lstm


model = build_model()
model.summary()
train_history = model.fit(x=x_train, y=y_train, validation_data=(x_test[:5000], y_test[:5000]), epochs=1)
# plot metric
epoch = train_history.epoch
train_accuracy = train_history.history['accuracy']
val_accuracy = train_history.history['val_accuracy']
plt.title('train vs validation')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.plot(epoch, train_accuracy, color='b', label='train')
plt.plot(epoch, val_accuracy, color='r', label='validation')
plt.legend()
plt.show()
# validation
prediction = [0 if p < .5 else 1 for p in model.predict(x_test)]
print(classification_report(y_test, prediction))