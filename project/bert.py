import numpy as np, pandas as pd, tensorflow_hub as hub, tensorflow_text as text, tensorflow as tf, tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

epoch = 2
max_sequence = 512
target_map = {'nothing': 0, 'Lead': 1, 'Position': 2, 'Claim': 3, 'Evidence': 4, 'Counterclaim': 5, 'Rebuttal': 6, 'Concluding Statement': 7}
data = pd.read_csv('dataset', usecols=['discourse_text', 'discourse_type'])
data['discourse_type'] = data['discourse_type'].map(target_map)
target = np.zeros((len(data), max_sequence), dtype='int32')
for index in tqdm(data.index):
	words_len = len(data.loc[index, 'discourse_text'].split())
	target[index, :words_len] = data.loc[index, 'discourse_type']
x_train, x_test, y_train, y_test = train_test_split(data['discourse_text'], target, test_size=.2)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(128)


def macro_f1(y_true, y_pred, beta=1, threshold=.5):
	y_true = K.cast(y_true, 'float')
	y_pred = tf.argmax(y_pred, -1)
	y_pred = K.cast(K.greater(K.cast(y_pred, 'float'), threshold), 'float')

	tp = K.sum(y_true * y_pred, axis=0)
	fp = K.sum((1 - y_true) * y_pred, axis=0)
	fn = K.sum(y_true * (1 - y_pred), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = (1 + beta ** 2) * p * r / ((beta ** 2) * p + r + K.epsilon())
	f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
	return K.mean(f1)


preprocessor = hub.load('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
tokenize = hub.KerasLayer(preprocessor.tokenize)
bert_pack_inputs = hub.KerasLayer(preprocessor.bert_pack_inputs, arguments=dict(seq_length=max_sequence))
encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/2', trainable=True)


def build_model(bert_layer):
	text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
	tokenized_input = [tokenize(text_input)]
	encoder_inputs = bert_pack_inputs(tokenized_input)
	sequence_output = bert_layer(encoder_inputs)['sequence_output']
	out = tf.keras.layers.Dense(len(target_map), activation='softmax')(sequence_output)

	bert = tf.keras.Model(inputs=text_input, outputs=out)
	bert.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=[macro_f1])
	return bert


model = build_model(encoder)
# checkpoint = tf.keras.callbacks.ModelCheckpoint('checkpoint', monitor='val_loss', save_best_only=True)
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir='logs/{}'.format('bert'))
model.summary()
train_history = model.fit(train_dataset, validation_data=test_dataset, epochs=epoch)