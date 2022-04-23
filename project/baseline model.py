import re, json, string, numpy as np, pandas as pd, nltk, tensorflow as tf, tensorflow.keras.backend as K, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download('stopwords')

with open('abbreviations.json', 'r', encoding='utf8') as fp:
	abbreviations = json.load(fp)


def convert_abbrev(word):
	return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word


def string_handle(text):
	crap = re.compile(r'RT|@[^\s]*|[^\s]*\…')
	text = crap.sub(r'', text)
	url = re.compile(r'https?://\S+|www\.\S+')
	text = url.sub(r'', text)  # remove urls
	html = re.compile(r'<.*?>')
	text = html.sub(r'', text)  # remove html tags
	emoji_pattern = re.compile(
		'['
		'\U0001F1E0-\U0001F1FF'  # flags (iOS)
		'\U0001F300-\U0001F5FF'  # symbols & pictographs
		'\U0001F600-\U0001F64F'  # emoticons
		'\U0001F680-\U0001F6FF'  # transport & map symbols
		'\U0001F700-\U0001F77F'  # alchemical symbols
		'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
		'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
		'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
		'\U0001FA00-\U0001FA6F'  # Chess Symbols
		'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
		'\U00002702-\U000027B0'  # Dingbats
		'\U000024C2-\U0001F251'
		']+'
	)
	text = emoji_pattern.sub(r'', text)  # remove emojis
	table = str.maketrans('', '', string.punctuation)
	text = text.translate(table)  # remove punctuations
	english_stops = set(stopwords.words('english'))
	text = [word for word in text.split() if word not in english_stops]  # remove stopwords
	text = [convert_abbrev(word) for word in text]  # stretch abbr
	return ' '.join(text)


# load and split dataset
max_sequence = 512  # input sequence length
embedding_dim = 128  # word vector size
target_map = {'nothing': 0, 'Lead': 1, 'Position': 2, 'Claim': 3, 'Evidence': 4, 'Counterclaim': 5, 'Rebuttal': 6, 'Concluding Statement': 7}
data = pd.read_csv('dataset', usecols=['discourse_text', 'discourse_type'])
data['discourse_text'] = data['discourse_text'].apply(lambda x: string_handle(x.lower()))
token = Tokenizer()
token.fit_on_texts(data['discourse_text'])
max_num_words = len(token.word_index) + 1  # dictionary size
texts = token.texts_to_sequences(data['discourse_text'])
texts = pad_sequences(texts, maxlen=max_sequence, padding='post', truncating='post')
data['discourse_type'] = data['discourse_type'].map(target_map)  # encode label as numerical
target = np.zeros((len(texts), max_sequence), dtype='int32')
for index in tqdm(range(len(texts))):
	words_len = np.count_nonzero(texts[index])
	target[index, :words_len] = data.loc[index, 'discourse_type']


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


# modeling
def build_model():
	input_layer = tf.keras.layers.Input((max_sequence,))
	embedding = tf.keras.layers.Embedding(max_num_words, embedding_dim, input_length=max_sequence)(input_layer)
	embedding = tf.keras.layers.LSTM(64, return_sequences=True)(embedding)
	sequence_output = tf.keras.layers.LSTM(32, return_sequences=True)(embedding)
	output_layer = tf.keras.layers.Dense(len(target_map), activation='softmax')(sequence_output)
	lstm = tf.keras.Model(inputs=input_layer, outputs=output_layer, name='baseline')
	lstm.compile(optimizer='Adamax', loss='sparse_categorical_crossentropy', metrics=[macro_f1])
	return lstm


epoch = 10
model = build_model()
model.summary()
train_history = model.fit(texts, target, validation_split=.2, epochs=epoch)
# plot metric
epoch = train_history.epoch
train_metric = train_history.history['macro_f1']
val_metric = train_history.history['val_macro_f1']
plt.title('train vs validation')
plt.xlabel('epoch')
plt.ylabel('macro f1')
plt.plot(epoch, train_metric, color='b', label='train')
plt.plot(epoch, val_metric, color='r', label='validation')
plt.legend()
plt.show()
