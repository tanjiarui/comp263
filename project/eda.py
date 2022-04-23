import numpy as np, pandas as pd, matplotlib.pyplot as plt, nltk
from tqdm.auto import tqdm
from glob import glob
from nltk.corpus import stopwords
nltk.download('stopwords')

data = pd.read_csv('train.csv', dtype={'discourse_id': int, 'discourse_start': int, 'discourse_end': int})
data['discourse_len'] = data['discourse_text'].apply(lambda x: len(x.split()))
data['pred_len'] = data['predictionstring'].apply(lambda x: len(x.split()))
print('number of mislabeled predictionstring: %d' % data.query('discourse_len != pred_len').shape[0])

figure = plt.figure(figsize=(12, 8))
plt.subplots_adjust(hspace=0.5)
# average sentence length of each discourse type
figure.add_subplot(211)
ax1 = data.groupby('discourse_type')['discourse_len'].mean().sort_values().plot(kind='barh')
ax1.set_title('average number of words versus discourse type', fontsize=14, fontweight='bold')
ax1.set_xlabel('average number of words', fontsize=10)
ax1.set_ylabel('')

# frequency of discourse type
figure.add_subplot(212)
ax2 = data['discourse_type'].value_counts().sort_values().plot(kind='barh')
ax2.set_title('frequency of discourse type in all essays', fontsize=14, fontweight='bold')
ax2.set_xlabel('frequency', fontsize=10)
ax2.set_ylabel('')
plt.show()

# frequency of discourse type by its relative position
figure = plt.figure(figsize=(12, 8))
type_num = data['discourse_type_num'].value_counts(ascending=True).rename_axis('discourse_type_num').reset_index(name='count').set_index('discourse_type_num')
type_num['percentage'] = round((type_num['count'] / data.id.nunique()), 3)
ax = type_num.query('percentage > .04')['percentage'].plot(kind='barh')
ax.set_title('discourse type percentage present in essays', fontsize=20, fontweight='bold')
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_xlabel('percent')
ax.set_ylabel('')
plt.show()

# the relative position of discourse type in an essay in average
data['discourse_part'], counter = 1, 1
for i in tqdm(data.index[1:]):
	if data.loc[i, 'id'] == data.loc[i - 1, 'id']:
		counter += 1
		data.loc[i, 'discourse_part'] = counter
	else:
		counter = 1
		data.loc[i, 'discourse_part'] = counter
print('the average position of discourse type')
print(data.groupby('discourse_type')['discourse_part'].mean().sort_values())
'''
the average position of discourse type
discourse_type
Lead                    1.004514
Position                2.201116
Claim                   5.163480
Evidence                6.627478
Counterclaim            7.304968
Rebuttal                8.684344
Concluding Statement    9.644650
'''

# count text length and words
len_dict, word_dict = dict(), dict()
train_txt = glob('./train/*.txt')
for file in tqdm(train_txt):
	with open(file, 'r') as txt_file:
		discourse_id = file.split('/')[-1].replace('.txt', '')
		text = txt_file.read()
		text_len = len(text.strip())
		words = len(text.split())
		len_dict[discourse_id] = text_len
		word_dict[discourse_id] = words
data['essay_len'] = data['id'].map(len_dict)
data['essay_words'] = data['id'].map(word_dict)

# gap between two discourses within an essay. it implies some nonsense sentences in that essay
data['gap_length'] = np.nan
# set the first one, check out the dataset
data.loc[0, 'gap_length'] = 7  # discourse start - 1 (previous end is always -1). \n is counted
for i in tqdm(data.index[1:]):
	# gap if difference is not 1 within an essay
	if (data.loc[i, 'id'] == data.loc[i - 1, 'id']) and (data.loc[i, 'discourse_start'] - data.loc[i - 1, 'discourse_end'] > 1):
		data.loc[i, 'gap_length'] = data.loc[i, 'discourse_start'] - data.loc[i - 1, 'discourse_end'] - 2  # minus 2 as the previous end is always -1 and the previous start always +1
	# gap if the first discourse of a new essay does not start at 0. \n is counted
	elif (data.loc[i, 'id'] != data.loc[i - 1, 'id']) and (data.loc[i, 'discourse_start'] != 0):
		data.loc[i, 'gap_length'] = data.loc[i, 'discourse_start'] - 1
# any text after the last discourse of an essay
last_ones = data.drop_duplicates(subset='id', keep='last')
last_ones.loc[:, 'gap_end_length'] = np.where(last_ones.discourse_end < last_ones.essay_len, last_ones.essay_len - last_ones.discourse_end, np.nan)
cols_to_merge = ['id', 'discourse_id', 'gap_end_length']
data = data.merge(last_ones[cols_to_merge], on=['id', 'discourse_id'], how='left')
all_gaps = (data.gap_length[data.gap_length.notna()]).append(data.gap_end_length[data.gap_end_length.notna()], ignore_index=True)
all_gaps = all_gaps[all_gaps < 300]
figure = plt.figure(figsize=(12, 6))
all_gaps.plot.hist(bins=100)
plt.title('histogram of gap length (gaps up to 300 characters only)')
plt.xticks(rotation=0)
plt.xlabel('length of gaps in characters')
plt.show()

# find bad essays mostly composed by unclassified discourses
total_gaps = data.groupby('id').agg({'essay_len': 'first', 'gap_length': 'sum', 'gap_end_length': 'sum'}).reset_index()
total_gaps.loc[:, 'unclassified_percentage'] = round(((total_gaps.gap_length + total_gaps.gap_end_length) / total_gaps.essay_len),2)
total_gaps.to_csv('unclassified percentage', index=False)

data['discourse_text'] = data['discourse_text'].str.lower()
redundancy = stopwords.words('english')
redundancy.extend(['school', 'students', 'people', 'would', 'could', 'many'])  # meaningless words
# put dataframe of top-10 words in dict for all discourse types
counts_dict = dict()
for discourse_type in data['discourse_type'].unique():
	df = data.query('discourse_type == @discourse_type')
	text = df.discourse_text.apply(lambda x: x.split()).tolist()
	text = [item for elem in text for item in elem]
	count = pd.Series(text).value_counts().to_frame().reset_index()
	count.columns = ['word', 'frequency']
	count = count[~count.word.isin(redundancy)].head(10)
	count = count.set_index('word').sort_values(by='frequency', ascending=True)
	counts_dict[discourse_type] = count
plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.5)
keys = list(counts_dict.keys())
for n, key in enumerate(keys):
	ax = plt.subplot(4, 2, n + 1)
	ax.set_title(f'most used words in {key}')
	counts_dict[keys[n]].plot(ax=ax, kind='barh')
	plt.ylabel('')
plt.show()

data.to_csv('data', index=False)