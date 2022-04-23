import pandas as pd, os
from tqdm.auto import tqdm

data = pd.read_csv('data').drop(columns=['discourse_len', 'pred_len', 'discourse_part', 'essay_len', 'essay_words'])
# drop bad essays. take .5 as the benchmark
bad_essay = pd.read_csv('unclassified percentage')
bad_essay = bad_essay.query('unclassified_percentage >= .5')
data.drop(data[data['id'].isin(bad_essay['id'])].index).reset_index(inplace=True)

# insert unclassified samples
discourse_id = 0
rows = list()
for i in tqdm(data.index[1:]):
	if data.loc[i, 'gap_length'] > 0:
		text_id = data.loc[i, 'id']
		path = os.path.join('./train', text_id + '.txt')
		file = open(path, 'r')
		text = file.read().lower()
		file.close()
		if data.loc[i - 1, 'id'] != data.loc[i, 'id']:
			start = 0  # as there is no i - 1 for first row
			end = data.loc[i, 'discourse_start'] - 1
			discourse_text = text[start:end + 1]
			prediction_start = 1
		else:
			start = data.loc[i - 1, 'discourse_end'] + 1
			end = data.loc[i, 'discourse_start'] - 1
			discourse_text = text[start - 1:end + 1]
			prediction_start = int(data.loc[i - 1, 'predictionstring'].split()[-1]) + 1
		discourse_type = 'nothing'
		prediction_end = int(data.loc[i, 'predictionstring'].split()[0])
		prediction_str = ' '.join([str(i) for i in range(prediction_start, prediction_end)])
		row = [text_id, discourse_id, start, end, discourse_text, discourse_type, prediction_str]
		discourse_id += 1
		rows.append(row)
rows = pd.DataFrame(rows, columns=['id', 'discourse_id', 'discourse_start', 'discourse_end', 'discourse_text', 'discourse_type', 'predictionstring'])
# handle column discourse_type_num
rows.insert(6, 'discourse_type_num', '')
rows.loc[0, 'discourse_type_num'] = 'nothing 1' if rows.loc[0, 'id'] != rows.loc[1, 'id'] else ''
discourse_part = None
for i in tqdm(rows.index[1:]):
	if rows.loc[i - 1, 'id'] != rows.loc[i, 'id']:
		discourse_part = 1
	rows.loc[i, 'discourse_type_num'] = 'nothing ' + str(discourse_part)
	discourse_part += 1
data = pd.concat([data, rows], ignore_index=True)
data.groupby('id').apply(lambda item: item.sort_values('discourse_start')).reset_index(drop=True, inplace=True)
data.to_csv('dataset', index=False)