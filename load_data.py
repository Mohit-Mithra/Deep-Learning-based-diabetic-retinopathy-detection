import pandas as pd

def load_data():
	train_df = pd.read_csv('train.csv')
	test_df = pd.read_csv('test.csv')
	#print(train_df.shape)
	#print(test_df.shape)
	train_df.head()

	#train_df['diagnosis'].hist()
	#train_df['diagnosis'].value_counts()

	return train_df, test_df
