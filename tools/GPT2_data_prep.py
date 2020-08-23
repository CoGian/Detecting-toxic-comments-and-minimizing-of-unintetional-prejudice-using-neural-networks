import pandas as pd
from transformers import *
from sklearn.model_selection import train_test_split
import numpy as np

MAX_LEN = 180

"""# Load Datasets"""

IDENTITY_COLUMNS = [
  'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
  'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]
TARGET_COLUMN = 'target'
TOXICITY_COLUMN = 'toxicity'
AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

train_df = pd.read_csv("train_cleared.csv")
# train_df = train_df[:1000]
test_public_df = pd.read_csv("test_public_cleared.csv")
# test_public_df = test_public_df.loc[:, ['toxicity', 'comment_text'] + IDENTITY_COLUMNS].dropna()[:500]
test_private_df = pd.read_csv("test_private_cleared.csv")
# test_private_df = test_private_df.loc[:, ['toxicity', 'comment_text'] + IDENTITY_COLUMNS].dropna()[:500]

# split
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=13, shuffle=True)

y_train = train_df[TARGET_COLUMN].values.reshape((-1, 1))
y_train_aux = train_df[AUX_COLUMNS].values

y_val = val_df[TARGET_COLUMN].values.reshape((-1, 1))
y_val_aux = val_df[AUX_COLUMNS].values

y_public_test = test_public_df[TOXICITY_COLUMN].values.reshape((-1, 1))

y_private_test = test_private_df[TOXICITY_COLUMN].values.reshape((-1, 1))

# add weights
train_df[IDENTITY_COLUMNS + [TARGET_COLUMN]] = np.where(train_df[IDENTITY_COLUMNS + [TARGET_COLUMN]] >= 0.5, True,
                                                        False)
test_public_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] = np.where(
  test_public_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] >= 0.5, True, False)
test_private_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] = np.where(
  test_private_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] >= 0.5, True, False)
val_df[IDENTITY_COLUMNS + [TARGET_COLUMN]] = np.where(val_df[IDENTITY_COLUMNS + [TARGET_COLUMN]] >= 0.5, True, False)

sample_weights_train = np.ones(len(train_df), dtype=np.float32)
sample_weights_val = np.ones(len(val_df), dtype=np.float32)

# Add 1 weight for each mention of Identity Subgroup (Subgroup AUC) ('e.g. any mention of ethnicX)
sample_weights_train += train_df[IDENTITY_COLUMNS].sum(axis=1)
sample_weights_val += val_df[IDENTITY_COLUMNS].sum(axis=1)

# Add 2 weight for Toxic comment but not mentioning Identity (BPSN) (e.g. you really suck!!)
sample_weights_train += (train_df[TARGET_COLUMN] >= 0.5) * (~train_df[IDENTITY_COLUMNS]).sum(axis=1).floordiv(
  len(IDENTITY_COLUMNS)) * 2
sample_weights_val += (val_df[TARGET_COLUMN] >= 0.5) * (~val_df[IDENTITY_COLUMNS]).sum(axis=1).floordiv(
  len(IDENTITY_COLUMNS)) * 2

# Add 10 weight for non-Toxic tag that mentions an Identity (e.g. you are ethnicX and it's fine !!)
sample_weights_train += (train_df[TARGET_COLUMN] < 0.5) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
sample_weights_val += (val_df[TARGET_COLUMN] < 0.5) * val_df[IDENTITY_COLUMNS].sum(axis=1) * 5

# scale weights
sample_weights_train = sample_weights_train / sample_weights_train.mean()
sample_weights_val = sample_weights_val / sample_weights_val.mean()

"""# Get datasets"""

gpt2_tokenizer_transformer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_tokenizer_transformer.pad_token = '<PAD>'


def encode_examples(df, PATH, sample_weights=None, labels=None, labels_aux=None, forTest=False):
  # prepare list, so that we can build up final TensorFlow dataset from slices.
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  
  for comment in df['comment_text'].astype(str).values:
    gpt2_input = gpt2_tokenizer_transformer.encode_plus(comment,
                                                        add_special_tokens=True,  # add [CLS], [SEP]
                                                        max_length=MAX_LEN,
                                                        # max length of the text 
                                                        pad_to_max_length=True,  # add [PAD] tokens
                                                        return_attention_mask=True
                                                        # add attention mask to not focus on pad tokens
                                                        )
    
    input_ids_list.append(gpt2_input['input_ids'], dtype=np.int32)
    token_type_ids_list.append(gpt2_input['token_type_ids'])
    attention_mask_list.append(gpt2_input['attention_mask'])
  
  with open(PATH + '/input_ids.npy', 'wb') as filehandle:
    # store the data as binary data stream
    np.save(filehandle, np.asarray(input_ids_list))
  with open(PATH + '/segment_ids.npy', 'wb') as filehandle:
    # store the data as binary data stream
    np.save(filehandle, np.asarray(token_type_ids_list))
  with open(PATH + '/input_mask.npy', 'wb') as filehandle:
    # store the data as binary data stream
    np.save(filehandle, np.asarray(attention_mask_list))
  
  if not forTest:
    with open(PATH + '/labels.npy', 'wb') as filehandle:
      # store the data as binary data stream
      np.save(filehandle, np.asarray(labels))
    with open(PATH + '/labels_aux.npy', 'wb') as filehandle:
      # store the data as binary data stream
      np.save(filehandle, np.asarray(labels_aux))
    with open(PATH + '/sample_weights.npy', 'wb') as filehandle:
      # store the data as binary data stream
      np.save(filehandle, np.asarray(sample_weights))


print("Start preprocess..")
encode_examples(train_df, '/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/gpt2/data/train', sample_weights_train, y_train, y_train_aux)
print("Finished train data..")
encode_examples(val_df, '/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/gpt2/data/val', sample_weights_val, y_val, y_val_aux)
print("Finished val data..")
encode_examples(test_private_df, '/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/gpt2/data/test_private', forTest=True)
print("Finished test private data..")
encode_examples(test_public_df, '/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/gpt2/data/test_public', forTest=True)
print("Finished test public data..")
