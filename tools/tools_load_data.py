import numpy as np 
import pandas as pd
import tensorflow as tf 
import random
import pickle 
import gc
import os
from sklearn.model_selection import train_test_split


def get_datasets(train_df, val_df, test_public_df, test_private_df):
    
  BATCH_SIZE = 512
  MAX_LEN = 180 
  EPOCHS = 10

  TARGET_COLUMN = 'target'
  TOXICITY_COLUMN = 'toxicity'
  GENDER_IDENTITIES = ['male', 'female', 'transgender', 'other_gender']
  SEXUAL_ORIENTATION_IDENTITIES = ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
    'other_sexual_orientation']
  RELIGION_IDENTINTIES = ['christian', 'jewish', 'muslim', 'hindu', 'buddhist',
   'atheist', 'other_religion']
  RACE_IDENTINTIES = ['black', 'white', 'latino', 'asian',
   'other_race_or_ethnicity']
  DISABILITY_IDENTINTIES = ['physical_disability','intellectual_or_learning_disability',
                          'psychiatric_or_mental_illness', 'other_disability']
  IDENTITY_COLUMNS  = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
  ] 

  AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']

  GLOVE_PATH = "/content/drive/My Drive/Glove/glove.840B.300d.pkl"
  CRAWL_PATH = "/content/drive/My Drive/Crawl/crawl-300d-2M.pkl"

  x_train = train_df["comment_text"].astype(str)
  y_train = train_df[TARGET_COLUMN].values.reshape((-1,1))
  y_train_aux = train_df[AUX_COLUMNS].values
  
  x_val = val_df["comment_text"].astype(str)
  y_val = val_df[TARGET_COLUMN].values.reshape((-1,1))
  y_val_aux = val_df[AUX_COLUMNS].values

  x_public_test = test_public_df["comment_text"].astype(str)
  y_public_test = test_public_df[TOXICITY_COLUMN].values.reshape((-1,1))

  x_private_test = test_private_df["comment_text"].astype(str)
  y_private_test = test_private_df[TOXICITY_COLUMN].values.reshape((-1,1))

  # create and pad sequences 
  #create tokenizer for our data
  tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=False)
  tokenizer.fit_on_texts(list(x_train) + list(x_public_test) + list(x_private_test)  + list(x_val) )
  #convert text data to numerical indexes
  x_train = tokenizer.texts_to_sequences(x_train)
  x_val = tokenizer.texts_to_sequences(x_val)
  x_public_test = tokenizer.texts_to_sequences(x_public_test)
  x_private_test = tokenizer.texts_to_sequences(x_private_test)

  #pad data up to MAX_LEN (note that we truncate if there are more than MAX_LEN tokens)
  x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
  x_val = tf.keras.preprocessing.sequence.pad_sequences(x_val, maxlen=MAX_LEN)
  x_public_test = tf.keras.preprocessing.sequence.pad_sequences(x_public_test, maxlen=MAX_LEN)
  x_private_test = tf.keras.preprocessing.sequence.pad_sequences(x_private_test, maxlen=MAX_LEN)

  # add weights 
  train_df[IDENTITY_COLUMNS+[TARGET_COLUMN]] = np.where(train_df[IDENTITY_COLUMNS+[TARGET_COLUMN]] >= 0.5, True, False)
  test_public_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] = np.where(test_public_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] >= 0.5, True, False)
  test_private_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] = np.where(test_private_df[IDENTITY_COLUMNS + [TOXICITY_COLUMN]] >= 0.5, True, False)
  val_df[IDENTITY_COLUMNS+[TARGET_COLUMN]] = np.where(val_df[IDENTITY_COLUMNS+[TARGET_COLUMN]] >= 0.5, True, False)
  
  sample_weights_train = np.ones(len(train_df), dtype=np.float32)
  sample_weights_val = np.ones(len(val_df), dtype=np.float32)

  # Add 1 weight for each mention of Identity Subgroup (Subgroup AUC) ('e.g. any mention of ethnicX)
  sample_weights_train += train_df[IDENTITY_COLUMNS].sum(axis=1)
  sample_weights_val += val_df[IDENTITY_COLUMNS].sum(axis=1)

  #Add 2 weight for Toxic comment but not mentioning Identity (BPSN) (e.g. you really suck!!)
  sample_weights_train +=  (train_df[TARGET_COLUMN] >= 0.5) * (~train_df[IDENTITY_COLUMNS]).sum(axis=1).floordiv(len(IDENTITY_COLUMNS)) * 2 
  sample_weights_val +=  (val_df[TARGET_COLUMN] >= 0.5) * (~val_df[IDENTITY_COLUMNS]).sum(axis=1).floordiv(len(IDENTITY_COLUMNS)) * 2 

  #Add 10 weight for non-Toxic tag that mentions an Identity (e.g. you are ethnicX and it's fine !!)
  sample_weights_train += (train_df[TARGET_COLUMN]<0.5) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
  sample_weights_val += (val_df[TARGET_COLUMN]<0.5) * val_df[IDENTITY_COLUMNS].sum(axis=1) * 5

  #scale weights 
  sample_weights_train = sample_weights_train / sample_weights_train.mean()
  sample_weights_val = sample_weights_val / sample_weights_val.mean()


  AUTO = tf.data.experimental.AUTOTUNE

  # feed train set to tf.Dataset
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train,{"target": y_train, "aux": y_train_aux},sample_weights_train.values ))
  train_dataset = train_dataset.repeat().shuffle(len(train_df)).batch(BATCH_SIZE).prefetch(AUTO)

  # feed validation set to tf.Dataset
  validation_dataset = tf.data.Dataset.from_tensor_slices((x_val,{"target": y_val, "aux": y_val_aux},sample_weights_val.values ))
  validation_dataset = validation_dataset.batch(BATCH_SIZE).cache().prefetch(AUTO)

  # feed test sets set to tf.Dataset
  public_test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_public_test)
    .batch(BATCH_SIZE)
  )

  private_test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_private_test)
    .batch(BATCH_SIZE)
  )

  # load glove embeddings 
  glove_embedding_matrix,unknown_words = build_matrix(tokenizer.word_index,GLOVE_PATH)
  print('n unknown words (glove): ', len(unknown_words))
  print('n known words (glove): ', len(glove_embedding_matrix))

  # load crawl embeddings 
  crawl_embedding_matrix,unknown_words = build_matrix(tokenizer.word_index,CRAWL_PATH)
  print('n unknown words (crawl): ', len(unknown_words))
  print('n known words (crawl): ', len(crawl_embedding_matrix))

  # concanate them 
  embedding_matrix = np.concatenate([glove_embedding_matrix, crawl_embedding_matrix], axis=-1)
  
  # save memmory 
  del crawl_embedding_matrix
  del glove_embedding_matrix
  gc.collect()


  return train_dataset,validation_dataset, public_test_dataset, private_test_dataset , embedding_matrix


def load_embeddings(path):
  with open(path,'rb') as f:
    embedding_index = pickle.load(f)

  return embedding_index

def build_matrix(word_index, path):
  embedding_index = load_embeddings(path)
  embedding_matrix = np.zeros((len(word_index) + 1, 300))
  unknown_words = []
    
  for word, i in word_index.items():
    try:
      embedding_matrix[i] = embedding_index[word]
    except KeyError:
      unknown_words.append(word)
      
  return embedding_matrix, unknown_words   
