from sklearn import metrics
import sys
sys.path.append('/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/tools')
sys.path.append('/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification')
from tools_benchmark import  compute_bias_metrics_for_model, calculate_overall_auc,get_final_metric
from benchmark import get_jigsaw_score
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import gridspec

def evaluate(y_public_pred, y_private_pred , test_public_df, test_private_df, PATH, MODEL_NAME):

  IDENTITY_COLUMNS  = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
  ] 

  TOXICITY_COLUMN = 'toxicity'

  y_public_test = test_public_df['toxicity'] >= .5 
  y_public_test = y_public_test.values.astype(int).reshape((-1,1))

  y_private_test = test_private_df['toxicity'] >= .5 
  y_private_test = y_private_test.values.astype(int).reshape((-1,1))
    
  # save predictions 
  with open('/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/'+ PATH +'/y_public_pred.npy', 'wb') as f:
    np.save(f, y_public_pred)
  with open('/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/'+ PATH +'/y_private_pred.npy', 'wb') as f:
    np.save(f, y_private_pred)
  # evaluate the model on public and private test sets
  
  public_acc = metrics.accuracy_score(y_public_test,np.where(y_public_pred >= .5 , 1,0))
  print('Accuracy on public test: {:f}'.format(public_acc) )
  public_prec = metrics.precision_score(y_public_test,np.where(y_public_pred >= .5 , 1,0), average='weighted')
  public_rec = metrics.recall_score(y_public_test,np.where(y_public_pred >= .5 , 1,0))
  public_f1 = metrics.f1_score(y_public_test,np.where(y_public_pred >= .5 , 1,0), average='weighted')
  
  private_acc = metrics.accuracy_score(y_private_test,np.where(y_private_pred >= .5 , 1,0))
  print('Accuracy on private test: {:f}'.format(private_acc) )
  private_prec = metrics.precision_score(y_private_test,np.where(y_private_pred >= .5 , 1,0), average='weighted')
  private_rec = metrics.recall_score(y_private_test,np.where(y_private_pred >= .5 , 1,0))
  private_f1 = metrics.f1_score(y_private_test,np.where(y_private_pred >= .5 , 1,0), average='weighted')
  
  bias_metrics_df,public_auc_score = get_jigsaw_score(y_public_pred,test_public_df)
  print('Public AUC score : {:f}'.format(public_auc_score))
  bias_metrics_df.to_csv(r'/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/'+ PATH +'/public_bias_metrics.csv')

 
  bias_metrics_df,private_auc_score = get_jigsaw_score(y_private_pred,test_private_df)
  print('Private AUC score : {:f}'.format(private_auc_score))
  bias_metrics_df.to_csv(r'/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/' + 
                                PATH +'/private_bias_metrics.csv')

  stats = {'Acc. public': public_acc , 
          'Prec. public': public_prec,
          'Rec. public': public_rec,
          'F1 public': public_f1,
          'Public AUC score': public_auc_score,
          'Acc. private': private_acc , 
          'Prec. private': private_prec,
          'Rec. private': private_rec,
          'F1 private': private_f1,
          'Private AUC score': private_auc_score}
  report_df = pd.DataFrame([stats])
  report_df = report_df.round(4)
  report_df.to_csv(r'/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/' + PATH +'/report.csv')


def plot_history_for_accuracy_and_loss(histories,PATH):
  fig = plt.figure(figsize=(15, 5))
  outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.5)
  for i in range(len(histories)):
    
    history = histories[i] 
    inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=outer[i], wspace=0.4, hspace=0.6)
    
    # summarize history for accuracy
    ax1 = plt.Subplot(fig, inner[0])
    ax1.plot(history.history['target_accuracy'])
    ax1.plot(history.history['val_target_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set(xlabel='epoch', ylabel='accuracy')
    ax1.legend(['train', 'val'], loc='upper left')
    fig.add_subplot(ax1)
    # summarize history for loss
    ax2 = plt.Subplot(fig, inner[1])
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax1.set(xlabel='epoch', ylabel='loss')
    ax2.legend(['train', 'val'], loc='upper left')
    fig.add_subplot(ax2)

  fig.savefig('/content/drive/My Drive/Jigsaw Unintended Bias in Toxicity Classification/models/' + PATH +'/report.png')  
  fig.show()