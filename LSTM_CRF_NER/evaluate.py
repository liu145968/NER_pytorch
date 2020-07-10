from LSTM_CRF.LSTM_CRF import LSTM_CRF
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch,pickle,time
from Config import X_test_OR_path,y_test_OR_path,word2id_path,tag2id_path,model_storage_path,DEVICE


def evaluate(tag_seq,pred_seq,tag2id,print_detailed=False):
    print("Compute Confusion Matrix...")
    confusion_matrix = np.zeros((len(tag2id), len(tag2id)), dtype=float)
    for yt, yp in zip(tag_seq, pred_seq):
        confusion_matrix[yt][yp] += 1
    score_dict = {}
    EPS = 1e-8
    tags_sum = 0
    print("evaluate...")
    for i in tqdm(range(len(confusion_matrix))):
        amount_true = np.sum(confusion_matrix[i, :])
        amount_pred = np.sum(confusion_matrix[:, i])
        precision = confusion_matrix[i][i] / (amount_pred + EPS)
        recall = confusion_matrix[i][i] / (amount_true + EPS)
        F1 = (2 * precision * recall) / (precision + recall + EPS)
        score_dict[list(tag2id.keys())[i]] = [precision, recall, F1, amount_true]
        tags_sum += amount_true

    tempP, tempR, tempF1 = 0, 0, 0
    for k,v in score_dict.items():
        tempP += v[0] * v[3]
        tempR += v[1] * v[3]
        tempF1 += v[2] * v[3]
    wei_averageP = tempP / tags_sum
    wei_averageeOP = (tempP - (score_dict["O"][0] * score_dict["O"][3])) / (tags_sum - score_dict["O"][3])
    wei_averageR = tempR / tags_sum
    wei_averageeOR = (tempR - (score_dict["O"][1] * score_dict["O"][3])) / (tags_sum - score_dict["O"][3])
    wei_averageF1 = tempF1 / tags_sum
    wei_averageeOF1 = (tempF1 - (score_dict["O"][2] * score_dict["O"][3])) / (tags_sum - score_dict["O"][3])

    average_dict = {"weighted avg": [wei_averageP, wei_averageR, wei_averageF1, tags_sum],
                    "weighted avg(excluding O)": [wei_averageeOP, wei_averageeOR, wei_averageeOF1, tags_sum - score_dict["O"][3]]}

    if print_detailed:
        res_pd=pd.DataFrame(np.zeros((len(score_dict)+2,4)))
        res_pd.columns=["Precision", "Recall", "F1", "Amount"]
        res_pd.index=list(score_dict.keys())+list(average_dict.keys())
        for k,v in score_dict.items():
            res_pd.loc[k, :] = v
        for k,v in average_dict.items():
            res_pd.loc[k, :] = v
        print(res_pd)

    return wei_averageeOF1


def predict(model,xtest,device):
    model.eval()
    model=model.to(device)
    y_pred=[]
    with torch.no_grad():
        print("predict...")
        for sents in tqdm(xtest):
            X=torch.tensor(sents).reshape(1,-1).to(device)
            y_pred.extend(model.predict(X))
    return y_pred


def pickleload(path):
    with open(path,"rb") as f:
        data=pickle.load(f)
    return data

X_test=pickleload(X_test_OR_path)
y_test=pickleload(y_test_OR_path)
word2id=pickleload(word2id_path)
tag2id=pickleload(tag2id_path)

y_true=[]
for i in y_test:
    y_true.extend(i)

model = torch.load(model_storage_path)

st=time.time()
y_pred=predict(model,X_test,device=DEVICE)
predict_time=time.time()-st
F1=evaluate(y_true,y_pred,tag2id,print_detailed=True)
evaluate_time=time.time()-st-predict_time
print("\npredict time: {:.3f}s, evaluate time: {:.3f}s, F1 score: {}".format(predict_time,evaluate_time,F1))
