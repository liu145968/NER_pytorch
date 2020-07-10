import numpy as np
import math,random,pickle

# Load Parameters
from Config import train_data_path,test_data_path,X_test_OR_path,y_test_OR_path,BATCH_SIZE,\
    word2id_path,tag2id_path,X_train_path,X_test_path,y_train_path,y_test_path,X_train_len_path,X_test_len_path,\
    USE_pretrained_vector,word_vector_path,EMBED_SIZE,PAD

def gen_dictanddata(traindatapath,testdatapath,PAD,word_start_id=1,tag_start_id=0):
    word2id,tag2id={},{}
    word_i=word_start_id
    tag_i=tag_start_id
    Xtrain,ytrain=[],[]
    Xtemp,ytemp=[],[]
    word2id[PAD]=0
    for line in open(traindatapath,encoding="utf-8"):
        if line!="\n":
            line=line.strip("\n").split("\t")
            Xtemp.append(line[0])
            ytemp.append(line[1])
            if line[0] not in word2id:
                word2id[line[0]]=word_i
                word_i+=1
            if line[1] not in tag2id:
                tag2id[line[1]]=tag_i
                tag_i+=1
        else:
            if Xtemp:
                Xtrain.append(Xtemp)
                ytrain.append(ytemp)
                Xtemp, ytemp = [], []

    Xtest,ytest=[],[]
    Xtemp,ytemp=[],[]
    for line in open(testdatapath,encoding="utf-8"):
        if line!="\n":
            line=line.strip("\n").split("\t")
            Xtemp.append(line[0])
            ytemp.append(line[1])
            if line[0] not in word2id:
                word2id[line[0]]=word_i
                word_i+=1
            if line[1] not in tag2id:
                tag2id[line[1]]=tag_i
                tag_i+=1
        else:
            if Xtemp:
                Xtest.append(Xtemp)
                ytest.append(ytemp)
                Xtemp, ytemp = [], []

    X_train=[[word2id[w] for w in sent] for sent in Xtrain]
    X_test=[[word2id[w] for w in sent] for sent in Xtest]
    y_train=[[tag2id[t] for t in sent] for sent in ytrain]
    y_test=[[tag2id[t] for t in sent] for sent in ytest]
    return word2id,tag2id,X_train,X_test,y_train,y_test

def creat_train_data(X,y,batch_size,tag_O_id,shuffle=True):
    lengths=[len(sen) for sen in X]
    len_sorted,sorted_id=np.sort(lengths),np.argsort(lengths)
    X,y=np.array(X),np.array(y)
    X_sorted,y_sorted=X[sorted_id],y[sorted_id]
    X_out,y_out,X_batch_len=[],[],[]
    for i in range(math.ceil(len(lengths)/batch_size)):
        start=i*batch_size
        end=start+batch_size if start+batch_size<=len(lengths) else len(lengths)
        X_temp,y_temp=X_sorted[start:end],y_sorted[start:end]
        max_len=max(len_sorted[start:end])
        X_t_array,y_t_array=np.zeros((end-start,max_len),dtype=int),np.ones((end-start,max_len),dtype=int)*tag_O_id
        for i in range(len(X_t_array)):
            X_t_array[i,0:len(X_temp[i])]=X_temp[i]
            y_t_array[i,0:len(y_temp[i])]=y_temp[i]
        X_out.append(X_t_array)
        y_out.append(y_t_array)
        X_batch_len.append(len_sorted[start:end])
    if shuffle:
        random.seed(10)
        random.shuffle(X_out)
        random.seed(10)
        random.shuffle(y_out)
        random.seed(10)
        random.shuffle(X_batch_len)
    return X_out,y_out,X_batch_len

def read_save_wordvector(vector_load_path,embed_size,word2id):
    word_vector=np.random.rand(len(word2id.keys()),embed_size)
    wid_vec_dic={}
    for line in open(vector_load_path,encoding="utf-8"):
        if line[0]==" ":
            v=line[2:].strip("\n").split(" ")
            w=" "
        else:
            line=line.strip("\n").split(" ")
            w,v=line[0],line[1:]
        if w not in word2id:
            continue
        v=list(map(float,v))
        wid_vec_dic[word2id[w]]=v
    for i in range(word_vector.shape[0]):
        if i not in wid_vec_dic:
            continue
        word_vector[i,:]=np.array(wid_vec_dic[i],dtype=float)

    pickledump(word_vector,vector_load_path.split(".")[0]+"_temp.pickle")

def pickledump(data,path):
    with open(path,"wb") as f:
        pickle.dump(data,f)

word2id,tag2id,Xtrain,Xtest,ytrain,ytest=gen_dictanddata(train_data_path,test_data_path,PAD)

pickledump(Xtest,X_test_OR_path)
pickledump(ytest,y_test_OR_path)

if USE_pretrained_vector:
    read_save_wordvector(vector_load_path=word_vector_path,embed_size=EMBED_SIZE,word2id=word2id)

X_train,y_train,X_trian_len=creat_train_data(X=Xtrain,y=ytrain,batch_size=BATCH_SIZE,tag_O_id=tag2id["O"],shuffle=True)
X_test,y_test,X_test_len=creat_train_data(X=Xtest,y=ytest,batch_size=BATCH_SIZE,tag_O_id=tag2id["O"],shuffle=True)


data_ls=[word2id,tag2id,X_train,X_test,y_train,y_test,X_trian_len,X_test_len]
data_path_ls=[word2id_path,tag2id_path,X_train_path,X_test_path,y_train_path,y_test_path,X_train_len_path,X_test_len_path]
for data,path in zip(data_ls,data_path_ls):
    pickledump(data,path)