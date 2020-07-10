import torch

train_data_path= "data/traindata.txt"
test_data_path= "data/testdata.txt"

BATCH_SIZE=64
EPOCH=10
LR=0.01
EMBED_SIZE=100
HIDDEN_SIZE=150

USE_pretrained_vector=False
word_vector_path="data/wordvec100d.txt"

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE=torch.device("cpu")

model_storage_path= "model/model_LSTM_CRF.pth"



PAD="<pad>"
word2id_path= "data/word2id_dict.pickle"
tag2id_path= "data/tag2id_dict.pickle"

X_train_path= "data/X_train_ls.pickle"
X_test_path= "data/X_test_ls.pickle"
y_train_path= "data/y_train_ls.pickle"
y_test_path= "data/y_test_ls.pickle"

X_test_OR_path= "data/X_test_OR_ls.pickle"
y_test_OR_path= "data/y_test_OR_ls.pickle"

X_train_len_path= "data/X_train_len_ls.pickle"
X_test_len_path= "data/X_test_len_ls.pickle"


