from LSTM_CRF.LSTM_CRF import LSTM_CRF
from tqdm import tqdm
import pickle,time,torch
from Config import word2id_path,tag2id_path,X_train_path,y_train_path,X_test_path,y_test_path,X_train_len_path,X_test_len_path,\
    BATCH_SIZE,EMBED_SIZE,HIDDEN_SIZE,LR,EPOCH,DEVICE,USE_pretrained_vector,word_vector_path,model_storage_path


def get_mask(batchdata_len):
    return torch.cat([torch.tensor([1.0]*i+[0.0]*(batchdata_len[-1]-i)) for i in batchdata_len]).reshape(len(batchdata_len),-1)

def train(model,traindata,traindata_len,optimizer,epoch,device,print_train_time=False,save_epoch_model=False):
    model.train()
    model=model.to(device)

    data, tags=traindata[0],traindata[1]
    start_time=time.time()

    for e in range(epoch):
        print("\nThe {} of {} epoch...".format(e+1,epoch))
        curtrain_stime,curtrain_loss=time.time(),0
        for i in tqdm(range(len(data))):
            sents, target = torch.from_numpy(data[i]).long().to(device), torch.from_numpy(tags[i]).long().to(device)
            mask,lengths=get_mask(traindata_len[i]).to(device),torch.from_numpy(traindata_len[i]).to(device)

            loss=model.compute_loss(sents,target,lengths,mask).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            curtrain_loss+=loss.item()
        print("train time:{:.3f} s , loss:{}".format(time.time()-curtrain_stime,curtrain_loss))
        if save_epoch_model:
            torch.save(model, "model/model_epoch{}.pth".format(e+1))

    if print_train_time:
        print("all_train_time:{:.3f} s ".format(time.time()-start_time),"epoch:",epoch)


def load_data(path):
    with open(path,"rb") as f:
        data=pickle.load(f)
    return data

word2id=load_data(word2id_path)
tag2id=load_data(tag2id_path)

X_train=load_data(X_train_path)
y_train=load_data(y_train_path)

X_test=load_data(X_test_path)
y_test=load_data(y_test_path)

X_train_len=load_data(X_train_len_path)
X_test_len=load_data(X_test_len_path)

model=LSTM_CRF(vocab_size=len(word2id),
               embed_size=EMBED_SIZE,
               hidden_size=HIDDEN_SIZE,
               num_tags=len(tag2id))

if USE_pretrained_vector:
    model.embed.weight.data.copy_(torch.from_numpy(load_data(word_vector_path.split(".")[0]+"_temp.pickle")))

optimizer=torch.optim.Adam(model.parameters(),lr=LR)

train(model=model,traindata=[X_train,y_train],traindata_len=X_train_len,optimizer=optimizer,
      epoch=EPOCH,device=DEVICE,print_train_time=True)

# save complete model
torch.save(model,model_storage_path)
print("The model has been completely saved! The path is:",model_storage_path)

# Only save the parameters of model
# torch.save(model.state_dict(),model_storage_path)
# print("model is saved! the path is:",model_storage_path)