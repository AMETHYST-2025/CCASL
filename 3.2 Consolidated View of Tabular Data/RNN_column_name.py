import string
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import nlpaug.augmenter.char as nac
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
aug_ocr = nac.OcrAug()
aug_del = nac.RandomCharAug(action="delete")

torch.manual_seed(42)
# define a character vocabulary depending on the use case, I ll use all the possible printable characters here.
character_vocabulary = [char for char in string.printable]
print(f"Character Vocabulary:\n{character_vocabulary}\n")
print(string.printable)
print(len(character_vocabulary))

df = pd.read_excel('/Users/tchagoue/Documents/AMETHYST/Springer_paper/3.2 Consolidated View of Tabular Data/Data/RNN_Prediction/Norm_column_name.xlsx', index_col=0)
df = df.reset_index()
print(df)

Char_Voc = [' ']
for i in range(len(df)):
    for j in df['Avant_Norm'][i]:
        if j not in Char_Voc :
            Char_Voc.append(j)
    for j in str(df['Apres_Norm'][i]):
        if j not in Char_Voc :
            Char_Voc.append(j)
for j in character_vocabulary:
    if j not in Char_Voc :
        Char_Voc.append(j)

print(Char_Voc)
print(len(Char_Voc), 'herehygkubju')
# ------------------------------------One hot encoding--------------------------------
char_to_idx_map = {char: idx for idx, char in enumerate(Char_Voc)}
print(f"Character to Index Mapping:\n{char_to_idx_map}\n")
ohe_characters = torch.eye(n=len(Char_Voc))  # using the eye method for identity matrix
print(f"One hot encoded characters:\n{ohe_characters}\n")
# Printing the one hot encoded representations of the digit '1'
ohe_repr_a = ohe_characters[char_to_idx_map['1']]
#print(f"One hot encoded representation of 1:\n{ohe_repr_a}\n")
def PRINT(mat):
    value = ''
    for j in range(len(mat[0])):
        for k in range(len(mat)):
            if mat[k][j] == 1:
                value+=Char_Voc[k]
                break
    return value
# ------------------------------------------------------------------------------------
def augment_ocr_data(df):
    for j in range(len(df)):
        Variations = []
        text = df['Avant_Norm'][j]
        texts_ocr = aug_ocr.augment(text,n=2)
        texts_del = aug_del.augment(text)
        for ele in texts_del:
            if ele != text :  Variations.append(ele)
        for ele in texts_ocr:
            if ele not in Variations: Variations.append(ele)
        for var in Variations:
            new_Row = [df['References'][j],var,df['Apres_Norm'][j],df['Modifications'][j],df['Frequences'][j],df['Unnamed: 6'][j]]
            df.loc[len(df)] = new_Row
    return df
"""
df = augment_ocr_data(df)
df = shuffle(df)
df.to_excel('df_augment.xlsx', index=True)
df.reset_index(drop=True, inplace=True)
"""
# ----------------------------Column name to One hot encoding-------------------------
# Convert the word into character indices
min_length = 3
max_length = 50
def encode(value, max_length = max_length):
    word_char_idx_tensor = torch.tensor([char_to_idx_map[char] for char in value]) # ne pointe que pour les caracteres du "word"
    #print(f"Indexed Representation of the word '{value}': {word_char_idx_tensor}")
    # converting the indexed word to one hot representation
    word_ohe_repr = ohe_characters[word_char_idx_tensor].T # multiplier pas la matrice identité
    #print(f"One Hot Encoded Representation: \n{word_ohe_repr}")
    #print(PRINT(word_ohe_repr))
    complete = max_length-len(word_ohe_repr[0])
    if complete > 0 :
        padd = torch.zeros([len(word_ohe_repr), complete], dtype=torch.int32)
        tensor = torch.cat([torch.tensor(word_ohe_repr), padd], dim=1)
    if complete == 0 :
        tensor = word_ohe_repr
    if complete < 0 :
        print('------------------')
        print(PRINT(word_ohe_repr))
    return tensor

DF = pd.DataFrame(columns=['ref','input','target','Frequences'])
for i in range(len(df)):
    if len(df.Avant_Norm[i]) <= max_length and len(df.Avant_Norm[i]) >= min_length and len(df['Apres_Norm'][i])<=max_length:
        new_Row = [df['References'][i], encode(str(df['Avant_Norm'][i])).detach().cpu().numpy(), encode(str(df['Apres_Norm'][i])).detach().cpu().numpy(),
                   df['Frequences'][i]]
        DF.loc[len(DF)] = new_Row
    else :
        print(df.Avant_Norm[i])

def Data_Augmentation(DF):
    # Dupliquer les plus frequents
    for j in range(len(DF)):
        if df['Frequences'][j] > 1:
            for ele in range(DF['Frequences'][j]):
                DF.loc[len(DF)] = DF.loc[j].copy()
        else :
            break
    DF = shuffle(DF)
    DF.reset_index(drop=True, inplace=True)
    return  DF
print(DF)
DF = Data_Augmentation(DF)
print(DF)
print(PRINT(DF.target[3]))
#---------------------------------------------------------------


X_train, X_test, y_train, y_test = train_test_split(DF['input'][:], DF['target'][:], train_size=0.7, shuffle=True,random_state = 42)
# Convert to 2D PyTorch tensors
print(len(X_test),len(X_train),len(y_train),len(y_test))
X_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)
featuresTrain = torch.Tensor(X_train) #torch.from_numpy
print('-------------')
targetsTrain = torch.Tensor(y_train)#.type(torch.LongTensor)
print('-------------')
print(X_test)
featuresTest = torch.Tensor(X_test)
print('-------------')
targetsTest = torch.Tensor(y_test)#.type(torch.LongTensor)

# Pytorch train and test sets
train = TensorDataset(featuresTrain,targetsTrain)
test = TensorDataset(featuresTest,targetsTest)

# batch_size, epoch and iteration
batch_size = 20
n_iters = 5000
num_epochs = n_iters / (len(featuresTrain) / batch_size)
num_epochs = int(num_epochs)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)


# Create RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        # Number of hidden layers
        self.layer_dim = layer_dim
        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.activation = nn.ReLU()  # Softmax(dim=None), ReLU()
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))

        # One time step
        out, hn = self.rnn(x, h0)
        out = self.activation(out)
        out = self.fc(out) #out[:, -1, :]
        #print(out)
        return out

# Create RNN
input_dim = 50  # input dimension
hidden_dim = 116  # hidden layer dimension
layer_dim = 1  # number of hidden layers
output_dim = 50  # output dimension

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# Cross Entropy Loss
error = nn.CrossEntropyLoss() #MSELoss() #L1Loss()  #CrossEntropyLoss()
# SGD Optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 116
loss_list = []
test_loss_list = []
iteration_list = []
accuracy_list = []
index_accuracy_list = []
count = 0

def decode(value):
    val = ''
    for ele in value:
        val += Char_Voc[ele]
    return val
print(num_epochs)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)
        # Clear gradients
        optimizer.zero_grad()
        # Forward propagation
        outputs = model(train)
        # Calculate softmax and ross entropy loss
        loss = error(outputs, labels)
        # Calculating gradients
        loss.backward()
        # Update parameters
        optimizer.step()
        count += 1
        if count % 500 == 0:
            # Calculate Accuracy
            correct = 0; index_correct =0
            total = 0; index_total =0
            # Iterate through test dataset
            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim))
                outputs = model(images)
                with torch.no_grad(): test_loss = error(outputs, images)
                predicted = torch.max(outputs.data, 1)[1]
                labels = torch.max(labels.data, 1)[1]
                #----------------------------------------------------
                for vec, vec_pred in zip(labels, predicted):
                    stop=[]
                    for i in range(len(vec)):
                        q = vec[len(vec)-i-1]
                        if q != 0:
                            break
                    if torch.equal(vec[:len(vec)-i],vec_pred[:len(vec)-i]) == True :
                        print('---------------------------------')
                        print(vec)
                        print(decode(vec[:len(vec)-i]))
                        print(decode(vec_pred[:len(vec)-i]))
                        correct += 1
                        print('---------------------------------')
                    for a,b in zip(vec[:len(vec)-i],vec_pred[:len(vec)-i]):
                        if a==b: index_correct+=1
                    index_total+=len(vec[:len(vec)-i])
                #-----------------------------------------------------
                # Total number of labels
                total += labels.size(0)
            accuracy = 100 * correct / float(total)
            index_accuracy = 100 * index_correct / float(index_total)
            # store loss and iteration
            loss_list.append(loss.data)
            test_loss_list.append(test_loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            index_accuracy_list.append(index_accuracy)
            if count % 500 == 0:
                # Print Loss
                print('Iteration: {}  Train_Loss: {} Test_Loss: {}  Test_Accuracy: {} %  Test_Accuracy_index: {}'.format(count, loss.data, test_loss.data , accuracy, index_accuracy))

def prediction():
    DF_ = pd.DataFrame(columns=['Input','Prediction','Label'])
    for images, labels in test_loader:
        images = Variable(images.view(-1, seq_dim, input_dim))
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1]
        labels = torch.max(labels.data, 1)[1]
        #----------------------------------------------------
        for vec, vec_pred, vec_input in zip(labels, predicted, images):
            for i in range(len(vec)):
                q = vec[len(vec)-i-1]
                if q != 0:
                    break
            new_Row = [PRINT(vec_input), decode(vec_pred[:len(vec) - i]), decode(vec[:len(vec) - i])]
            DF_.loc[len(DF_)] = new_Row
        DF_.to_excel('/Users/tchagoue/Documents/AMETHYST/Springer_paper/3.2 Consolidated View of Tabular Data/Data/RNN_Prediction/Noms_Colonnes_Norm_Prediction.xlsx', index=True)
prediction()

plt.plot(iteration_list,loss_list, label='Train_loss',color='red')
plt.plot(iteration_list,test_loss_list,color = "blue",label='Test_loss')
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of iteration")
plt.show()

# visualization accuracy
plt.plot(iteration_list,index_accuracy_list,color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of iteration")
plt.savefig('graph.png')
plt.show()
