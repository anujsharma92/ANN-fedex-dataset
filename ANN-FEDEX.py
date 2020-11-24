from tensorflow.keras.models import Sequential
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import keras
#loadind data set
fed=pd.read_csv("C:\\Users\\Anuj Kumar\\Desktop\\data science\\data set\\Ai\\module 6 (ANN)\\fedex.csv")

fed.info() #info 

fed=fed.dropna() # droppping na value as its a bigger dataset
fed.columns

#checking  correlation

corr1=pd.DataFrame(fed.iloc[:,:-1].corr())

#here correlation of year is least removing year column and categorical feature as it cannot be consider as i need to check for selectkBEst

fed_new=fed.drop(['Year','Source','Destination','Carrier_Name'],axis=1)

#output is categorical in tw0 categories so we are checking best feature using Selecet Baise

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# separating x and y

x=fed_new.iloc[:,0:11]
y=fed_new['Delivery_Status']

#taking absolute value as we cannot i/p the negative value for Select K best

x['Shipment_Delay']=x['Shipment_Delay'].abs()
x['Planned_Delivery_Time']=x['Planned_Delivery_Time'].abs()
x['Planned_TimeofTravel']=x['Planned_TimeofTravel'].abs()

# Rank fo all Features

ordered_rank_features=SelectKBest(score_func=chi2,k=11)
ordered_feature=ordered_rank_features.fit(x,y)
dfscores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
dfcolumns=pd.DataFrame(x.columns)
features_rank=pd.concat([dfcolumns,dfscores],axis=1)    
features_rank.columns=['Features','Score']
features_rank
#top 8 features 
features_rank.nlargest(5,'Score')
 #               Features         Score
#8          Shipment_Delay  1.094900e+08
#3    Actual_Shipment_Time  3.672496e+07
#4   Planned_Shipment_Time  2.025423e+07
#5   Planned_Delivery_Time  1.714078e+07
#10        Delivery_Status  2.804073e+06

#checking for highy correlated feature 
threshold=0.8
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

correlation(fed_new.iloc[:,:],threshold)
#{'Distance', 'Planned_Shipment_Time'} highly correlated features
#month and day of month is not palying important role while day of week is playing importrole in our inference
#removing planned shipment of time as both planned shipment of time and distance plays same role
#here i am removing  all unwanted feature from original data set 



fed_updated=fed.drop(['Year','Month','DayofMonth','Source','Destination','Carrier_Name','Planned_Shipment_Time'],axis=1)

#Creating dummy for day of week

day_dummy=pd.get_dummies(fed_updated['DayOfWeek'])

#renaming all dummyies column

day_dummy.columns=["DayOfWeek"+str(i) for i in range(0, 7)]

#dropping Day of Week

fed_updated=fed_updated.drop(['DayOfWeek'],axis=1)

#creating frame fo cocating dummy and data set

frames=[fed_updated,day_dummy]

#concating fed_updated and day dummy
fed_final=pd.concat(frames,axis=1)
#checking info 

fed_final.info()
fed_final.head()
fed_final.tail()

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(fed_final)
df_norm.describe()

#separating i/p and output

x=df_norm.iloc[:,[0,1,2,3,4,5,7,8,9,10,11,12,13]]
y=df_norm['Delivery_Status']

#converting into float

y=np.asarray(y).astype('float32')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=10)

x.shape[1]

def design_mlp():
    model=Sequential()
    model.add(Dense(x.shape[1],activation='relu',input_dim=x.shape[1]))
    model.add(Dense(150,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    return model
    #compile model
model= design_mlp()
a=model.compile(optimizer='adam',loss="binary_crossentropy",metrics=["accuracy"])

#now splitting training into new train and validation data set
x_train_new, x_test_val, y_train_new ,y_test_val = train_test_split(x_train,y_train,test_size=0.2,random_state=10)

model = model.fit(x_train_new,y_train_new,epochs=20,batch_size=50000,validation_data=(x_test_val, y_test_val))

history_dict = model.history
history_dict.keys()

#Plotting validation scores
#graph between  Loss and validation loss
import matplotlib.pyplot as plt
acc = model.history['accuracy']
val_acc = model.history['val_accuracy']
loss = model.history['loss']
val_loss = model.history['val_loss']
epochs= range(1,len(acc)+1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
#here we are getting minimum  loss in 20th epoch

#graph between accuracy and validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
#here we are getting maximum accuracy in 20th epoch

# final model

model=Sequential()
model.add(Dense(x.shape[1],activation='relu',input_dim=x.shape[1]))
model.add(Dense(150,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=50000)
#evaluating model

test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
# test_mae_score 0.983335554599762
#test_mse_score 0.05623653903603554

#prediction
predict=model.predict(x_test)
error=np.mean(y_test)-np.mean(predict)
error #0.0006636152292971986
