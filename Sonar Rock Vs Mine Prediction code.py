#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[43]:


data = pd.read_csv('sonar_data.csv',header=None)


# In[44]:


data.head()


# In[45]:


data.tail()


# In[46]:


data.shape


# In[47]:


print("Number of Rows",data.shape[0])
print("Number of Columns",data.shape[1])


# In[48]:


data.info()


# In[49]:


data.isnull().sum()


# In[50]:


data.describe()


# In[51]:


data.columns


# In[52]:


data.groupby(60).mean()


# In[53]:


X = data.drop(60,axis=1)


# In[54]:


y = data[60]


# In[55]:


y


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[57]:


X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# In[58]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[59]:


lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred1=lr.predict(X_test)


# In[60]:


accuracy_score(y_test,y_pred1)


# In[61]:


knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred2=knn.predict(X_test)


# In[62]:


accuracy_score(y_test,y_pred2)


# In[63]:


rf= RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred3=rf.predict(X_test)


# In[64]:


accuracy_score(y_test,y_pred3)


# In[65]:


from sklearn.linear_model import SGDClassifier


# In[66]:


sgd = SGDClassifier()


# In[67]:


for i in range(len(X_train)):
    sgd.partial_fit(X_train[i:i+1],y_train[i:i+1],classes=['R','M'])


# In[68]:


score=sgd.score(X_test,y_test)


# In[69]:


print("Acc:",score)


# In[70]:


final_data = pd.DataFrame({'Models':['LR','KNN','RF','SGD'],
             'ACC':[accuracy_score(y_test,y_pred1),
                   accuracy_score(y_test,y_pred2),
                   accuracy_score(y_test,y_pred3),
                   score]})


# In[71]:


final_data


# In[72]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler


# In[73]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[74]:


# Hyperparameter Tuning
search_space = {'n_neighbors':[3,5,7,9], 'weights': ['uniform', 'distance'], 'p': [1, 2]}
scorer = make_scorer(accuracy_score)
grid_search = GridSearchCV(KNeighborsClassifier(), search_space, scoring=scorer, cv=5)
grid_search.fit(X_train_scaled, y_train)


# In[75]:


knn = grid_search.best_estimator_
y_pred = knn.predict(X_test_scaled)


# In[76]:


accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)


# In[77]:


knn1=KNeighborsClassifier(n_neighbors=3)
knn1.fit(X,y)


# In[78]:


import joblib


# In[79]:


joblib.dump(knn1,'rock_mine_prediction_model')


# In[ ]:


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import pandas as pd
import joblib

# Define a function to load a CSV file and make predictions using the saved model
def predict_data():
    # Open a file dialog to select a CSV file
    file_path = filedialog.askopenfilename(title="Select prediction data file", filetypes=(("CSV files", "*.csv"),))

    # Load the data from the CSV file into a Pandas dataframe
    data = pd.read_csv(file_path,header=None)

    
    # Load the saved KNN model from a file
    knn_model = joblib.load('rock_mine')

    # Make predictions on the data using the model
    predictions = knn_model.predict(data)

    # Show a message box with the predicted labels
    print(predictions)
    if predictions == 'M':
        s = "Mine"
    else:
        s= "Rock"
    messagebox.showinfo(title="Predictions", message=str(s))
    

# Create a Tkinter window
window = tk.Tk()

# Set the size of the window
window.geometry("300x100")

# Set the background color of the window
window.configure(bg="light gray")

# Get the width and height of the screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# Calculate the position of the window to center it on the screen
x = (screen_width - window.winfo_reqwidth()) / 2
y = (screen_height - window.winfo_reqheight()) / 2

# Set the position of the window
window.geometry("+%d+%d" % (x, y))



# Add a button to the window to select a CSV file and make predictions
button = tk.Button(window, text="Select prediction data", command=predict_data)
button.pack()

button_width = 150
button_height = 30
button_x = (300 - button_width) / 2
button_y = (100 - button_height) / 2

# Set the position of the button
button.place(relx=button_x/300, rely=button_y/100, relwidth=button_width/300, relheight=button_height/100)

# Run the Tkinter event loop

# Run the Tkinter event loop
window.mainloop()


# In[ ]:


import os

# Get the current working directory
cwd = os.getcwd()

# Construct the full path to the file
file_path = os.path.join(cwd, 'rock_mine')

# Load the model from the file
knn_model = joblib.load(file_path)


# In[ ]:




