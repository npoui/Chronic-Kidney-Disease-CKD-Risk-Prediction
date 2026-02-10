
# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.124668Z","iopub.execute_input":"2026-01-12T11:53:53.125693Z","iopub.status.idle":"2026-01-12T11:53:53.142827Z","shell.execute_reply.started":"2026-01-12T11:53:53.125656Z","shell.execute_reply":"2026-01-12T11:53:53.141683Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        # linear algebra

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.144437Z","iopub.execute_input":"2026-01-12T11:53:53.145063Z","iopub.status.idle":"2026-01-12T11:53:53.182147Z","shell.execute_reply.started":"2026-01-12T11:53:53.145021Z","shell.execute_reply":"2026-01-12T11:53:53.181117Z"}}
kidney =pd.read_csv("/kaggle/input/drugpatient-dataset-for-ckd-prediction/CKD_NephrotoxicDrug_Dataset.csv")
kidney.head()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.183749Z","iopub.execute_input":"2026-01-12T11:53:53.184079Z","iopub.status.idle":"2026-01-12T11:53:53.191489Z","shell.execute_reply.started":"2026-01-12T11:53:53.184050Z","shell.execute_reply":"2026-01-12T11:53:53.190469Z"}}
kidney['ckd_risk_label'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.193026Z","iopub.execute_input":"2026-01-12T11:53:53.193660Z","iopub.status.idle":"2026-01-12T11:53:53.214450Z","shell.execute_reply.started":"2026-01-12T11:53:53.193616Z","shell.execute_reply":"2026-01-12T11:53:53.213423Z"}}
kidney.isnull().sum()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.216661Z","iopub.execute_input":"2026-01-12T11:53:53.216948Z","iopub.status.idle":"2026-01-12T11:53:53.238345Z","shell.execute_reply.started":"2026-01-12T11:53:53.216921Z","shell.execute_reply":"2026-01-12T11:53:53.237328Z"}}
kidney.duplicated().sum()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.239481Z","iopub.execute_input":"2026-01-12T11:53:53.239926Z","iopub.status.idle":"2026-01-12T11:53:53.255502Z","shell.execute_reply.started":"2026-01-12T11:53:53.239897Z","shell.execute_reply":"2026-01-12T11:53:53.254620Z"}}
kidney.info()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.256756Z","iopub.execute_input":"2026-01-12T11:53:53.257072Z","iopub.status.idle":"2026-01-12T11:53:53.305662Z","shell.execute_reply.started":"2026-01-12T11:53:53.257033Z","shell.execute_reply":"2026-01-12T11:53:53.304694Z"}}
kidney_numeric=kidney.select_dtypes(include='number')
kidney_numeric.corr()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.306952Z","iopub.execute_input":"2026-01-12T11:53:53.307322Z","iopub.status.idle":"2026-01-12T11:53:53.377375Z","shell.execute_reply.started":"2026-01-12T11:53:53.307284Z","shell.execute_reply":"2026-01-12T11:53:53.376511Z"}}
kidney.describe()

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.378455Z","iopub.execute_input":"2026-01-12T11:53:53.378883Z","iopub.status.idle":"2026-01-12T11:53:53.382625Z","shell.execute_reply.started":"2026-01-12T11:53:53.378853Z","shell.execute_reply":"2026-01-12T11:53:53.381776Z"}}
#kidney["ckd_risk_label"]=kidney['ckd_risk_label'].map({"1":1,"2":2,"0":0})

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.385222Z","iopub.execute_input":"2026-01-12T11:53:53.385529Z","iopub.status.idle":"2026-01-12T11:53:53.403095Z","shell.execute_reply.started":"2026-01-12T11:53:53.385493Z","shell.execute_reply":"2026-01-12T11:53:53.402003Z"}}
X=kidney.drop('ckd_risk_label',axis=1)
Y=kidney['ckd_risk_label']
X.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.404414Z","iopub.execute_input":"2026-01-12T11:53:53.404736Z","iopub.status.idle":"2026-01-12T11:53:53.422190Z","shell.execute_reply.started":"2026-01-12T11:53:53.404705Z","shell.execute_reply":"2026-01-12T11:53:53.421279Z"}}
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
X_train.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.423235Z","iopub.execute_input":"2026-01-12T11:53:53.423593Z","iopub.status.idle":"2026-01-12T11:53:53.436485Z","shell.execute_reply.started":"2026-01-12T11:53:53.423548Z","shell.execute_reply":"2026-01-12T11:53:53.435620Z"}}
X_test.shape

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.438060Z","iopub.execute_input":"2026-01-12T11:53:53.438400Z","iopub.status.idle":"2026-01-12T11:53:53.457418Z","shell.execute_reply.started":"2026-01-12T11:53:53.438370Z","shell.execute_reply":"2026-01-12T11:53:53.456521Z"}}
X_train.dtypes

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.458718Z","iopub.execute_input":"2026-01-12T11:53:53.459017Z","iopub.status.idle":"2026-01-12T11:53:53.478475Z","shell.execute_reply.started":"2026-01-12T11:53:53.458991Z","shell.execute_reply":"2026-01-12T11:53:53.477352Z"}}
X_train = pd.get_dummies(X_train, drop_first=True)
X_test  = pd.get_dummies(X_test, drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.479608Z","iopub.execute_input":"2026-01-12T11:53:53.479887Z","iopub.status.idle":"2026-01-12T11:53:53.496751Z","shell.execute_reply.started":"2026-01-12T11:53:53.479860Z","shell.execute_reply":"2026-01-12T11:53:53.495879Z"}}
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:53.498026Z","iopub.execute_input":"2026-01-12T11:53:53.498356Z","iopub.status.idle":"2026-01-12T11:53:54.690972Z","shell.execute_reply.started":"2026-01-12T11:53:53.498315Z","shell.execute_reply":"2026-01-12T11:53:54.690089Z"}}
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200,max_depth=None,random_state=42)
rf.fit(X_train, Y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)


# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:54.692937Z","iopub.execute_input":"2026-01-12T11:53:54.693216Z","iopub.status.idle":"2026-01-12T11:53:54.722699Z","shell.execute_reply.started":"2026-01-12T11:53:54.693191Z","shell.execute_reply":"2026-01-12T11:53:54.721682Z"}}

Y_pred= rf.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:55:05.754816Z","iopub.execute_input":"2026-01-12T11:55:05.755616Z","iopub.status.idle":"2026-01-12T11:55:05.808556Z","shell.execute_reply.started":"2026-01-12T11:55:05.755545Z","shell.execute_reply":"2026-01-12T11:55:05.807501Z"}}
from sklearn.metrics import accuracy_score
rf_pred  = rf.predict(X_test)
knn_pred = knn.predict(X_test)

print("RF Accuracy:", accuracy_score(Y_test, rf_pred))
print("KNN Accuracy:", accuracy_score(Y_test, knn_pred))


# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:54.759704Z","iopub.status.idle":"2026-01-12T11:53:54.759989Z","shell.execute_reply.started":"2026-01-12T11:53:54.759855Z","shell.execute_reply":"2026-01-12T11:53:54.759871Z"}}
X_train[9]

# %% [code] {"execution":{"iopub.status.busy":"2026-01-12T11:53:54.761134Z","iopub.status.idle":"2026-01-12T11:53:54.761417Z","shell.execute_reply.started":"2026-01-12T11:53:54.761283Z","shell.execute_reply":"2026-01-12T11:53:54.761299Z"}}
input_text=([ 1.66269704,  0.28661784,  0.17057683,  0.58720878, -1.7945775 ,
        0.27893999, -0.07398342, -0.8789345 ,  0.89392419,  1.54431772,
        0.45412163,  0.80239368,  1.71401613, -1.24258316, -0.71363964,
       -0.01444781,  1.13082256,  2.36814613, -0.30743549, -0.97641798,
       -0.26550856, -1.6607716 ,  0.30705682, -0.42801211,  0.11531777,
        1.04284621, -0.17501936,  1.18966164,  0.55809888,  1.20746441,
        0.25451706,  1.46935479,  0.97045501,  1.377703  , -0.95756614,
       -0.3822719 ,  2.64575131, -0.37940297, -0.39505615, -0.38513031,
       -0.36781592, -0.38083877])
np_df = np.asarray(input_text)
prediction=rf.predict(np_df.reshape(1,-1))
if prediction[0] ==1:
    print("CKD")
else:
    print("No CKD")
