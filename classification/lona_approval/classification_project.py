import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# lets check the try and except
try:
  df=pd.read_csv("Dataset_1_LoanApproval.csv")
except FileNotFoundError:
  print("File not found")
  
df.head()
df.dtypes

# creating the copy of original dataframe
df1=df.copy()
df1

# droping the (loan_Id) feature because it is not making sence
df1.drop(['Loan_ID'],axis=1,inplace=True)

# checking duplicate rows of dataframe (here no duplicates rows then no need to drop duplicates) 
df1.duplicated().sum()

# checking the missing values in dataframe
df1.isnull().sum()

# imputing missing values 

num_var=df1.select_dtypes(include='number').columns.to_list()
plt.figure(figsize=(12,12))
for i,val in enumerate(num_var):
    plt.subplot(3,2,i+1)
    sns.histplot(df1[val],kde=True)
    
from sklearn.impute import SimpleImputer

#here we can see that feature applicantincome,coapplicantincome,loanamount are highly right skewed so we need to median imputation here

md=SimpleImputer(missing_values=np.nan,strategy='median')

df1[['ApplicantIncome']]=md.fit_transform(df1[['ApplicantIncome']])
df1[['CoapplicantIncome']]=md.fit_transform(df1[['CoapplicantIncome']])
df1[['LoanAmount']]=md.fit_transform(df1[['LoanAmount']])
df1[['Loan_Amount_Term']]=md.fit_transform(df1[['Loan_Amount_Term']])
df1[['Credit_History']]=md.fit_transform(df1[['Credit_History']])

# now filling the categorical value in dataframe

from sklearn.impute import SimpleImputer

ma=SimpleImputer(missing_values=np.nan,strategy='most_frequent')

df1[['Gender']]=ma.fit_transform(df1[['Gender']])
df1[['Married']]=ma.fit_transform(df1[['Married']])
df1[['Education']]=ma.fit_transform(df1[['Education']])

df1.isnull().sum()

# now we will fix the nan values and class imbalance in the dataset

df1['Loan_Status'].value_counts()
# first we will fill the missing values

df1['Loan_Status'] = df1['Loan_Status'].fillna(df1['Loan_Status'].mode()[0])
df1['Loan_Status'].isnull().sum()
# null values has beeen filled


# before this we neee to convert target column to numeric 

df1['Loan_Status']=df1['Loan_Status'].map({'Y':1,'N':0})

# converting categorical data into numeric

cat_var=df1.select_dtypes(include='object').columns.to_list()


import category_encoders as ce

encoder = ce.OneHotEncoder(cols=cat_var, use_cat_names=True)
df2 = encoder.fit_transform(df1)
df2.isnull().sum()
# Most imp step feature selection 

sns.heatmap(df2.corr(),annot=True)
# from this we can see that only credit score is strongly relate to loan status

# now we will apply embedded method for feature selection 
from sklearn.model_selection import train_test_split

X=df2.drop(['Loan_Status'],axis=1)
y=df2['Loan_Status']

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)

feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance from Random Forest:")
top_3=(feat_imp*100).head(3).index.to_list()
print(top_3)
# from this it we can see that the Feature (ApplicantIncome,Credit_History,CoapplicantIncome)

# Now we will fix the skewness of the data 
for i,val in enumerate(top_3):
    plt.subplot(3,1,i+1)
    sns.histplot(df2[val],kde=True)
    plt.tight_layout()
    
# now here all features are right skewed we use log transformation

for i in top_3:
    df2[i]=np.log1p(df2[i])
    
for i,val in enumerate(top_3):
    plt.subplot(3,1,i+1)
    sns.histplot(df2[val],kde=True)
    plt.tight_layout()
df2[['CoapplicantIncome']]=md.fit_transform(df2[['CoapplicantIncome']])
# now the values are normalize

fea_df=df2[['ApplicantIncome','Credit_History','CoapplicantIncome','Loan_Status']]

fea_df.isnull().sum()
fea_df['Loan_Status'].value_counts()

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X=fea_df.drop(['Loan_Status'],axis=1)
y=fea_df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# now we will do scaling 

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train_sm_scaled = sc.fit_transform(X_train_sm)
X_test_scaled = sc.transform(X_test)  

##################################################################################
#------------Now every thing is done -----------------------#
#------------Now we will train model=======================#




from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100,max_depth=None,random_state=42)

rf.fit(X_train_sm_scaled, y_train_sm)

pred=rf.predict(X_test_scaled)
print(pred)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print(" Accuracy:", accuracy_score(y_test, pred)*100)
print("\nClassification Report:\n", classification_report(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))

'''
Overall Performance

The model achieved an accuracy of 95% on the test dataset.

This means the model correctly predicted 57 out of 60 cases.

Class-wise Results

Loan Rejected (Class 0):

Precision = 0.91 → When the model predicts a loan will be rejected, it is correct 91% of the time.

Recall = 0.95 → Out of all actual rejected cases, the model correctly identified 95%.

Loan Approved (Class 1):

Precision = 0.97 → When the model predicts a loan will be approved, it is correct 97% of the time.

Recall = 0.95 → Out of all actual approved cases, the model correctly identified 95%.

Confusion Matrix Insight

True Negatives (Loan Rejected correctly): 20

False Positives (Predicted Approved but actually Rejected): 1

False Negatives (Predicted Rejected but actually Approved): 2

True Positives (Loan Approved correctly): 37

Conclusion

The Random Forest model is performing very well with balanced precision and recall for both classes.

It is reliable for predicting loan approval status and can be used for decision-making with high confidence.
'''















