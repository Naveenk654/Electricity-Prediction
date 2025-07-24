
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import  ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import OneHotEncoder
import pickle

df=pd.read_csv('/content/electricity_cost_dataset.csv')

df.sample(5)

df.info()

df.describe()

df.columns

numeric=df.drop(columns=['structure type'])
cat=df['structure type']

cat



size_num=len(numeric.columns)
size_num
for col in numeric.columns:
  plt.figure()
  sns.kdeplot(data=df[col])
  plt.xlabel(col)
plt.tight_layout()
plt.show()

for col in numeric.columns:
  plt.figure()
  sns.boxplot(data=df[col])
  plt.xlabel(col)
plt.tight_layout()
plt.show()

for col in numeric.columns:
  sns.regplot(data=numeric, x=col, y='electricity cost')
  plt.xlabel(col)
  plt.ylabel('electricity cost')
  plt.show()

columns = [
    'electricity cost',
    'site area',
    'water consumption',
    'resident count',
    'utilisation rate',
    'recycling rate',
    'issue reolution time',
    'air qality index'
]

df_subset = df[columns]
correlation_matrix = df_subset.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

#more coorealted are-site area,water consumption,resident count,utilisation rate
imp_feature=['site area','water consumption','resident count','utilisation rate','structure type']
target='electricity cost'

df['structure type'].value_counts()

X=df[imp_feature]
y=df[target]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

numeric_col=['site area','water consumption','resident count','utilisation rate']
categorical_cols=['structure type']

preprocessor=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric_col),
        ('cat',OneHotEncoder(),categorical_cols)
    ]
)

model_pipeline=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',RandomForestRegressor())
    ]
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.3f}")

model_pipeline=Pipeline(
    steps=[
        ('preprocessor',preprocessor),
        ('regressor',LinearRegression())
    ]
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.3f}")

"""#by using all the features

"""

X=df.drop(columns=['electricity cost'])
y=df['electricity cost']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

numeric_col_all=df.drop(columns=['structure type','electricity cost']).columns
categorical_cols_all=['structure type']

preprocessor_all=ColumnTransformer(
    transformers=[
        ('num',StandardScaler(),numeric_col_all),
        ('cat',OneHotEncoder(),categorical_cols_all)
    ]
)

model_pipeline=Pipeline(
    steps=[
        ('preprocessor',preprocessor_all),
        ('regressor',RandomForestRegressor())
    ]
)

model_pipeline2=Pipeline(
    steps=[
        ('preprocessor',preprocessor_all),
        ('regressor',LinearRegression())
    ]
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.3f}")

model_pipeline2.fit(X_train, y_train)
y_pred = model_pipeline2.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"MAE: ₹{mae:.2f}")
print(f"RMSE: ₹{rmse:.2f}")
print(f"R² Score: {r2:.3f}")

"""#since we can see there is very sligt change in the metrics after using all columns in random forest but using less columns and producing almost same result is preferred"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV



preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_col),
    ('cat', OneHotEncoder(), categorical_cols)
])
rf2=RandomForestRegressor(random_state=42)

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('regressor', rf2)
])

from sklearn.model_selection import RandomizedSearchCV
# iam using randomizedSerachCv because i have tried using grid search but it was very slow taking very long time
param_tuned = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['sqrt', 'log2']
}

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_tuned,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)
print("Best CV Score:", random_search.best_score_)

final_model = random_search.best_estimator_

with open('model.pkl','wb') as f:
  pickle.dump(final_model,f)