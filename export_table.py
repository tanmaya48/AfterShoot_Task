import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from constants import sliders_table_path

df = pd.read_csv(sliders_table_path)

target_columns = ['Temperature', 'Tint']

columns_to_keep = ["image_id","currTemp","currTint", 
                   "aperture", "flashFired", "focalLength", "isoSpeedRating", "shutterSpeed",
                   "camera_model","camera_group","intensity","ev",
                   "Temperature","Tint"]

df = df[columns_to_keep]

df['aperture'] = df['aperture'].fillna(df['aperture'].median())
df['focalLength'] = df['focalLength'].fillna(df['focalLength'].median())


df.to_csv('profile.csv')


numerical_columns = ['currTemp', 'currTint', 'aperture', 'flashFired',
       'focalLength', 'isoSpeedRating', 'shutterSpeed', 'intensity', 'ev']


for col in numerical_columns:
    df[col] = (df[col]-df[col].mean(0))/df[col].std()

scaler = MinMaxScaler()



threshold = 50
category_counts = df['camera_model'].value_counts()
df['camera_model'] = df['camera_model'].apply(lambda x: x if category_counts[x] >= threshold else 'Other')
df = pd.get_dummies(df, columns=['camera_model'])


df['Temperature_regr'] = (df['Temperature'])/5000
df['Tint_regr'] = scaler.fit_transform(df[['Tint']])

df[df.select_dtypes(include='bool').columns] = df.select_dtypes(include='bool').astype(int)
print(df.describe())


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)


df.to_csv('profile.csv')
train_df.to_csv('profile_train.csv')
test_df.to_csv('profile_test.csv')
