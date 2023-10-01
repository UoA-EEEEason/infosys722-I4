# -*- coding: utf-8 -*-

# 01-BU

"""

Describe the business objectives here

"""

# 02-DU

# Load Dataset
file='Total_Emissions_Per_Country.xlsx'
import pandas as pd
df = pd.read_excel(file)

# Explore Data
df.info()
print(df.describe())

# Add Visualisations

emission_counts = df['Item'].value_counts()
print(emission_counts)
import matplotlib.pyplot as plt
df['Item'].value_counts().plot(kind='bar')
plt.title('Count of each Greenhouse Gas Emission Source')
plt.ylabel('Count')
plt.xlabel('Emission Source')
plt.show()

element_counts = df['Element'].value_counts()
print(element_counts)
import matplotlib.pyplot as plt
df['Element'].value_counts().plot(kind='barh')
plt.title('Count of each Greenhouse Gas Emission Element')
plt.ylabel('Count')
plt.xlabel('Element')
plt.show()

top_sources = df.groupby('Item')['year_2020'].sum().nlargest(10).index
filtered_data = df[df['Item'].isin(top_sources)]
import networkx as nx
import matplotlib.pyplot as plt
G = nx.Graph()
for source in filtered_data['Item'].unique():
    G.add_node(source, type='source')
for element in filtered_data['Element'].unique():
    G.add_node(element, type='element')
for index, row in filtered_data.iterrows():
    G.add_edge(row['Item'], row['Element'])
pos = nx.circular_layout(G)
node_colors = ["red" if G.nodes[node]['type'] == 'source' else "blue" for node in G.nodes()]
fig, ax = plt.subplots(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_colors, font_size=10, width=0.5, ax=ax)
ax.set_title("Web Graph of Top 10 Emission Sources and Gas Types in 2020")
plt.show()

import pandas as pd
# Missing values
missing_percentage = (df.isnull().sum() / len(df)) * 100
# Duplicate rows
duplicates = df.duplicated().sum()
# constant column
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
import matplotlib.pyplot as plt
import seaborn as sns
# Set canvas size
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)

missing_percentage.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Percentage of Missing Values by Column')
plt.ylabel('Percentage')
plt.xlabel('Columns')

plt.subplot(2, 2, 2)
sns.barplot(x=['Duplicates', 'Unique'], y=[duplicates, len(df) - duplicates], palette='pastel')
plt.title('Duplicate Rows in Data')
plt.ylabel('Count')

plt.subplot(2, 2, 3)
sns.barplot(x=list(df.columns), y=df.nunique(), palette='pastel')
plt.title('Number of Unique Values by Column')
plt.ylabel('Unique Count')
plt.xlabel('Columns')

plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 03-DP

years = ['year_' + str(year) for year in range(2000, 2021)]
filtered_df = df[df[years].ge(1).all(axis=1)]
print(filtered_df.describe())

# 添加排放源数量字段
source_counts = filtered_df.groupby('Country')['Item'].nunique().reset_index()
source_counts.columns = ['Country', 'Record_Count']
source_counts.info()

# 合并排放源数量
df1 = pd.merge(filtered_df, source_counts, on='Country', how='left')

# 添加排放源数量等级字段

def classify_level(record_count):
    if record_count < 20:
        return 'Level1'
    elif 20 <= record_count < 25:
        return 'Level2'
    elif 25 <= record_count < 30:
        return 'Level3'
    elif 30 <= record_count < 35:
        return 'Level4'
    elif 35 <= record_count:
        return 'Level5'
df1['Level_PollutionSources'] = df1['Record_Count'].apply(classify_level)

# 添加排放总量字段
country_sum = df1.groupby('Country')['year_2020'].sum().reset_index()
country_sum.rename(columns={'year_2020': 'year_2020_Sum'}, inplace=True)
df2 = pd.merge(df1, country_sum, on='Country', how='left')
ranking = country_sum.sort_values('year_2020_Sum', ascending=False).reset_index(drop=True)
print(ranking)

# Clean the EU dataset using the same steps
file1='Total_Emissions_of_European_Union.xlsx'
dfEU = pd.read_excel(file1)
filtered_dfEU = dfEU[dfEU[years].ge(0).all(axis=1)]
EU_counts = filtered_dfEU.groupby('Country')['Item'].nunique().reset_index()
EU_counts.columns = ['Country', 'Record_Count']
EU_counts.info()
dfEU1 = pd.merge(filtered_dfEU, EU_counts, on='Country', how='left')
dfEU1['Level_PollutionSources'] = dfEU1['Record_Count'].apply(classify_level)
countryEU_sum = dfEU1.groupby('Country')['year_2020'].sum().reset_index()
countryEU_sum.rename(columns={'year_2020': 'year_2020_Sum'}, inplace=True)
dfEU2 = pd.merge(dfEU1, countryEU_sum, on='Country', how='left')

# Merge EU datasets
merged_df = pd.concat([df2, dfEU2], ignore_index=True)

# 04-DT

# Delete Unit field
merged_df = merged_df.drop(columns=['Unit'])

#Convert data to LogN
import numpy as np
years = [f'year_{year}' for year in range(2000, 2021)]
for year in years:
    merged_df[year] = np.log(merged_df[year] + 1)

#Generate a normal distribution plot of the data
import matplotlib.pyplot as plt
import seaborn as sns
years = [f'year_{year}' for year in range(2000, 2021)]
sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(15, 20))
fig.tight_layout(pad=5.0)
for i, year in enumerate(years):
    row = i // 3
    col = i % 3
    sns.histplot(merged_df[year], kde=True, ax=axes[row, col])
    axes[row, col].set_title(f'Distribution of {year}')
    axes[row, col].set_xlabel(year)
    axes[row, col].set_ylabel('Frequency')
plt.show()

# Group data
level_counts = merged_df['Level_PollutionSources'].value_counts()
level_percentage = (level_counts / level_counts.sum()) * 100
# Generate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=level_counts.index, y=level_counts.values, palette="viridis")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(level_percentage[p.get_x()]),
            ha="center")
plt.title('Distribution of Pollution Sources Levels')
plt.xlabel('Pollution Sources Level')
plt.ylabel('Count')
plt.show()

#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
level_mapping = {
    'Level1': 'Low Level',
    'Level2': 'Low Level',
    'Level3': 'Low Level',
    'Level4': 'Normal Level',
    'Level5': 'High Level'
}
merged_df['Level_PollutionSources'] = merged_df['Level_PollutionSources'].replace(level_mapping)
level_counts1 = merged_df['Level_PollutionSources'].value_counts()
level_percentage1 = (level_counts1 / level_counts1.sum()) * 100
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=level_counts1.index, y=level_counts1.values, palette="viridis")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(level_percentage1[p.get_x()]),
            ha="center")
plt.title('Distribution of Grouped Pollution Sources Levels')
plt.xlabel('Pollution Sources Level Group')
plt.ylabel('Count')
plt.show()

# Balence data
from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
X = merged_df.drop('Level_PollutionSources', axis=1)
y = merged_df['Level_PollutionSources']
# RandomOverSampler
ros = RandomOverSampler(sampling_strategy='auto')
X_resampled, y_resampled = ros.fit_resample(X, y)
# RandomUnderSampler
# rus = RandomUnderSampler(sampling_strategy='auto')
# X_resampled, y_resampled = rus.fit_resample(X, y)
balanced_data = pd.concat([X_resampled, y_resampled], axis=1)
levelB_counts = balanced_data['Level_PollutionSources'].value_counts()
levelB_percentage = (levelB_counts / levelB_counts.sum()) * 100
# Generate
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
ax = sns.barplot(x=levelB_counts.index, y=levelB_counts.values, palette="viridis")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(levelB_percentage[p.get_x()]),
            ha="center")
plt.title('Distribution of Grouped Pollution Sources Levels (Balanced)')
plt.xlabel('Pollution Sources Level Group')
plt.ylabel('Count')
plt.show()

# 05-DMM

"""

Identify the Data Mining method
Describe how it aligns with the objectives

"""

# 06-DMA

# 
emission_2020 = balanced_data[['Country', 'year_2020_Sum']]
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 10))
sns.barplot(x=emission_2020['Country'], y=emission_2020['year_2020_Sum'], palette="viridis")
plt.xticks(rotation=90)
plt.title('Total Greenhouse Gas Emissions by Country for 2020')
plt.xlabel('Country')
plt.ylabel('Total Emissions in 2020')
plt.show()

#
def label_country(row):
    if row['Country_ID'] == 48:
        return 'Top1(China mainland)'
    elif row['Country_ID'] == 233:
        return 'Top2(USA)'
    elif row['Country_ID'] == 102:
        return 'Top3(India)'
    elif row['Country_ID'] == 247:
        return 'Top4(European Union)'
    elif row['Country_ID'] == 29:
        return 'Top5(Brazil)'
    else:
        return 'Others'
balanced_data['Country'] = balanced_data.apply(label_country, axis=1)
grouped_data = balanced_data.groupby('Country')['year_2020_Sum'].sum().reset_index()
import matplotlib.pyplot as plt

# data and labels
sizes = grouped_data['year_2020_Sum']
labels = grouped_data['Country']

# Pie figure
plt.figure(figsize=(10, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", 7))
plt.title('Greenhouse Gas Emissions Proportion in 2020 by Group')
plt.show()

# training model
from sklearn.model_selection import train_test_split
X = balanced_data[['Record_Count']]
y = balanced_data['year_2020_Sum']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
# initial model
model = LinearRegression()
# training data
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training Score: {train_score}")
print(f"Test Score: {test_score}")
from sklearn.metrics import mean_squared_error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# 获取系数和截距
coefficients = model.coef_
intercept = model.intercept_
print("Coefficients:", coefficients)
print("Intercept:", intercept)
#
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, model.predict(X_test), color='red', linewidth=2, label='Linear Regression')
plt.title('Linear Regression on Greenhouse Gas Emissions')
plt.xlabel('Record_Count (Emission Sources Quantity)')
plt.ylabel('year_2020_Sum (Total Emission)')
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=False, useOffset=False))
plt.legend()
plt.show()


#DecisionTree

from sklearn.model_selection import train_test_split
X = balanced_data[['year_2020_Sum']]
y = balanced_data['Level_PollutionSources']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier
# Initial decesion tree
clf = DecisionTreeClassifier()
# training
clf = DecisionTreeClassifier(max_depth=10)
clf.fit(X_train, y_train)
# score
train_accuracy = clf.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
test_accuracy = clf.score(X_test, y_test)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(30, 15))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=True, rounded=True)
plt.savefig("decision_tree.png", dpi=300)  # 保存为高DPI的PNG文件

# 07-DM

# Execute DM task

# 08-INT

# Iteration
from sklearn.model_selection import train_test_split
X = balanced_data[['year_2020_Sum']]
y = balanced_data['Level_PollutionSources']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
# Initial decesion tree
clf = DecisionTreeClassifier()
# training
clf = DecisionTreeClassifier(max_depth=15)
clf.fit(X_train, y_train)
# score
train_accuracy = clf.score(X_train, y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
test_accuracy = clf.score(X_test, y_test)
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# 09-ACT

"""

Desribe the Action Plan to Implement, Observe and Improve

"""
