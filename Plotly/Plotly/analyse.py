import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from IPython.display import HTML


import warnings
warnings.filterwarnings('ignore')

width_graph= 600
heigh_graph= 600

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

data= pd.read_csv(f"{BASE_DIR}\\Plotly\\staticfiles\\static\\marketing_campaign.csv",sep="\t")  
data = data[(data["Income"]<400000)&(data["Year_Birth"]>1920)]
data.dropna(inplace=True)





# Calculer l'âge des clients
today = date.today()
current_year = today.year
data['client_age'] = current_year - data['Year_Birth']
first_graph= px.histogram(data, x="client_age", nbins=20,color_discrete_sequence=['#FF5733'])
first_graph.update_layout(
    width=width_graph,
    height=heigh_graph,
    xaxis_title="Income",  # Add x-axis label
    yaxis_title="Count",   # Add y-axis label
    bargap=0.05,           # Reduce the gap between bars
    margin=dict(l=50, r=50, b=50, t=50),  # Add margins
)

first_graph.update_traces(
    marker=dict(color="skyblue"),  # Change bar color
    opacity=0.75,                   # Set bar opacity
    hoverinfo="y",                  # Show count on hover
    hovertemplate="Count: %{y}"     # Customize hover template
)





income= px.histogram(data, x="Income", nbins=20)
income.update_layout(
    width=width_graph,
    height=heigh_graph,
    xaxis_title="Income",  # Add x-axis label
    yaxis_title="Count",   # Add y-axis label
    bargap=0.05,           # Reduce the gap between bars
    margin=dict(l=50, r=50, b=50, t=50),  # Add margins
)

income.update_traces(
    marker=dict(color="skyblue"),  # Change bar color
    opacity=0.75,                   # Set bar opacity
    hoverinfo="y",                  # Show count on hover
    hovertemplate="Count: %{y}"     # Customize hover template
)


#2nd graph - EDUCATION
edu_percentage = data['Education'].value_counts(normalize=True) * 100
marital_percentage = data['Marital_Status'].value_counts(normalize=True) * 100

fig1 = px.bar(x=edu_percentage.index, y=edu_percentage.values, labels={'x': 'Education', 'y': 'Percentage'}, title='Education Level of Clients')
fig2 = px.bar(x=marital_percentage.index, y=marital_percentage.values, labels={'x': 'Marital Status', 'y': 'Percentage'}, title='Marital Status of Clients')
fig = make_subplots(rows=1, cols=2, subplot_titles=("Education Level of Clients", "Marital Status of Clients"))


for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)

for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
fig.update_layout(
    width=1200,
    height=800
)

#Heatmap
data = data.rename(columns={'MntWines': "sp_wines", 'MntFruits': 'sp_fruits', 'MntMeatProducts': 'sp_meat',
                            'MntFishProducts': "sp_fish", 'MntSweetProducts': 'sp_sweet', 'MntGoldProds': 'sp_gold'})


data['Spending'] = data['sp_wines'] + data['sp_fruits'] + data['sp_meat'] + data['sp_fish'] + data['sp_sweet'] + data[
    'sp_gold']
data["Frequency"] = data["NumWebPurchases"] + data["NumCatalogPurchases"] + data["NumStorePurchases"]
data['children'] = data['Kidhome'] + data['Teenhome']

df_encoded = data[['client_age', 'Education', 'Marital_Status', 'Recency', 'Income', 'Spending', 'children', 'sp_wines',
                   'sp_fruits', 'sp_meat', 'sp_fish', 'sp_sweet', 'sp_gold', 'Frequency']]

for col in df_encoded.select_dtypes(include='number'):
    decile_threshold = df_encoded[col].quantile(0.99)  
    df_encoded = df_encoded[df_encoded[col] <= decile_threshold]
    
label_encoder = LabelEncoder()
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object':
        df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

corr_matrix = df_encoded.corr()


heat=px.imshow(corr_matrix,labels=dict(color="Correlation Score"),aspect="auto",color_continuous_scale='RdBu_r')

heat.update_layout(
    width=1200,
    height=900,
    xaxis_showgrid=False,
    yaxis_showgrid=False,)
boxplotsO = make_subplots(rows=3, cols=2, subplot_titles=("Spending on Wine", "Spending on Fruit", "Spending on Meat", "Spending on Fish", "Spending on Sweet", "Spending on Gold"))

boxplotsO.add_trace(go.Box(x=data["sp_wines"], name="Wine"), row=1, col=1)
boxplotsO.add_trace(go.Box(x=data["sp_fruits"], name="Fruit"), row=1, col=2)
boxplotsO.add_trace(go.Box(x=data["sp_meat"], name="Meat"), row=2, col=1)
boxplotsO.add_trace(go.Box(x=data["sp_fish"], name="Fish"), row=2, col=2)
boxplotsO.add_trace(go.Box(x=data["sp_sweet"], name="Sweet"), row=3, col=1)
boxplotsO.add_trace(go.Box(x=data["sp_gold"], name="Gold"), row=3, col=2)
boxplotsO.update_layout( height=1000, width=1200)


## GRAPH 3D HUE

education_mapping = {
    'Graduation': 2,
    'Basic': 1,
    '2n Cycle': 3,
    'Master': 4,
    'PhD': 5
}

df=data.copy()
df_numeric = df.select_dtypes(include=['int64', 'float64'])
df_numeric = df_numeric[['Income', 'sp_wines', 'sp_fruits',
                         'sp_meat', 'sp_fish', 'sp_sweet'
                        ]]
for col in df_numeric.columns:
    decile_threshold = df[col].quantile(0.999)  
    df = df[df[col] <= decile_threshold]

df=df.dropna()


# Mapper les valeurs de la colonne 'education' en utilisant le dictionnaire de mappage
df['education_mapped'] = df['Education'].map(education_mapping)


# Encoder les catégorielles sur df
encoder = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = encoder.fit_transform(df[col])

#sp_fruits
green_3D_1 = go.Figure(data=[go.Scatter3d(
    x=df['Income'],
    y=df['Year_Birth'],
    z=df['education_mapped'],
    mode='markers',
    marker=dict(
        size=3,
        color=df['sp_fruits'],       
        colorscale='Greens', 
        opacity=0.8
))])

green_3D_1.update_layout(scene=dict(
                        xaxis_title='Income',
                        yaxis_title='Year_Birth',
                        zaxis_title='Education'))
#sweet
green_3D_1.update_layout(
    width=1000,
    height=800,
    xaxis_showgrid=False,
    yaxis_showgrid=False,)

green_3D_2 = go.Figure(data=[go.Scatter3d(
    x=df['Income'],
    y=df['Year_Birth'],
    z=df['education_mapped'],
    mode='markers',
    marker=dict(
        size=3,
        color=df['sp_sweet'],       
        colorscale='Greens', 
        opacity=0.8
))])
green_3D_2.update_layout(
    width=1000,
    height=800,
    xaxis_showgrid=False,
    yaxis_showgrid=False,)
green_3D_2.update_layout(scene=dict(
                        xaxis_title='Income',
                        yaxis_title='Year_Birth',
                        zaxis_title='Education'))


################################CLUSTERING#############################
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
from io import StringIO
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
from plotly import tools
import chart_studio.plotly as py
from sklearn.decomposition import PCA

#elbow
sum_of_sq_dist = {}
for k in range(1,12):
    kmean = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
    km = kmean.fit(df)
    sum_of_sq_dist[k] = km.inertia_

trace = go.Scatter(x=list(sum_of_sq_dist.keys()), y=list(sum_of_sq_dist.values()), mode='lines+markers')
elbow = go.Figure(trace)

elbow.update_layout(
    xaxis_title='Number of Clusters(k)',
    yaxis_title='Sum of Square Distances',
    width=1200,
    height=900
)


####PCA 2D PLOT

X = df.select_dtypes(include='number') #
scaler = StandardScaler() #
X_scaled = scaler.fit_transform(X) #

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled) #

df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_pca)
df_pca['Cluster'] = kmeans.labels_.astype(str)  

PCA2D = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster',
                 labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                 color_discrete_sequence=px.colors.qualitative.Set1)
PCA2D.update_layout(
    width=1000,
    height=800
)

#PCA3D
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled) #

df_pca_3d = pd.DataFrame(data=X_pca_3d, columns=['PC1', 'PC2', 'PC3'])

kmeans_3d = KMeans(n_clusters=4)
kmeans_3d.fit(df_pca_3d)
df_pca_3d['Cluster'] = kmeans_3d.labels_

PCA3D = go.Figure(data=[go.Scatter3d(
    x=df_pca_3d['PC1'],
    y=df_pca_3d['PC2'],
    z=df_pca_3d['PC3'],
    mode='markers',
    marker=dict(color=df_pca_3d['Cluster'], colorscale='Viridis', size=5),
)])


PCA3D.update_layout(scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'), width=1000,
    height=800
                  )


# copy df for proper analysis based on kmeans cluster (4 of them)

df_analysis = df.copy()
df_analysis["kclusters"] = kmeans_3d.labels_

df_means = pd.DataFrame(index=df_analysis["kclusters"].unique())


for i in ["Income", "Recency", "Year_Birth"]:
    df_means = pd.merge(
        df_means, 
        df_analysis[["kclusters",i]].groupby(by="kclusters").mean(),
        how="left",left_index=True,right_on="kclusters")

for i in ["Income", "Recency", "Year_Birth"]:
    df_means = pd.merge(
        df_means, 
        df_analysis[["kclusters",i]].groupby(by="kclusters").median(),
        how="left",left_index=True,right_on="kclusters")


df_means.columns = ["mean_income","mean_recency","mean_Birth","median_income","median_recency","median_Birth"]

df_means = round(df_means,2)
df_means.sort_index(inplace=True)



#PARTIE RFM ; Réimport du dataset


df = pd.read_csv(f"{BASE_DIR}\\Plotly\\staticfiles\\static\\marketing_campaign.csv",sep="\t")  
df = df[(df["Income"]<400000)&(df["Year_Birth"]>1920)]

df["Frequency"] = df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
df["MonetaryValue"] = df["MntWines"] + df["MntFruits"] + df["MntWines"] + df["MntMeatProducts"]+ df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]

Rlabel = range(4,0,-1)
Flabel = range(1,5)
Mlabel = range(1,5)

Rgrp = pd.qcut(df["Recency"], q=4, labels = Rlabel)
Fgrp = pd.qcut(df["Frequency"], q=4, labels = Flabel)
Mgrp = pd.qcut(df["MonetaryValue"], q=4, labels = Mlabel)

df["R"] = Rgrp.values
df["F"] = Fgrp.values
df["M"] = Mgrp.values

df["RFM_Concat"] = df.apply(lambda x : str(x["R"]) + str(x["F"]) + str(x["M"]), axis=1 )
df["Score"] = df[['R',"F","M"]].sum(axis=1)

def RFMlevel(df):
    if(df['Score']>9):
        return "1Première"
    elif (df["Score"]>9)and(df["Score"] >=7): return "2Champions"
    elif (df["Score"]>7)and(df["Score"] >=6): return "3Loyal"
    elif (df["Score"]>6)and(df["Score"] >=5): return "4Potential"
    elif (df["Score"]>5)and(df["Score"] >=4): return "5Promising"
    elif (df["Score"]>4)and(df["Score"] >=3): return "6Need attention"
    else : return "7Require activation"

df["RFM_Level"] = df.apply(RFMlevel,axis=1)

df_stats= df.groupby("RFM_Level").agg({
    "Recency":"mean",
    'Frequency':"mean",
    "MonetaryValue":["mean",'count']
}).round(1)
df_stats.columns = df_stats.columns.droplevel()
df_stats.columns = ["Recency_Mean","Frequency_Mean","MonetaryValue_Mean","MonetaryValue_Count"]

#graph

df_fig = pd.DataFrame({
    'label': df_stats.index.unique().tolist(),
    'size': df_stats["Frequency_Mean"],
    'color': ["Green","Orange","Purple","Maroon","Pink","Teal"]
})

# Créez un diagramme de "trésors"(mdrrRRRR) avec Plotly Express
figRFM = px.treemap(df_fig, 
                 path=['label'], 
                 values='size',
                 color='color',
                 color_discrete_map={i: c for i, c in enumerate(["Green","Orange","Purple","Maroon","Pink","Teal"])})

figRFM.update_layout(
                        width=1200, 
                        height=800
                    )

##################################################################################################
############### Reprise du df
#################################################################################################
#GRAPH WGF
data= pd.read_csv(f"{BASE_DIR}\\Plotly\\staticfiles\\static\\marketing_campaign.csv",sep="\t")  
data = data[(data["Income"]<400000)&(data["Year_Birth"]>1920)]
data.dropna(inplace=True)


X = data[['MntGoldProds', 'MntFishProducts', 'MntWines']]
kmeans = KMeans(n_clusters=4) 

kmeans.fit(X)
data['Cluster'] = kmeans.labels_
cluster_stats = data.groupby('Cluster')[['MntGoldProds', 'MntFishProducts', 'MntWines']].mean()

cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['MntGoldProds', 'MntFishProducts', 'MntWines'])

X['Cluster'] = kmeans.labels_
WGF = px.scatter_3d(X, x='MntGoldProds', y='MntFishProducts', z='MntWines', color='Cluster',color_discrete_map={'0': 'red', '1': 'blue', '2': 'green', '3': 'yellow'},
                   
                    labels={'Cluster': 'Cluster'})
WGF.add_trace(px.scatter_3d(cluster_centers, x='MntGoldProds', y='MntFishProducts', z='MntWines').data[0])
WGF.update_traces(marker_size = 4)
WGF.update_layout(width=900,
    height=800)
WGF.update_layout(
    legend=dict(
        x=0,  # Positionnement de la légende à gauche du graphique
        y=0.5  # Ajustez la position verticale de la légende si nécessaire
    )
)
#2ND GRAPH 3D KMEANS
X = data[['Income', 'MntWines', 'Year_Birth']]
kmeans = KMeans(n_clusters=4)  
kmeans.fit(X)
data['Cluster'] = kmeans.labels_

cluster_stats = data.groupby('Cluster')[['Income', 'MntWines', 'Year_Birth']].mean()
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Income', 'MntWines', 'Year_Birth'])
X['Cluster'] = kmeans.labels_

YIW = px.scatter_3d(data, x='Income', y='MntWines', z='Year_Birth', color='Cluster',
                     labels={'Cluster': 'Cluster'},
                     color_discrete_map={'0': 'red', '1': 'blue', '2': 'green', '3': 'yellow'},
                   )

YIW.add_trace(px.scatter_3d(cluster_centers, x='Income', y='MntWines', z='Year_Birth').data[0])

YIW.update_traces(marker_size=4)

YIW.update_layout(width=900, height=800)

