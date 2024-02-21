import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder


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
    km = kmean.fit(df_encoded)
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
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_encoded)

# Créer un DataFrame avec les données PCA
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

# K-means clustering avec 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_pca)
df_pca['Cluster'] = kmeans.labels_.astype(str)  # Convertir les étiquettes en chaînes de caractères

# Créer le graphique 2D avec Plotly Express
PCA2D = px.scatter(df_pca, x='PC1', y='PC2', color='Cluster', title='PCA + Clustering',
                 labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                 color_discrete_sequence=px.colors.qualitative.Set1)

# Réduire à 3 dimensions pour visualisation
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(df_encoded)

# Créer un DataFrame avec les données PCA 3D
df_pca_3d = pd.DataFrame(data=X_pca_3d, columns=['PC1', 'PC2', 'PC3'])

# K-means clustering avec 4 clusters
kmeans_3d = KMeans(n_clusters=4)
kmeans_3d.fit(df_pca_3d)
df_pca_3d['Cluster'] = kmeans_3d.labels_

# Créer le graphique 3D avec Plotly
PCA3D = go.Figure(data=[go.Scatter3d(
    x=df_pca_3d['PC1'],
    y=df_pca_3d['PC2'],
    z=df_pca_3d['PC3'],
    mode='markers',
    marker=dict(color=df_pca_3d['Cluster'], colorscale='Viridis', size=5),
)])

# Ajouter des étiquettes aux axes
PCA3D.update_layout(scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'),
                  )


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

# Créez un diagramme de "trésors"(mdrr) avec Plotly Express
figRFM = px.treemap(df_fig, 
                 path=['label'], 
                 values='size',
                 color='color',
                 color_discrete_map={i: c for i, c in enumerate(["Green","Orange","Purple","Maroon","Pink","Teal"])})

figRFM.update_layout(
                        width=800, 
                        height=500
                    )
