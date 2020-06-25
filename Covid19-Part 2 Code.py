# Databricks notebook source
# MAGIC %md
# MAGIC #Now let's focus on India 

# COMMAND ----------

# MAGIC %md #### Death due to Corona every month

# COMMAND ----------

# MAGIC %sh pip install plotly dash

# COMMAND ----------

import plotly.express as px
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
%matplotlib inline
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from  plotly.offline import *
from pyspark.sql.functions import *
# import cufflinks as cf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import  KMeans
import pandas as pd
from plotly.offline import *
init_notebook_mode(connected=True)
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

 

init_notebook_mode(connected=True)
from plotly import __version__




# COMMAND ----------

payload = {'country': 'India'} # or {'code': 'DE'}
URL = 'https://api.statworx.com/covid'
response = requests.post(url=URL, data=json.dumps(payload))


df7 = pd.DataFrame.from_dict(json.loads(response.text))


fig = px.line(df7, x="day", y="deaths", color='month')
fig=plot(fig,output_type='div')
displayHTML(fig)

# COMMAND ----------

# MAGIC %md ## Clustering data To Get Hotspots

# COMMAND ----------

from pyspark.sql.types import  StructField,StringType,IntegerType,StructType,FloatType,DateType,DateConverter

data_schema=[StructField("Date",DateType(),True),
             StructField("State_or_UT",StringType(),True),
             StructField("Indian_cases",IntegerType(),True),
             StructField("Foreign_cases" ,IntegerType(),True),
             StructField("Cured",IntegerType(),True),
             StructField("Latitude",FloatType(),True),
             StructField("Longitude",FloatType(),True),
             StructField("Death",IntegerType(),True),
             StructField("Total_cases",IntegerType(),True)
            ]

dfschema=StructType(data_schema)

# COMMAND ----------

df=spark.read.csv("dbfs:/FileStore/tables/complete1.csv",mode='FAILFAST',header=True,schema=dfschema)

# COMMAND ----------

df=df.withColumn('active',df['Total_cases']-df['Death']-df['Cured'])
source= df.withColumn('month',month('Date')).withColumn('day',dayofyear('Date')).toPandas()
source1=source.loc[:,['Cured','Death','Total_cases','active']]

# COMMAND ----------

csrc=source[['month','Total_cases','Death','Latitude','Longitude','Cured']]
x=csrc.values

# COMMAND ----------

# MAGIC %md Finding The most optimal number of clusters

# COMMAND ----------

from sklearn.cluster import KMeans
def best_knumber_cluster(x,iter_number):
    wwss=[]
    for i in range(1,iter_number+1):
        kmeans=KMeans(n_clusters=i)
        kmeans.fit(x)
        wwss.append(kmeans.inertia_)
    plt.figure(figsize=(20,10))
    c=plt.plot(np.arange(1,iter_number+1),wwss,marker='o',markersize=10,markerfacecolor='black')
    plt.xlabel('number of clusters')
    plt.ylabel('wwss')
    plt.title('Elbow Curve')
    
    return plt.show()


best_knumber_cluster(x,15)

# COMMAND ----------

# MAGIC %md ## Fitting Model

# COMMAND ----------

kmeans=KMeans(n_clusters=4,init='k-means++',max_iter=300,n_init=100,random_state=55)
kmeans.fit(x)
newdf=csrc.copy()
newdf['Cluster']=kmeans.predict(x)

# COMMAND ----------

newdf['Cluster']=newdf['Cluster'].astype(np.str)
def put(num):
  if(num=='2'):
    return 'Highly Alarmed'
  elif(num=='1'):
    return 'Emergency zone'
  elif(num=='3'):
    return "Moderate"
  else:
    return 'safe'
  
newdf['Cluster']=newdf['Cluster'].map(lambda x:put(x))

# COMMAND ----------

import plotly.graph_objects as go
a=metro.toPandas()
fig = go.Figure()

fig.add_trace(go.Scatter(x=a['district'], y=a['national_death_rate'],mode='lines',name='national death rate'))

fig.add_trace(go.Scatter(x=a['district'], y=a['national_active_rate'],mode='lines',name='national active rate'))

fig.add_trace(go.Scatter(x=a['district'], y=a['national_recovered_rate'],mode='lines',name='national recovery rate'))

fig.add_trace(go.Scatter(x=a['district'], y=a['city_recovered_rate'],mode='lines',name='metro cities recovery rate'))

fig.add_trace(go.Scatter(x=a['district'], y=a['city_active_rate'],mode='markers',name='metro cities active rate'))

fig.add_trace(go.Scatter(x=a['district'], y=a['city_death_rate'],mode='lines+markers',name='metro cities death rate'))

fig.update_layout(margin=dict(l=30, r=30, t=20, b=20))

displayHTML(plot(fig,output_type='div'))

# COMMAND ----------

# MAGIC %md #Hotspots (Deadly states)
# MAGIC * Which states are Hotspots and Which are safe
# MAGIC 1. Here same color repersents same condition (safe to deadly)
# MAGIC 2. 'month','Total_cases','Death','Latitude','Longitude','Cured' are Factors on the basis of which Hotspots are generated

# COMMAND ----------

import plotly.express as px
px.set_mapbox_access_token('pk.eyJ1IjoiYXJuYXZzb29kc29vZCIsImEiOiJjazlhMG9hd3AwNGIwM21zNzY3NTRwcGRqIn0.VPR5rODFUVvZ5wwXSSDggg')
fig = px.scatter_mapbox(newdf, lat="Latitude", lon="Longitude", color='Cluster',hover_name=source['State_or_UT'], hover_data=['Total_cases','Death'],
                       mapbox_style='dark',zoom=3,size='Total_cases',)



fig=plot(fig,output_type='div')
displayHTML(fig)

# COMMAND ----------

# MAGIC %md ###District Wise Hotspots in India

# COMMAND ----------

file_location = "/FileStore/tables/finalindia.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)




temp_table_name = "india"

df.createOrReplaceTempView(temp_table_name)
mdf=spark.sql("select * from india ").toPandas()


mdf['co']=mdf['active']*100
px.set_mapbox_access_token("pk.eyJ1Ijoia3JhbTEyMyIsImEiOiJjazlheGlrazIwYWlxM3Jsa3g1Mmk1ZWp6In0.aqT-edMtVi9If2A70LG37w")
fig=px.scatter_mapbox(mdf,lat="lat",lon="lon",color='stat',hover_data=["dist"],zoom=3,size='co')
displayHTML(plot(fig,output_type='div'))

# COMMAND ----------

# MAGIC %md ###District Wise Status

# COMMAND ----------

file_location = "/FileStore/tables/finalindia.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)


temp_table_name = "india"

df.createOrReplaceTempView(temp_table_name)
mdf=spark.sql("select * from india ").toPandas()
import pandas as pd
import plotly.express as px
px.set_mapbox_access_token("pk.eyJ1Ijoia3JhbTEyMyIsImEiOiJjazlheGlrazIwYWlxM3Jsa3g1Mmk1ZWp6In0.aqT-edMtVi9If2A70LG37w")
fig=px.scatter_mapbox(mdf,lat="lat",lon="lon",color='stat',hover_data=["dist"],zoom=3)
displayHTML(plot(fig,output_type='div'))

# COMMAND ----------

df4=spark.read.csv('dbfs:/FileStore/tables/owid_covid_data-649ee.csv',inferSchema=True,header=True)

# COMMAND ----------

columns_to_drop = ['new_deaths_per_million', 'total_tests','new_tests','total_tests_per_thousand']
df4 = df4.drop(*columns_to_drop)

# COMMAND ----------

df1=df4.groupBy('location').agg({'total_cases':'sum'}).dropDuplicates()

# COMMAND ----------

# MAGIC %md
# MAGIC #How much effect China had on it's neighbour ?

# COMMAND ----------

init_notebook_mode(connected=True)
df12=df4.toPandas()
# Create traces
fig = go.Figure()
ind=df12[df12['location']=='India']
bhu=df12[df12['location']=='Bhutan']
afg=df12[df12['location']=='Afghanistan']
Kaz=df12[df12['location']=='Kazakhstan']
Rus=df12[df12['location']=='Russia']
mya=df12[df12['location']=='Myanmar']
nor=df12[df12['location']=='North Korea']
mon=df12[df12['location']=='Mongolia']
vie=df12[df12['location']=='Vietnam']
chi=df12[df12['location']=='China']


 

fig.add_trace(go.Scatter(x=ind['date'], y=ind['total_cases'],
                         mode='lines+markers',
                    name='India'))
fig.add_trace(go.Scatter(x=bhu['date'], y=bhu['total_cases'],
                         mode='lines+markers',
                    name='Bhutan'))
fig.add_trace(go.Scatter(x=chi['date'], y=chi['total_cases'],
                         mode='lines+markers',
                    name='China'))
fig.add_trace(go.Scatter(x=afg['date'], y=afg['total_cases'],
                         mode='lines+markers',
                    name='Afganistan'))
fig.add_trace(go.Scatter(x=Kaz['date'], y=Kaz['total_cases'],
                         mode='lines+markers',
                    name='Kazakhstan'))
fig.add_trace(go.Scatter(x=mya['date'], y=mya['total_cases'],
                         mode='lines+markers',
                    name='Myanmar'))
fig.add_trace(go.Scatter(x=mon['date'], y=mon['total_cases'],
                         mode='lines+markers',
                    name='Mongolia'))
fig.add_trace(go.Scatter(x=Rus['date'], y=Rus['total_cases'],
                         mode='lines+markers',
                    name='Russia'))
fig.add_trace(go.Scatter(x=nor['date'], y=nor['total_cases'],
                         mode='lines+markers',
                    name='North Korea'))
fig.add_trace(go.Scatter(x=vie['date'], y=vie['total_cases'],
                         mode='lines+markers',
                    name='Vietnam'))

displayHTML(plot(fig,output_type='div'))

# COMMAND ----------

state=df.groupby('State_or_UT').agg({'Total_cases': 'max'}).sort('max(Total_cases)', ascending=False)
st=df.toPandas()

# COMMAND ----------

# MAGIC %md #Dividing Indian Cases by Phases

# COMMAND ----------

import requests
import json
payload = {'country': 'India'} # or {'code': 'DE'}
URL = 'https://api.statworx.com/covid'
response = requests.post(url=URL, data=json.dumps(payload))
mydf = pd.DataFrame.from_dict(json.loads(response.text))
mydf=mydf.loc[:,[    'day'    ,'month'    ,'cases',    'deaths'    ,'population' , 'cases_cum'    ,'deaths_cum']]
cpdf=mydf


kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=100)
kmeans.fit(mydf.fillna(0))
newdf=mydf.copy()
newdf['Cluster']=kmeans.predict(newdf.fillna(0))
newdf['Cluster']=newdf['Cluster'].astype(np.str)

 

def put(num):
  if(num=='0'):
    return 'phase1'
  elif(num=='1'):
    return 'phase3'
  elif(num=='2'):
    return "phase2"
  else:
    return 'phase4'
  
newdf['Cluster']=newdf['Cluster'].map(lambda x:put(x))
display(newdf)

# COMMAND ----------

# MAGIC %md ### State wise comparision of Death compared of national

# COMMAND ----------

from pyspark import SparkFiles
url_state="https://api.covid19india.org/csv/latest/state_wise.csv"
spark.sparkContext.addFile(url_state)

state_df_x = spark.read.csv("file://"+SparkFiles.get("state_wise.csv"), header=True, inferSchema= True)

# COMMAND ----------

state_df1=state_df_x.filter(state_df_x['State']!="Total")
state_total=state_df_x.filter(state_df_x['State']=="Total")
state_total=state_total.drop(state_total["Delta_Confirmed"]).drop(state_total["Delta_Recovered"]).drop(state_total["Delta_Deaths"]).drop(state_total["State_Notes"])
state_df=state_df1.drop(state_df1["Delta_Confirmed"]).drop(state_df1["Delta_Recovered"]).drop(state_df1["Delta_Deaths"]).drop(state_df1["State_Notes"])

# COMMAND ----------

state_total1=state_total.withColumn('tdeath rate',(state_total['Deaths']/state_total['Confirmed'])*100)
state_total2=state_total1.withColumn('tactive rate',(state_total['Active']/state_total['Confirmed'])*100)
state_total3=state_total2.withColumn('trecovery rate',(state_total['Recovered']/state_total['Confirmed'])*100)
display(state_total3)

# COMMAND ----------

from pyspark.sql.functions import lit
state_df=state_df.withColumn('death rate',(state_df['Deaths']/state_df['Confirmed'])*100)
state_df=state_df.withColumn('active rate',(state_df['Active']/state_df['Confirmed'])*100)
state_df=state_df.withColumn('recovery rate',(state_df['Recovered']/state_df['Confirmed'])*100)
state_df=state_df.filter(state_df['Confirmed']!=0)
state_df=state_df.withColumn('national death rate',lit(state_total3.select('tdeath rate').first()['tdeath rate']))
state_df=state_df.withColumn('national active rate',lit(state_total3.select('tactive rate').first()['tactive rate']))
state_df=state_df.withColumn('national recovery rate',lit(state_total3.select('trecovery rate').first()['trecovery rate']))
display(state_df)

# COMMAND ----------

# MAGIC %md  ###Sate wise Death comparision with national average

# COMMAND ----------

import plotly.graph_objects as go
a=state_df.toPandas()
fig = go.Figure()
fig.add_trace(go.Scatter(x=a['State'], y=a['death rate'],mode='lines+markers',name='state death rate'))
fig.add_trace(go.Scatter(x=a['State'], y=a['national death rate'],mode='lines',name='national death rate'))
displayHTML(plot(fig,output_type='div'))

# COMMAND ----------

dt=df.groupby('State_or_UT').agg({'Death': 'max'}).sort('max(Death)', ascending=False)
dt.toPandas()

# COMMAND ----------

file_location = "/FileStore/tables/ts.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df99 = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)



df99=df99.withColumnRenamed("Country/Region","country").withColumnRenamed("Province/State","state")

df99.createOrReplaceTempView("world")
display(df99)

# COMMAND ----------


px.set_mapbox_access_token("pk.eyJ1Ijoia3JhbTEyMyIsImEiOiJjazlheGlrazIwYWlxM3Jsa3g1Mmk1ZWp6In0.aqT-edMtVi9If2A70LG37w")

# COMMAND ----------

spark.sql("select country,max(Deaths) from world group by country").collect()

# COMMAND ----------

# MAGIC %md
# MAGIC #How Corona has effect our World

# COMMAND ----------

dfw=spark.sql("select country,(max(Confirmed)-max(Deaths)) as active,max(Lat) as lat,max(Long) as lon,max(Deaths) as death from world group by country").toPandas()
fig = px.scatter_mapbox(dfw, lat="lat", lon="lon",hover_data=["active","death","country"],color="active", size=(dfw.death*50),hover_name="country",color_continuous_scale=px.colors.cyclical.IceFire,size_max=100, zoom=3)

displayHTML(plot(fig,output_type='div'))

# COMMAND ----------

fig = px.scatter_geo(dfw, lat="lat",lon="lon",color="country",hover_data=["active","death","country"],
                  projection="orthographic",size="active",size_max=30,)

# COMMAND ----------

figdes = px.density_mapbox(dfw, lat='lat', lon='lon', z='active', radius=20,
                        center=dict(lat=0, lon=180), zoom=0,
                        mapbox_style="stamen-terrain")

displayHTML(plot(figdes,output_type='div'))
