# My Task Segmenting and Clustering Neighborhoods in Toronto

## Step 1 - Install and Import Libraries

| Library | Description |
| ----------- | ----------- |
| numpy | NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.|
| pandas | In computer programming, pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series. |
| matplotlib | Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.|
| json | json exposes an API familiar to users of the standard library marshal and pickle modules. |
| lxml | lxml is the most feature-rich and easy-to-use library for processing XML and HTML in the Python language. |
| geopy | geopy is a Python client for several popular geocoding web services. The geopy makes it easy for Python developers to locate the coordinates of addresses, cities, countries, and landmarks across the globe using third-party geocoders and other data sources. |
| requests | Requests is an elegant and simple HTTP library for Python, built for human beings. |
| sklearn | Scikit-learn is an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection and evaluation, and many other utilities. |
| bs4 | Beautiful Soup is a Python library for pulling data out of HTML and XML files. It works with your favorite parser to provide idiomatic ways of navigating, searching, and modifying the parse tree. It commonly saves programmers hours or days of work. |
| warnings | Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program. For example, one might want to issue a warning when a program uses an obsolete module. |


```python
## My Installs
!pip install lxml
!pip install geopy
!pip install beautifulsoup4
!pip install folium
## My Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import lxml
from geopy.geocoders import Nominatim
import requests
from pandas.io.json import json_normalize
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
import folium
import warnings
warnings.filterwarnings('ignore')
```

    Requirement already satisfied: lxml in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (4.6.1)
    Requirement already satisfied: geopy in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (2.0.0)
    Requirement already satisfied: geographiclib<2,>=1.49 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from geopy) (1.50)
    Requirement already satisfied: beautifulsoup4 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (4.9.3)
    Requirement already satisfied: soupsieve>1.2; python_version >= "3.0" in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from beautifulsoup4) (2.0.1)
    Requirement already satisfied: folium in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (0.5.0)
    Requirement already satisfied: branca in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from folium) (0.4.1)
    Requirement already satisfied: jinja2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from folium) (2.11.2)
    Requirement already satisfied: requests in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from folium) (2.24.0)
    Requirement already satisfied: six in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from folium) (1.15.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from jinja2->folium) (1.1.1)
    Requirement already satisfied: idna<3,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->folium) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->folium) (2020.6.20)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->folium) (1.25.10)
    Requirement already satisfied: chardet<4,>=3.0.2 in /home/jupyterlab/conda/envs/python/lib/python3.6/site-packages (from requests->folium) (3.0.4)


## Step 2 - Reading Datas on Page Wikipedia


```python
origin = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
source = requests.get(origin).text
soup = BeautifulSoup(source)

table_data = soup.find('div', class_='mw-parser-output')
table = table_data.table.tbody

columns = ['PostalCode', 'Borough', 'Neighbourhood']
data = dict({key:[] * len(columns) for key in columns})

for row in table.find_all('tr'):
    for i,column in zip(row.find_all('td'),columns):
        i = i.text
        i = i.replace('\n', '')
        data[column].append(i)

df = pd.DataFrame.from_dict(data = data)[columns]
print('Original Data Frame --> Shape is: ', df.shape)
df.head(10)
```

    Original Data Frame --> Shape is:  (180, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M2A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M8A</td>
      <td>Not assigned</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue, Humber Valley Village</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3 - Cleaning Data Frame


```python
df = df[df.Borough != 'Not assigned'].reset_index(drop = True)
print('After Clean Data Frame --> Shape is: ', df.shape)
df.head(10)
```

    After Clean Data Frame --> Shape is:  (103, 3)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue, Humber Valley Village</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Parkview Hill, Woodbine Gardens</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
    </tr>
  </tbody>
</table>
</div>



## Step 4 - Reading Datas on File CSV


```python
lat_lon = pd.read_csv('https://cocl.us/Geospatial_data')
lat_lon.rename(columns = {'Postal Code':'PostalCode'}, inplace = True) ## Preparing Column for Merge
lat_lon.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M1G</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M1H</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M1J</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1K</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M1L</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1M</td>
      <td>43.716316</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1N</td>
      <td>43.692657</td>
      <td>-79.264848</td>
    </tr>
  </tbody>
</table>
</div>



## Step 5 - Merging Tables - Creating Final Dataset


```python
ds = pd.merge(df,lat_lon, on = 'PostalCode')
ds.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.753259</td>
      <td>-79.329656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td>43.725882</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
      <td>43.718518</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue, Humber Valley Village</td>
      <td>43.667856</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills</td>
      <td>43.745906</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Parkview Hill, Woodbine Gardens</td>
      <td>43.706397</td>
      <td>-79.309937</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M6B</td>
      <td>North York</td>
      <td>Glencairn</td>
      <td>43.709577</td>
      <td>-79.445073</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M9B</td>
      <td>Etobicoke</td>
      <td>West Deane Park, Princess Gardens, Martin Grov...</td>
      <td>43.650943</td>
      <td>-79.554724</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M1C</td>
      <td>Scarborough</td>
      <td>Rouge Hill, Port Union, Highland Creek</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M3C</td>
      <td>North York</td>
      <td>Don Mills</td>
      <td>43.725900</td>
      <td>-79.340923</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M4C</td>
      <td>East York</td>
      <td>Woodbine Heights</td>
      <td>43.695344</td>
      <td>-79.318389</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M6C</td>
      <td>York</td>
      <td>Humewood-Cedarvale</td>
      <td>43.693781</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M9C</td>
      <td>Etobicoke</td>
      <td>Eringate, Bloordale Gardens, Old Burnhamthorpe...</td>
      <td>43.643515</td>
      <td>-79.577201</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M1E</td>
      <td>Scarborough</td>
      <td>Guildwood, Morningside, West Hill</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
    <tr>
      <th>19</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.676357</td>
      <td>-79.293031</td>
    </tr>
    <tr>
      <th>20</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <th>21</th>
      <td>M6E</td>
      <td>York</td>
      <td>Caledonia-Fairbanks</td>
      <td>43.689026</td>
      <td>-79.453512</td>
    </tr>
    <tr>
      <th>22</th>
      <td>M1G</td>
      <td>Scarborough</td>
      <td>Woburn</td>
      <td>43.770992</td>
      <td>-79.216917</td>
    </tr>
    <tr>
      <th>23</th>
      <td>M4G</td>
      <td>East York</td>
      <td>Leaside</td>
      <td>43.709060</td>
      <td>-79.363452</td>
    </tr>
    <tr>
      <th>24</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <th>25</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <th>26</th>
      <td>M1H</td>
      <td>Scarborough</td>
      <td>Cedarbrae</td>
      <td>43.773136</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>27</th>
      <td>M2H</td>
      <td>North York</td>
      <td>Hillcrest Village</td>
      <td>43.803762</td>
      <td>-79.363452</td>
    </tr>
    <tr>
      <th>28</th>
      <td>M3H</td>
      <td>North York</td>
      <td>Bathurst Manor, Wilson Heights, Downsview North</td>
      <td>43.754328</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>29</th>
      <td>M4H</td>
      <td>East York</td>
      <td>Thorncliffe Park</td>
      <td>43.705369</td>
      <td>-79.349372</td>
    </tr>
    <tr>
      <th>30</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Richmond, Adelaide, King</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <th>31</th>
      <td>M6H</td>
      <td>West Toronto</td>
      <td>Dufferin, Dovercourt Village</td>
      <td>43.669005</td>
      <td>-79.442259</td>
    </tr>
    <tr>
      <th>32</th>
      <td>M1J</td>
      <td>Scarborough</td>
      <td>Scarborough Village</td>
      <td>43.744734</td>
      <td>-79.239476</td>
    </tr>
    <tr>
      <th>33</th>
      <td>M2J</td>
      <td>North York</td>
      <td>Fairview, Henry Farm, Oriole</td>
      <td>43.778517</td>
      <td>-79.346556</td>
    </tr>
    <tr>
      <th>34</th>
      <td>M3J</td>
      <td>North York</td>
      <td>Northwood Park, York University</td>
      <td>43.767980</td>
      <td>-79.487262</td>
    </tr>
    <tr>
      <th>35</th>
      <td>M4J</td>
      <td>East York</td>
      <td>East Toronto, Broadview North (Old East York)</td>
      <td>43.685347</td>
      <td>-79.338106</td>
    </tr>
    <tr>
      <th>36</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <th>37</th>
      <td>M6J</td>
      <td>West Toronto</td>
      <td>Little Portugal, Trinity</td>
      <td>43.647927</td>
      <td>-79.419750</td>
    </tr>
    <tr>
      <th>38</th>
      <td>M1K</td>
      <td>Scarborough</td>
      <td>Kennedy Park, Ionview, East Birchmount Park</td>
      <td>43.727929</td>
      <td>-79.262029</td>
    </tr>
    <tr>
      <th>39</th>
      <td>M2K</td>
      <td>North York</td>
      <td>Bayview Village</td>
      <td>43.786947</td>
      <td>-79.385975</td>
    </tr>
    <tr>
      <th>40</th>
      <td>M3K</td>
      <td>North York</td>
      <td>Downsview</td>
      <td>43.737473</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>41</th>
      <td>M4K</td>
      <td>East Toronto</td>
      <td>The Danforth West, Riverdale</td>
      <td>43.679557</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>42</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
    <tr>
      <th>43</th>
      <td>M6K</td>
      <td>West Toronto</td>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>43.636847</td>
      <td>-79.428191</td>
    </tr>
    <tr>
      <th>44</th>
      <td>M1L</td>
      <td>Scarborough</td>
      <td>Golden Mile, Clairlea, Oakridge</td>
      <td>43.711112</td>
      <td>-79.284577</td>
    </tr>
    <tr>
      <th>45</th>
      <td>M2L</td>
      <td>North York</td>
      <td>York Mills, Silver Hills</td>
      <td>43.757490</td>
      <td>-79.374714</td>
    </tr>
    <tr>
      <th>46</th>
      <td>M3L</td>
      <td>North York</td>
      <td>Downsview</td>
      <td>43.739015</td>
      <td>-79.506944</td>
    </tr>
    <tr>
      <th>47</th>
      <td>M4L</td>
      <td>East Toronto</td>
      <td>India Bazaar, The Beaches West</td>
      <td>43.668999</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>48</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>43.648198</td>
      <td>-79.379817</td>
    </tr>
    <tr>
      <th>49</th>
      <td>M6L</td>
      <td>North York</td>
      <td>North Park, Maple Leaf Park, Upwood Park</td>
      <td>43.713756</td>
      <td>-79.490074</td>
    </tr>
  </tbody>
</table>
</div>



## Step 6 - Preparing Location


```python
## Location: Toronto, Ontario
address_toronto = 'Toronto, Ontario'
geolocator_toronto = Nominatim(user_agent="ny_explorer")
location_toronto = geolocator_toronto.geocode(address_toronto)
latitude_toronto = location_toronto.latitude
longitude_toronto = location_toronto.longitude
print('The geograpical coordinate of Toronto, Ontario are {}, {}.'.format(latitude_toronto, longitude_toronto))

## Location: Downtown Toronto ,Toronto, Ontario
address_downtown = 'Downtown Toronto ,Toronto, Ontario'
geolocator_downtown = Nominatim(user_agent = "ny_explorer")
location_downtown = geolocator_downtown.geocode(address_downtown)
latitude_downtown = location_downtown.latitude
longitude_downtown = location_downtown.longitude
print('The geograpical coordinate of Downtown Toronto ,Toronto, Ontario are {}, {}.'.format(latitude_downtown, longitude_downtown))
```

    The geograpical coordinate of Toronto, Ontario are 43.6534817, -79.3839347.
    The geograpical coordinate of Downtown Toronto ,Toronto, Ontario are 43.6563221, -79.3809161.


## Step 7 - Show Maps with Folium

### Map Toronto


```python
map_toronto = folium.Map(location = [latitude_toronto, longitude_toronto], zoom_start = 10)

# add markers to map
for lat, lng, borough, neighborhood in zip(ds['Latitude'], ds['Longitude'], ds['Borough'], ds['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDAgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfNzY0NThjMjY4ZTg0NGE0MWI3MTYwOGNmNTM3OWFlNGQgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUyZTUxNDk1YTZhNzQ0ZTA5YmNmOTM4Y2EyZjZhMGFhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUzMjU4NiwtNzkuMzI5NjU2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZjNhMmU3MTJiODQ0YmFjYjg4NjI1NTdmOTc2NGE4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hOWUwODg2MDQyN2E0OGMzOTQ1YTM5MDUyOThkZWJhNSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTllMDg4NjA0MjdhNDhjMzk0NWEzOTA1Mjk4ZGViYTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmt3b29kcywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmYzYTJlNzEyYjg0NGJhY2I4ODYyNTU3Zjk3NjRhODEuc2V0Q29udGVudChodG1sX2E5ZTA4ODYwNDI3YTQ4YzM5NDVhMzkwNTI5OGRlYmE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUyZTUxNDk1YTZhNzQ0ZTA5YmNmOTM4Y2EyZjZhMGFhLmJpbmRQb3B1cChwb3B1cF9iZjNhMmU3MTJiODQ0YmFjYjg4NjI1NTdmOTc2NGE4MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lOTRlOGRiYmFkYmM0NzgwOTVjYzE3OGZhNzI2NzNhYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg4MjI5OTk5OTk5NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2JlY2QzNjJlMDQ3NDlkYzg4YzE3MmM3OTQ3NmVlYWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmFjNTBkZTQ2ODE0NGE1NGE1NWNkY2UxNzUzZDQ4YzcgPSAkKCc8ZGl2IGlkPSJodG1sX2JhYzUwZGU0NjgxNDRhNTRhNTVjZGNlMTc1M2Q0OGM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5WaWN0b3JpYSBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zYmVjZDM2MmUwNDc0OWRjODhjMTcyYzc5NDc2ZWVhYy5zZXRDb250ZW50KGh0bWxfYmFjNTBkZTQ2ODE0NGE1NGE1NWNkY2UxNzUzZDQ4YzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTk0ZThkYmJhZGJjNDc4MDk1Y2MxNzhmYTcyNjczYWIuYmluZFBvcHVwKHBvcHVwXzNiZWNkMzYyZTA0NzQ5ZGM4OGMxNzJjNzk0NzZlZWFjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ2OWY2ZWYxODAwNTQ4MGNiYzczNTgxMzE2ZmUyZTRmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iYzNkMjkyZTNlZWM0M2IwOTE2MjZiMjZjYmQyN2Y2NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYzRmYTBjMzA3NDg0NWQ3ODM5M2E3MTYyYjYwODRlYiA9ICQoJzxkaXYgaWQ9Imh0bWxfZWM0ZmEwYzMwNzQ4NDVkNzgzOTNhNzE2MmI2MDg0ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZ2VudCBQYXJrLCBIYXJib3VyZnJvbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JjM2QyOTJlM2VlYzQzYjA5MTYyNmIyNmNiZDI3ZjY2LnNldENvbnRlbnQoaHRtbF9lYzRmYTBjMzA3NDg0NWQ3ODM5M2E3MTYyYjYwODRlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NjlmNmVmMTgwMDU0ODBjYmM3MzU4MTMxNmZlMmU0Zi5iaW5kUG9wdXAocG9wdXBfYmMzZDI5MmUzZWVjNDNiMDkxNjI2YjI2Y2JkMjdmNjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTAzOWY4M2ZkMTQwNDAwYWIwY2NjMjhlODAyMDIwOWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTg1MTc5OTk5OTk5OTYsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I3ZTBhNGEzNWUxNTRkYjRhYzJkNTgzNzhmOGJlNmFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FlZGNkZjEyM2JkYjQ3NzBiODJiMmIxMDc3YTY1ODdlID0gJCgnPGRpdiBpZD0iaHRtbF9hZWRjZGYxMjNiZGI0NzcwYjgyYjJiMTA3N2E2NTg3ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgTWFub3IsIExhd3JlbmNlIEhlaWdodHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I3ZTBhNGEzNWUxNTRkYjRhYzJkNTgzNzhmOGJlNmFkLnNldENvbnRlbnQoaHRtbF9hZWRjZGYxMjNiZGI0NzcwYjgyYjJiMTA3N2E2NTg3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMDM5ZjgzZmQxNDA0MDBhYjBjY2MyOGU4MDIwMjA5ZC5iaW5kUG9wdXAocG9wdXBfYjdlMGE0YTM1ZTE1NGRiNGFjMmQ1ODM3OGY4YmU2YWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTljNjgzMDI5YmIxNDNkODk1OGY4ODc0MTkwMWJkODYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjIzMDE1LC03OS4zODk0OTM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRjYzAwNWY4YmVmNjQ4NzM4NGE4MjUxNTJkNWM4NDFhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBiMjJhOTUxYTNlZDQ4ZWQ4NzM0ODAyMDViOGZmY2NkID0gJCgnPGRpdiBpZD0iaHRtbF8wYjIyYTk1MWEzZWQ0OGVkODczNDgwMjA1YjhmZmNjZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UXVlZW4mIzM5O3MgUGFyaywgT250YXJpbyBQcm92aW5jaWFsIEdvdmVybm1lbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRjYzAwNWY4YmVmNjQ4NzM4NGE4MjUxNTJkNWM4NDFhLnNldENvbnRlbnQoaHRtbF8wYjIyYTk1MWEzZWQ0OGVkODczNDgwMjA1YjhmZmNjZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lOWM2ODMwMjliYjE0M2Q4OTU4Zjg4NzQxOTAxYmQ4Ni5iaW5kUG9wdXAocG9wdXBfNGNjMDA1ZjhiZWY2NDg3Mzg0YTgyNTE1MmQ1Yzg0MWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODRkYmNlMWZhZDJkNGFmOTg3MWMzYzZmOTRhNmQwZjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc4NTU2LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOGJhMmU3ZGNlNWM0ZGU1YjAzOTkwZWQ3ZTljOGE5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMTZmOWUyZGI0NWQ0MTNjYjJiNDhlNjYyZDc3ZDMzZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZjE2ZjllMmRiNDVkNDEzY2IyYjQ4ZTY2MmQ3N2QzM2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPklzbGluZ3RvbiBBdmVudWUsIEh1bWJlciBWYWxsZXkgVmlsbGFnZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yOGJhMmU3ZGNlNWM0ZGU1YjAzOTkwZWQ3ZTljOGE5Ny5zZXRDb250ZW50KGh0bWxfZjE2ZjllMmRiNDVkNDEzY2IyYjQ4ZTY2MmQ3N2QzM2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODRkYmNlMWZhZDJkNGFmOTg3MWMzYzZmOTRhNmQwZjEuYmluZFBvcHVwKHBvcHVwXzI4YmEyZTdkY2U1YzRkZTViMDM5OTBlZDdlOWM4YTk3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M4NDA1NTc5NjE2ZDQyZDFiYmRhZGVkNjNmM2IzMmZhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMzhlMTg1Y2YwYjU0ZTZmYTFjMDM1MGZkNGMzNGE2YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xYmExMGRjMDI3N2Y0MGIzOTg2OGNjODVlMDNhODgzMiA9ICQoJzxkaXYgaWQ9Imh0bWxfMWJhMTBkYzAyNzdmNDBiMzk4NjhjYzg1ZTAzYTg4MzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hbHZlcm4sIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzM4ZTE4NWNmMGI1NGU2ZmExYzAzNTBmZDRjMzRhNmEuc2V0Q29udGVudChodG1sXzFiYTEwZGMwMjc3ZjQwYjM5ODY4Y2M4NWUwM2E4ODMyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M4NDA1NTc5NjE2ZDQyZDFiYmRhZGVkNjNmM2IzMmZhLmJpbmRQb3B1cChwb3B1cF8zMzhlMTg1Y2YwYjU0ZTZmYTFjMDM1MGZkNGMzNGE2YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NWQ4OTVmNjc0ZDg0ZjNmOTM3ODI3YTNiYmI1YjUxMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc0NTkwNTc5OTk5OTk5NiwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcwZTk5NmJiNTJkODQ1OGY4NDg3YjllY2RmMDcxOTdjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M1Mzk3NDUzMDk1YTQ2YjVhYWNmZTJjOWQ4NTcxZTMyID0gJCgnPGRpdiBpZD0iaHRtbF9jNTM5NzQ1MzA5NWE0NmI1YWFjZmUyYzlkODU3MWUzMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MGU5OTZiYjUyZDg0NThmODQ4N2I5ZWNkZjA3MTk3Yy5zZXRDb250ZW50KGh0bWxfYzUzOTc0NTMwOTVhNDZiNWFhY2ZlMmM5ZDg1NzFlMzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzVkODk1ZjY3NGQ4NGYzZjkzNzgyN2EzYmJiNWI1MTIuYmluZFBvcHVwKHBvcHVwXzcwZTk5NmJiNTJkODQ1OGY4NDg3YjllY2RmMDcxOTdjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E5YzE4MDBlMDlkNDRjMTQ4ODlkMmM3MDNiYzViMjc4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2Mzk3MiwtNzkuMzA5OTM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZhM2RhZjk0ZDQwYjRjY2Q4MTljZjAwNTRjNzZkMjczID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYyZDVjNjdhMzZjZTRjNTU5ZGVlMjg4MzY1ZDdiMGE4ID0gJCgnPGRpdiBpZD0iaHRtbF82MmQ1YzY3YTM2Y2U0YzU1OWRlZTI4ODM2NWQ3YjBhOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3ZpZXcgSGlsbCwgV29vZGJpbmUgR2FyZGVucywgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YTNkYWY5NGQ0MGI0Y2NkODE5Y2YwMDU0Yzc2ZDI3My5zZXRDb250ZW50KGh0bWxfNjJkNWM2N2EzNmNlNGM1NTlkZWUyODgzNjVkN2IwYTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTljMTgwMGUwOWQ0NGMxNDg4OWQyYzcwM2JjNWIyNzguYmluZFBvcHVwKHBvcHVwXzZhM2RhZjk0ZDQwYjRjY2Q4MTljZjAwNTRjNzZkMjczKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc4ZWJmOWYzZmJhOTQzY2I4YWI5NmQyMmE0NzNjMTQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGYzMTMwNzc4ZGRjNDI4NThlMGZmZTg3MDI2MTk4N2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGEwNGE1ZGQ0MmFlNDIyMmI0NzFlZTY2NGUxYWFjZDIgPSAkKCc8ZGl2IGlkPSJodG1sX2RhMDRhNWRkNDJhZTQyMjJiNDcxZWU2NjRlMWFhY2QyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HYXJkZW4gRGlzdHJpY3QsIFJ5ZXJzb24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhmMzEzMDc3OGRkYzQyODU4ZTBmZmU4NzAyNjE5ODdkLnNldENvbnRlbnQoaHRtbF9kYTA0YTVkZDQyYWU0MjIyYjQ3MWVlNjY0ZTFhYWNkMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83OGViZjlmM2ZiYTk0M2NiOGFiOTZkMjJhNDczYzE0Mi5iaW5kUG9wdXAocG9wdXBfOGYzMTMwNzc4ZGRjNDI4NThlMGZmZTg3MDI2MTk4N2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDQxYTFkNDIxMjRmNDUxYWJkODliMDI4M2NhNzllZDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDk1NzcsLTc5LjQ0NTA3MjU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZiMTI0OWFmNWRmZDQ2YjQ5OTk0MjEwMzRkZWU0YzA4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIzNzQ1MzJkZDYyODQ0N2FiZjIxNzIyMDA4ZWEzNjFjID0gJCgnPGRpdiBpZD0iaHRtbF8yMzc0NTMyZGQ2Mjg0NDdhYmYyMTcyMjAwOGVhMzYxYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2xlbmNhaXJuLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YjEyNDlhZjVkZmQ0NmI0OTk5NDIxMDM0ZGVlNGMwOC5zZXRDb250ZW50KGh0bWxfMjM3NDUzMmRkNjI4NDQ3YWJmMjE3MjIwMDhlYTM2MWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDQxYTFkNDIxMjRmNDUxYWJkODliMDI4M2NhNzllZDQuYmluZFBvcHVwKHBvcHVwXzZiMTI0OWFmNWRmZDQ2YjQ5OTk0MjEwMzRkZWU0YzA4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI5Y2U4ZDczNDU0YzQ3ODc4ZTVjMGQzN2VlMDMyZTc0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwOTQzMiwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDFiMGUxNDIyNjliNGNhZmFiNDkxMzg2Zjk0MzRiYjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDYwNDM5MjZhNGE5NGJiNWE2ZTg0NTUyOGZhNGViM2MgPSAkKCc8ZGl2IGlkPSJodG1sX2Q2MDQzOTI2YTRhOTRiYjVhNmU4NDU1MjhmYTRlYjNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IERlYW5lIFBhcmssIFByaW5jZXNzIEdhcmRlbnMsIE1hcnRpbiBHcm92ZSwgSXNsaW5ndG9uLCBDbG92ZXJkYWxlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QxYjBlMTQyMjY5YjRjYWZhYjQ5MTM4NmY5NDM0YmIzLnNldENvbnRlbnQoaHRtbF9kNjA0MzkyNmE0YTk0YmI1YTZlODQ1NTI4ZmE0ZWIzYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yOWNlOGQ3MzQ1NGM0Nzg3OGU1YzBkMzdlZTAzMmU3NC5iaW5kUG9wdXAocG9wdXBfZDFiMGUxNDIyNjliNGNhZmFiNDkxMzg2Zjk0MzRiYjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWJjMmExMTViMzgwNDc4MTg1YjAzZTBhODk5YzY4NTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODQ1MzUxLC03OS4xNjA0OTcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kOWM1Y2FkNjQ4NjQ0MWNhYmY2Y2Q5ZjI5OGQ4ZjIxZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTE0ZjFlNTQzYWQ0NmIwYmI1NmQ2ODkxZGNiMGE3OCA9ICQoJzxkaXYgaWQ9Imh0bWxfN2ExNGYxZTU0M2FkNDZiMGJiNTZkNjg5MWRjYjBhNzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvdWdlIEhpbGwsIFBvcnQgVW5pb24sIEhpZ2hsYW5kIENyZWVrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDljNWNhZDY0ODY0NDFjYWJmNmNkOWYyOThkOGYyMWUuc2V0Q29udGVudChodG1sXzdhMTRmMWU1NDNhZDQ2YjBiYjU2ZDY4OTFkY2IwYTc4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzliYzJhMTE1YjM4MDQ3ODE4NWIwM2UwYTg5OWM2ODUxLmJpbmRQb3B1cChwb3B1cF9kOWM1Y2FkNjQ4NjQ0MWNhYmY2Y2Q5ZjI5OGQ4ZjIxZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NzZlMGE2MTNmODc0MGVlODk1MDZlNjhlOTViMDY4ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg5OTcwMDAwMDAxLC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDliMjJhOTVmZjY5NGNiZWIzNDg4MjE5ZjJmYTU3NWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNDBjNmUyZjY0ZDNmNDhmOGJhMDdlNzA1M2E5Nzc2MWMgPSAkKCc8ZGl2IGlkPSJodG1sXzQwYzZlMmY2NGQzZjQ4ZjhiYTA3ZTcwNTNhOTc3NjFjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb24gTWlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q5YjIyYTk1ZmY2OTRjYmViMzQ4ODIxOWYyZmE1NzViLnNldENvbnRlbnQoaHRtbF80MGM2ZTJmNjRkM2Y0OGY4YmEwN2U3MDUzYTk3NzYxYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NzZlMGE2MTNmODc0MGVlODk1MDZlNjhlOTViMDY4ZS5iaW5kUG9wdXAocG9wdXBfZDliMjJhOTVmZjY5NGNiZWIzNDg4MjE5ZjJmYTU3NWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWI0MTdlMDNiZTdiNGViZTg3MzM3YjJhYzVmNjc4YTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTUzNDM5MDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzcxZTRmYjRjY2NkNGI1NWIzNjhiN2IxMzZkYzRlZWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTU2NGIzNTJjNDE2NDlmZmJlYzY0MTdiM2EwMmQ0ZWYgPSAkKCc8ZGl2IGlkPSJodG1sXzk1NjRiMzUyYzQxNjQ5ZmZiZWM2NDE3YjNhMDJkNGVmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb29kYmluZSBIZWlnaHRzLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM3MWU0ZmI0Y2NjZDRiNTViMzY4YjdiMTM2ZGM0ZWVjLnNldENvbnRlbnQoaHRtbF85NTY0YjM1MmM0MTY0OWZmYmVjNjQxN2IzYTAyZDRlZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYjQxN2UwM2JlN2I0ZWJlODczMzdiMmFjNWY2NzhhOS5iaW5kUG9wdXAocG9wdXBfMzcxZTRmYjRjY2NkNGI1NWIzNjhiN2IxMzZkYzRlZWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGFhNDM2MjQ2ZDVkNDU4Y2IzNjE4OGE0NWJlN2Q3ZTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y5YTk5ZWRmMGUxNjQ3NmE5ZTgzNTA0ZGMyOGU1MDUwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M0NDJlNDE4MTA4YTRkZDhiYzczZTM0N2ZjODNlMjdjID0gJCgnPGRpdiBpZD0iaHRtbF9jNDQyZTQxODEwOGE0ZGQ4YmM3M2UzNDdmYzgzZTI3YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y5YTk5ZWRmMGUxNjQ3NmE5ZTgzNTA0ZGMyOGU1MDUwLnNldENvbnRlbnQoaHRtbF9jNDQyZTQxODEwOGE0ZGQ4YmM3M2UzNDdmYzgzZTI3Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYWE0MzYyNDZkNWQ0NThjYjM2MTg4YTQ1YmU3ZDdlNi5iaW5kUG9wdXAocG9wdXBfZjlhOTllZGYwZTE2NDc2YTllODM1MDRkYzI4ZTUwNTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWUxZGNlM2Q0OWI1NDQ5YmE2MWNkODZhOGZhYmM2MTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTM3ODEzLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMmI3MzMxMDUwMmI0YjllOTFlNTUzYjE4MjMyOWRjOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NzJhNzk2YzhmZGQ0M2NjYWRhYTFmZjU2ZTlhZmU2NCA9ICQoJzxkaXYgaWQ9Imh0bWxfODcyYTc5NmM4ZmRkNDNjY2FkYWExZmY1NmU5YWZlNjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWV3b29kLUNlZGFydmFsZSwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDJiNzMzMTA1MDJiNGI5ZTkxZTU1M2IxODIzMjlkYzguc2V0Q29udGVudChodG1sXzg3MmE3OTZjOGZkZDQzY2NhZGFhMWZmNTZlOWFmZTY0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFlMWRjZTNkNDliNTQ0OWJhNjFjZDg2YThmYWJjNjExLmJpbmRQb3B1cChwb3B1cF9kMmI3MzMxMDUwMmI0YjllOTFlNTUzYjE4MjMyOWRjOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YWExNjI2YzNhYzU0NDAzOTNmOWU2MDY4ODY4ZDQ5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MzUxNTIsLTc5LjU3NzIwMDc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkwMzc2ZDA2NTQ1ZDQ4NzI4YTY0ZDhmODRhZjQzZmJhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JmZjdkNzA4MzYxNzRjYjU4YzA0M2Q5MmNjZmZkZjhkID0gJCgnPGRpdiBpZD0iaHRtbF9iZmY3ZDcwODM2MTc0Y2I1OGMwNDNkOTJjY2ZmZGY4ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RXJpbmdhdGUsIEJsb29yZGFsZSBHYXJkZW5zLCBPbGQgQnVybmhhbXRob3JwZSwgTWFya2xhbmQgV29vZCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MDM3NmQwNjU0NWQ0ODcyOGE2NGQ4Zjg0YWY0M2ZiYS5zZXRDb250ZW50KGh0bWxfYmZmN2Q3MDgzNjE3NGNiNThjMDQzZDkyY2NmZmRmOGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOWFhMTYyNmMzYWM1NDQwMzkzZjllNjA2ODg2OGQ0OWQuYmluZFBvcHVwKHBvcHVwXzkwMzc2ZDA2NTQ1ZDQ4NzI4YTY0ZDhmODRhZjQzZmJhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVjYmMwZTJkZTUxNDRjOTJhYWE3YTc4MWZhNzIyYmMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzYzNTcyNiwtNzkuMTg4NzExNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNTJlNTdhMTc4ODI0ZjRjODBjNDZjZWI0OTUxOTEwMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yN2VhNWM5N2U2MWI0ZDUyYWIzMzIzZjMxM2RlNjFhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjdlYTVjOTdlNjFiNGQ1MmFiMzMyM2YzMTNkZTYxYTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1aWxkd29vZCwgTW9ybmluZ3NpZGUsIFdlc3QgSGlsbCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM1MmU1N2ExNzg4MjRmNGM4MGM0NmNlYjQ5NTE5MTAwLnNldENvbnRlbnQoaHRtbF8yN2VhNWM5N2U2MWI0ZDUyYWIzMzIzZjMxM2RlNjFhMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81Y2JjMGUyZGU1MTQ0YzkyYWFhN2E3ODFmYTcyMmJjMy5iaW5kUG9wdXAocG9wdXBfMzUyZTU3YTE3ODgyNGY0YzgwYzQ2Y2ViNDk1MTkxMDApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTE1MDEyNjc2NjlhNDA1MThiNzUzNWRjZWFjOGVlMzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOTYyY2M5MzdkNTU0M2E5OTgwNjk2OGU2MWU5MDkyZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85M2U3NWEzNDg0ZDQ0MzI4OWE5ZDZkYmZlZjNmMWMxMSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTNlNzVhMzQ4NGQ0NDMyODlhOWQ2ZGJmZWYzZjFjMTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM5NjJjYzkzN2Q1NTQzYTk5ODA2OTY4ZTYxZTkwOTJlLnNldENvbnRlbnQoaHRtbF85M2U3NWEzNDg0ZDQ0MzI4OWE5ZDZkYmZlZjNmMWMxMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MTUwMTI2NzY2OWE0MDUxOGI3NTM1ZGNlYWM4ZWUzMC5iaW5kUG9wdXAocG9wdXBfMzk2MmNjOTM3ZDU1NDNhOTk4MDY5NjhlNjFlOTA5MmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDllODY2ZjRlNzQ4NGFkNjkxYmU5MmExYjJmZDg2NmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGZhY2VlZDZlYWE0NDkwZDk0MGJlZjk5YWE3ODdiZTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGI1YjNmNDQzYzEzNGExYWE0ZDM0ZTVlNDY0MTAxNWIgPSAkKCc8ZGl2IGlkPSJodG1sXzhiNWIzZjQ0M2MxMzRhMWFhNGQzNGU1ZTQ2NDEwMTViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGZhY2VlZDZlYWE0NDkwZDk0MGJlZjk5YWE3ODdiZTYuc2V0Q29udGVudChodG1sXzhiNWIzZjQ0M2MxMzRhMWFhNGQzNGU1ZTQ2NDEwMTViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA5ZTg2NmY0ZTc0ODRhZDY5MWJlOTJhMWIyZmQ4NjZiLmJpbmRQb3B1cChwb3B1cF84ZmFjZWVkNmVhYTQ0OTBkOTQwYmVmOTlhYTc4N2JlNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YTdkN2RmZjc1NWE0ZmE5OGRkMTc2MjQ3NDk3ZDZlNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4OTAyNTYsLTc5LjQ1MzUxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNDU5OGI4OTM5YzE0Yzk3YjhhYmY0MmJiZWFmYmE1NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMzZmMzM1MzJiMzI0Mzk1YWZjY2YyYjFhMjc1OTYwZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTM2ZjMzNTMyYjMyNDM5NWFmY2NmMmIxYTI3NTk2MGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbGVkb25pYS1GYWlyYmFua3MsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q0NTk4Yjg5MzljMTRjOTdiOGFiZjQyYmJlYWZiYTU0LnNldENvbnRlbnQoaHRtbF9hMzZmMzM1MzJiMzI0Mzk1YWZjY2YyYjFhMjc1OTYwZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81YTdkN2RmZjc1NWE0ZmE5OGRkMTc2MjQ3NDk3ZDZlNi5iaW5kUG9wdXAocG9wdXBfZDQ1OThiODkzOWMxNGM5N2I4YWJmNDJiYmVhZmJhNTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODM2YjJiM2M3Yzg4NGNiY2E3Mjc0ZGZjNDFiMGYxYjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzA5OTIxLC03OS4yMTY5MTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYjFhNjYxOTU3MjI0M2U1YWM4ZmM2ZDZlMWQxMDJkOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZTk3MjI2MDA4NjI0YTY4OGJjZDgxMjE1OWY0YTg2MCA9ICQoJzxkaXYgaWQ9Imh0bWxfOGU5NzIyNjAwODYyNGE2ODhiY2Q4MTIxNTlmNGE4NjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvYnVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFiMWE2NjE5NTcyMjQzZTVhYzhmYzZkNmUxZDEwMmQ5LnNldENvbnRlbnQoaHRtbF84ZTk3MjI2MDA4NjI0YTY4OGJjZDgxMjE1OWY0YTg2MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MzZiMmIzYzdjODg0Y2JjYTcyNzRkZmM0MWIwZjFiNy5iaW5kUG9wdXAocG9wdXBfMWIxYTY2MTk1NzIyNDNlNWFjOGZjNmQ2ZTFkMTAyZDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDNiZWNiYzVhMjkzNDYxZjhlM2Y5MDBhZDk0MGIxZWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDkwNjA0LC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE5M2Y0ZjhjMWNiMzQ2NzQ4N2NkYjA1MjlhM2M2NmJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M3OWNiNGYyZmE4NTQ5MWFhNTY0YmFkNmRmNDQ3N2M2ID0gJCgnPGRpdiBpZD0iaHRtbF9jNzljYjRmMmZhODU0OTFhYTU2NGJhZDZkZjQ0NzdjNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGVhc2lkZSwgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xOTNmNGY4YzFjYjM0Njc0ODdjZGIwNTI5YTNjNjZiYi5zZXRDb250ZW50KGh0bWxfYzc5Y2I0ZjJmYTg1NDkxYWE1NjRiYWQ2ZGY0NDc3YzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDNiZWNiYzVhMjkzNDYxZjhlM2Y5MDBhZDk0MGIxZWYuYmluZFBvcHVwKHBvcHVwXzE5M2Y0ZjhjMWNiMzQ2NzQ4N2NkYjA1MjlhM2M2NmJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg5NmExYjhlMzRmYjQyMzU4YWYyMWViNTM1NTk2ODliID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNTk3YzM3NGM5NzE0OGY4OTIyZmYzMDcwZmJiNWRhZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNjJjNDAwZTZlYzc0YTFlOGFlM2EyNjgxYmU1NThmYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMjYyYzQwMGU2ZWM3NGExZThhZTNhMjY4MWJlNTU4ZmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjU5N2MzNzRjOTcxNDhmODkyMmZmMzA3MGZiYjVkYWQuc2V0Q29udGVudChodG1sXzI2MmM0MDBlNmVjNzRhMWU4YWUzYTI2ODFiZTU1OGZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg5NmExYjhlMzRmYjQyMzU4YWYyMWViNTM1NTk2ODliLmJpbmRQb3B1cChwb3B1cF8yNTk3YzM3NGM5NzE0OGY4OTIyZmYzMDcwZmJiNWRhZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNDJhNzYwOTIyNjY0NDFmODcxNGQ2ZGY2NjIwZTBhYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zY2UyZTVhNDBhNGI0MWQyODAwN2FhZTEyZWUxYWY3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NDY2MWQxZGEwZjc0MWJiYjNkMDZkMWQzNTVjNmFlYSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjQ2NjFkMWRhMGY3NDFiYmIzZDA2ZDFkMzU1YzZhZWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zY2UyZTVhNDBhNGI0MWQyODAwN2FhZTEyZWUxYWY3YS5zZXRDb250ZW50KGh0bWxfNjQ2NjFkMWRhMGY3NDFiYmIzZDA2ZDFkMzU1YzZhZWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzQyYTc2MDkyMjY2NDQxZjg3MTRkNmRmNjYyMGUwYWEuYmluZFBvcHVwKHBvcHVwXzNjZTJlNWE0MGE0YjQxZDI4MDA3YWFlMTJlZTFhZjdhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U0ZTBhYjFiYjE3NjQzYmNiNWNhYTU5ZWMyNmUwMDU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzczMTM2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yZmE1Yzg0Nzg1YjI0M2E4YTljMjVkNGE2MDBiMDc4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NjFhOWY3YmI1YTI0Y2MwODYyMTU3Y2ExMzk5YzZhOCA9ICQoJzxkaXYgaWQ9Imh0bWxfNjYxYTlmN2JiNWEyNGNjMDg2MjE1N2NhMTM5OWM2YTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlZGFyYnJhZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJmYTVjODQ3ODViMjQzYThhOWMyNWQ0YTYwMGIwNzgxLnNldENvbnRlbnQoaHRtbF82NjFhOWY3YmI1YTI0Y2MwODYyMTU3Y2ExMzk5YzZhOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNGUwYWIxYmIxNzY0M2JjYjVjYWE1OWVjMjZlMDA1NC5iaW5kUG9wdXAocG9wdXBfMmZhNWM4NDc4NWIyNDNhOGE5YzI1ZDRhNjAwYjA3ODEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjlhODhhNDZhYjcxNDg1YjllNzY3NGEzZWY1ZTE5MGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MDM3NjIyLC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdhZjYwOWM0NmZmNzQxMTM5YzZmOTc3NjIzZWZmMmQ4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcxYjUyNzZiOGQ3MjRiMjU4MDZiNjQ4ZjYyNGVlMjViID0gJCgnPGRpdiBpZD0iaHRtbF83MWI1Mjc2YjhkNzI0YjI1ODA2YjY0OGY2MjRlZTI1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlsbGNyZXN0IFZpbGxhZ2UsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdhZjYwOWM0NmZmNzQxMTM5YzZmOTc3NjIzZWZmMmQ4LnNldENvbnRlbnQoaHRtbF83MWI1Mjc2YjhkNzI0YjI1ODA2YjY0OGY2MjRlZTI1Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82OWE4OGE0NmFiNzE0ODViOWU3Njc0YTNlZjVlMTkwZS5iaW5kUG9wdXAocG9wdXBfN2FmNjA5YzQ2ZmY3NDExMzljNmY5Nzc2MjNlZmYyZDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzQ4YWE1YThmMDRmNDBhMWFmZmU0MTZhM2FjZjhhNWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTQzMjgzLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk0ZDAwNzhlMzA1YzRhYjhhNzVmZjE3N2FkZDk2MGU5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YxYTgyMjExZWRkYzQ1MWJhZGQzNDIxNzc0YzFjMGRkID0gJCgnPGRpdiBpZD0iaHRtbF9mMWE4MjIxMWVkZGM0NTFiYWRkMzQyMTc3NGMxYzBkZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIFdpbHNvbiBIZWlnaHRzLCBEb3duc3ZpZXcgTm9ydGgsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk0ZDAwNzhlMzA1YzRhYjhhNzVmZjE3N2FkZDk2MGU5LnNldENvbnRlbnQoaHRtbF9mMWE4MjIxMWVkZGM0NTFiYWRkMzQyMTc3NGMxYzBkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNDhhYTVhOGYwNGY0MGExYWZmZTQxNmEzYWNmOGE1Yy5iaW5kUG9wdXAocG9wdXBfOTRkMDA3OGUzMDVjNGFiOGE3NWZmMTc3YWRkOTYwZTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTVlMzExYjc3NmVmNGMwNGI2NDBhYzQ4NTIxZjM4NzkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDUzNjg5LC03OS4zNDkzNzE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZWIxNDFjMzdlZGY0ZGFjOTBmNzY5MjJiOTEwNjc2MCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNjIwYjVmZWEzMjI0MjRkYTU3ZmFlZmE0YjMyYWZjNyA9ICQoJzxkaXYgaWQ9Imh0bWxfYTYyMGI1ZmVhMzIyNDI0ZGE1N2ZhZWZhNGIzMmFmYzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRob3JuY2xpZmZlIFBhcmssIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmViMTQxYzM3ZWRmNGRhYzkwZjc2OTIyYjkxMDY3NjAuc2V0Q29udGVudChodG1sX2E2MjBiNWZlYTMyMjQyNGRhNTdmYWVmYTRiMzJhZmM3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU1ZTMxMWI3NzZlZjRjMDRiNjQwYWM0ODUyMWYzODc5LmJpbmRQb3B1cChwb3B1cF82ZWIxNDFjMzdlZGY0ZGFjOTBmNzY5MjJiOTEwNjc2MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NDczNjJiZjk0NDg0MWM3ODU0NjBhMjc1YzNlYTAxOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM4YzgyOGNkYTA5MTQ2OTViMmFkOWQzMGEyOTdmNjgyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAwZDY5ZWMwYzc3YjQ5M2U4ZGI4M2M3MTQyNGQ0OGY0ID0gJCgnPGRpdiBpZD0iaHRtbF8wMGQ2OWVjMGM3N2I0OTNlOGRiODNjNzE0MjRkNDhmNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaG1vbmQsIEFkZWxhaWRlLCBLaW5nLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOGM4MjhjZGEwOTE0Njk1YjJhZDlkMzBhMjk3ZjY4Mi5zZXRDb250ZW50KGh0bWxfMDBkNjllYzBjNzdiNDkzZThkYjgzYzcxNDI0ZDQ4ZjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODQ3MzYyYmY5NDQ4NDFjNzg1NDYwYTI3NWMzZWEwMTkuYmluZFBvcHVwKHBvcHVwXzM4YzgyOGNkYTA5MTQ2OTViMmFkOWQzMGEyOTdmNjgyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZjNWIyOTY5Zjg0ODQ2YWZhN2JmMzkyMjM0NDY3MDNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmNiNzIxNjhhNDFhNGUzMzkzNWQ4MDcyYjU5NDI3YzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWJlNjI4YmZmYjc3NDg2MjhjMTljZjViYTE1ZWM5NDggPSAkKCc8ZGl2IGlkPSJodG1sXzliZTYyOGJmZmI3NzQ4NjI4YzE5Y2Y1YmExNWVjOTQ4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EdWZmZXJpbiwgRG92ZXJjb3VydCBWaWxsYWdlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZjYjcyMTY4YTQxYTRlMzM5MzVkODA3MmI1OTQyN2MzLnNldENvbnRlbnQoaHRtbF85YmU2MjhiZmZiNzc0ODYyOGMxOWNmNWJhMTVlYzk0OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YzViMjk2OWY4NDg0NmFmYTdiZjM5MjIzNDQ2NzAzYy5iaW5kUG9wdXAocG9wdXBfNmNiNzIxNjhhNDFhNGUzMzkzNWQ4MDcyYjU5NDI3YzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmQ3ZjNjMjAyZmNmNGQ1ZjlmMGFhMjk5ZGU2MjkwY2YgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NDQ3MzQyLC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83N2JlNmMyZDYxZGU0NTg5ODc0OWRjZTQzYzhlYjU3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYzdjYzBhZjlhODU0Mjk0YTA5ZTMxODUzMDMwYzhlYiA9ICQoJzxkaXYgaWQ9Imh0bWxfYmM3Y2MwYWY5YTg1NDI5NGEwOWUzMTg1MzAzMGM4ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNjYXJib3JvdWdoIFZpbGxhZ2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83N2JlNmMyZDYxZGU0NTg5ODc0OWRjZTQzYzhlYjU3YS5zZXRDb250ZW50KGh0bWxfYmM3Y2MwYWY5YTg1NDI5NGEwOWUzMTg1MzAzMGM4ZWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmQ3ZjNjMjAyZmNmNGQ1ZjlmMGFhMjk5ZGU2MjkwY2YuYmluZFBvcHVwKHBvcHVwXzc3YmU2YzJkNjFkZTQ1ODk4NzQ5ZGNlNDNjOGViNTdhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlhM2RjNTRlNjk4YjQ5Mjg5MzY1MTA2MDlmMDIxMzE2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzc4NTE3NSwtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZTUzMWUyZjdjNDg0YTMxODA3ZjkxNDRhNGY3YTNmNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lODBmMzBmYzI5OTQ0NjdlYmU1NTc3ZDAwYWM2Y2E4OSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTgwZjMwZmMyOTk0NDY3ZWJlNTU3N2QwMGFjNmNhODkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZhaXJ2aWV3LCBIZW5yeSBGYXJtLCBPcmlvbGUsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllNTMxZTJmN2M0ODRhMzE4MDdmOTE0NGE0ZjdhM2Y3LnNldENvbnRlbnQoaHRtbF9lODBmMzBmYzI5OTQ0NjdlYmU1NTc3ZDAwYWM2Y2E4OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YTNkYzU0ZTY5OGI0OTI4OTM2NTEwNjA5ZjAyMTMxNi5iaW5kUG9wdXAocG9wdXBfOWU1MzFlMmY3YzQ4NGEzMTgwN2Y5MTQ0YTRmN2EzZjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmI2MWQ4YWI0YWE0NDhmYWEzYzZkMTExNDFlOTdiMmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Njc5ODAzLC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZTRjZGIwNDkzZmE0MzYxYTAzM2YwYTg1YTExNWQzZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NThkYTc5NzJhZDg0ZmVlYjc5ZDA0OTA4NWRjYTBiMyA9ICQoJzxkaXYgaWQ9Imh0bWxfODU4ZGE3OTcyYWQ4NGZlZWI3OWQwNDkwODVkY2EwYjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod29vZCBQYXJrLCBZb3JrIFVuaXZlcnNpdHksIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhlNGNkYjA0OTNmYTQzNjFhMDMzZjBhODVhMTE1ZDNmLnNldENvbnRlbnQoaHRtbF84NThkYTc5NzJhZDg0ZmVlYjc5ZDA0OTA4NWRjYTBiMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYjYxZDhhYjRhYTQ0OGZhYTNjNmQxMTE0MWU5N2IyYS5iaW5kUG9wdXAocG9wdXBfOGU0Y2RiMDQ5M2ZhNDM2MWEwMzNmMGE4NWExMTVkM2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWE3OWQzMWY5MmY5NDAwOGFlOWQzMmMwNzE2YTQ5NTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODUzNDcsLTc5LjMzODEwNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmRjYWMxMGU4YjQ3NGM5ZTg3NDFlN2UxNTBkNDQ2MjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDU1M2ZiMTRiOTgzNDdlM2E4Y2M3YWJjYTQ2NjdmNDAgPSAkKCc8ZGl2IGlkPSJodG1sX2Q1NTNmYjE0Yjk4MzQ3ZTNhOGNjN2FiY2E0NjY3ZjQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8sIEJyb2FkdmlldyBOb3J0aCAoT2xkIEVhc3QgWW9yayksIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmRjYWMxMGU4YjQ3NGM5ZTg3NDFlN2UxNTBkNDQ2MjUuc2V0Q29udGVudChodG1sX2Q1NTNmYjE0Yjk4MzQ3ZTNhOGNjN2FiY2E0NjY3ZjQwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVhNzlkMzFmOTJmOTQwMDhhZTlkMzJjMDcxNmE0OTU2LmJpbmRQb3B1cChwb3B1cF9iZGNhYzEwZThiNDc0YzllODc0MWU3ZTE1MGQ0NDYyNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yM2E4YTkzOGIwOGU0NGEyOGY4MzVlMmI3NDNlZGE0MiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc1YmU3MTAxMjIzZTRmOWFhYmRjZTg0ODRlODllNmQ4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQwN2QzNzYyOTU5OTRiNDc4MDg5ZWZlMjdkZDNkNGU5ID0gJCgnPGRpdiBpZD0iaHRtbF80MDdkMzc2Mjk1OTk0YjQ3ODA4OWVmZTI3ZGQzZDRlOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIFVuaW9uIFN0YXRpb24sIFRvcm9udG8gSXNsYW5kcywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzViZTcxMDEyMjNlNGY5YWFiZGNlODQ4NGU4OWU2ZDguc2V0Q29udGVudChodG1sXzQwN2QzNzYyOTU5OTRiNDc4MDg5ZWZlMjdkZDNkNGU5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIzYThhOTM4YjA4ZTQ0YTI4ZjgzNWUyYjc0M2VkYTQyLmJpbmRQb3B1cChwb3B1cF83NWJlNzEwMTIyM2U0ZjlhYWJkY2U4NDg0ZTg5ZTZkOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82N2FhNTU4ZGQ1YzE0NmZlODQ1ZjU4ZjcwODEzMmRlYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wZmE1MzExZTFlZjE0MmY4YTVlZGQ0Y2UyNjYzMWFkMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZDFiZjk5YTY0MDE0ZjU2YWJmMGMzOGJlYzZjYjk3YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZmQxYmY5OWE2NDAxNGY1NmFiZjBjMzhiZWM2Y2I5N2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZmE1MzExZTFlZjE0MmY4YTVlZGQ0Y2UyNjYzMWFkMi5zZXRDb250ZW50KGh0bWxfZmQxYmY5OWE2NDAxNGY1NmFiZjBjMzhiZWM2Y2I5N2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjdhYTU1OGRkNWMxNDZmZTg0NWY1OGY3MDgxMzJkZWIuYmluZFBvcHVwKHBvcHVwXzBmYTUzMTFlMWVmMTQyZjhhNWVkZDRjZTI2NjMxYWQyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UzMzMxNjExZDgyZjQ1Njc5YTM4ZGM5YTcxNzQ2YWM2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI3OTI5MiwtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTNhYTg5NDBhMGJlNGI2YmI2NjU3MGFkNzAwNDY0ZGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjMwMjJiNjY4N2MyNDIzYzljOWVkZWVmMjRjZDA0MTcgPSAkKCc8ZGl2IGlkPSJodG1sX2IzMDIyYjY2ODdjMjQyM2M5YzllZGVlZjI0Y2QwNDE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5uZWR5IFBhcmssIElvbnZpZXcsIEVhc3QgQmlyY2htb3VudCBQYXJrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTNhYTg5NDBhMGJlNGI2YmI2NjU3MGFkNzAwNDY0ZGEuc2V0Q29udGVudChodG1sX2IzMDIyYjY2ODdjMjQyM2M5YzllZGVlZjI0Y2QwNDE3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UzMzMxNjExZDgyZjQ1Njc5YTM4ZGM5YTcxNzQ2YWM2LmJpbmRQb3B1cChwb3B1cF8xM2FhODk0MGEwYmU0YjZiYjY2NTcwYWQ3MDA0NjRkYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kZDRmMmFkOGMzMWQ0ZTdmYjQzNmE3ZGY2ZGUyNTgyNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4Njk0NzMsLTc5LjM4NTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NGI2OThkNGZmMTI0Y2JmOWZmNWY3YmI1ZmMwYzA5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMzVkYTNlZjc5OWQ0MjI1YTE3MTZlMWRmZGMyNTRhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDM1ZGEzZWY3OTlkNDIyNWExNzE2ZTFkZmRjMjU0YTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJheXZpZXcgVmlsbGFnZSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODRiNjk4ZDRmZjEyNGNiZjlmZjVmN2JiNWZjMGMwOTcuc2V0Q29udGVudChodG1sX2QzNWRhM2VmNzk5ZDQyMjVhMTcxNmUxZGZkYzI1NGExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RkNGYyYWQ4YzMxZDRlN2ZiNDM2YTdkZjZkZTI1ODI1LmJpbmRQb3B1cChwb3B1cF84NGI2OThkNGZmMTI0Y2JmOWZmNWY3YmI1ZmMwYzA5Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNzJiMjUyMjYyYzE0Mzg1YTQ4Yjc3Njk3MWRiZDVjMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczNzQ3MzIwMDAwMDAwNCwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzc5NDYzYTllZjQzNDEyODgwMDFkOGIzNTc3MmU4N2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWY3ZjI0NDI3NTk0NGRlZWJkYWZhNWUzNDQ2YTgyMzQgPSAkKCc8ZGl2IGlkPSJodG1sX2FmN2YyNDQyNzU5NDRkZWViZGFmYTVlMzQ0NmE4MjM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM3OTQ2M2E5ZWY0MzQxMjg4MDAxZDhiMzU3NzJlODdjLnNldENvbnRlbnQoaHRtbF9hZjdmMjQ0Mjc1OTQ0ZGVlYmRhZmE1ZTM0NDZhODIzNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNzJiMjUyMjYyYzE0Mzg1YTQ4Yjc3Njk3MWRiZDVjMi5iaW5kUG9wdXAocG9wdXBfMzc5NDYzYTllZjQzNDEyODgwMDFkOGIzNTc3MmU4N2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2RmMGIxMjY2MzM5NGRkM2EzMmUzNDNmNDlmNjUzM2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGM5Yjk2NzhjYmZiNDVlNDlkOWQ0NGVkOWJkNDg0NzkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTZkNWMxY2FlYmJlNGJjZjlmZjYzZDM1MDlhN2M0NDMgPSAkKCc8ZGl2IGlkPSJodG1sXzU2ZDVjMWNhZWJiZTRiY2Y5ZmY2M2QzNTA5YTdjNDQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRjOWI5Njc4Y2JmYjQ1ZTQ5ZDlkNDRlZDliZDQ4NDc5LnNldENvbnRlbnQoaHRtbF81NmQ1YzFjYWViYmU0YmNmOWZmNjNkMzUwOWE3YzQ0Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZGYwYjEyNjYzMzk0ZGQzYTMyZTM0M2Y0OWY2NTMzYS5iaW5kUG9wdXAocG9wdXBfNGM5Yjk2NzhjYmZiNDVlNDlkOWQ0NGVkOWJkNDg0NzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGY1ODRkMmM2NjlhNDFkZjg2NTg3YTUwMjE5MmJmNmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ODkxYmRkYzUyYjY0YjMzYjJiZmNjMTg1Yzg5YzYxNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZGVlMmRmMTIwMjM0ZTFhYWM4ZTRiN2RmODYwMWFlZiA9ICQoJzxkaXYgaWQ9Imh0bWxfYWRlZTJkZjEyMDIzNGUxYWFjOGU0YjdkZjg2MDFhZWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2UsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg4OTFiZGRjNTJiNjRiMzNiMmJmY2MxODVjODljNjE2LnNldENvbnRlbnQoaHRtbF9hZGVlMmRmMTIwMjM0ZTFhYWM4ZTRiN2RmODYwMWFlZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZjU4NGQyYzY2OWE0MWRmODY1ODdhNTAyMTkyYmY2ZC5iaW5kUG9wdXAocG9wdXBfODg5MWJkZGM1MmI2NGIzM2IyYmZjYzE4NWM4OWM2MTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTQ5NTIxN2Y5MjBkNGIyY2E2NTU0MDgxMGJmYmNkYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZGM1ZWQxMjUwZWU0MGFlYjUxNDQyMjY4MmI3Y2Q2MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNWJmMDY4YzliYjQ0NWQ2YTE1Y2E4NWNiZGQxYTQ1MSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTViZjA2OGM5YmI0NDVkNmExNWNhODVjYmRkMWE0NTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkYzVlZDEyNTBlZTQwYWViNTE0NDIyNjgyYjdjZDYxLnNldENvbnRlbnQoaHRtbF9lNWJmMDY4YzliYjQ0NWQ2YTE1Y2E4NWNiZGQxYTQ1MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNDk1MjE3ZjkyMGQ0YjJjYTY1NTQwODEwYmZiY2RhZS5iaW5kUG9wdXAocG9wdXBfZGRjNWVkMTI1MGVlNDBhZWI1MTQ0MjI2ODJiN2NkNjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDZlMDY1YzVlN2FmNDhmOWIwZmQyZDM2MWVmYWFlMmIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTExMTE3MDAwMDAwMDQsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWM2YzFlNTYwYTJjNDU2NmJkNTEwZDYwNzdiNjAyZDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODJjZTZlNmNiMzcyNGExMDljOGUxNWJmZDE3OWYzZTEgPSAkKCc8ZGl2IGlkPSJodG1sXzgyY2U2ZTZjYjM3MjRhMTA5YzhlMTViZmQxNzlmM2UxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Hb2xkZW4gTWlsZSwgQ2xhaXJsZWEsIE9ha3JpZGdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWM2YzFlNTYwYTJjNDU2NmJkNTEwZDYwNzdiNjAyZDQuc2V0Q29udGVudChodG1sXzgyY2U2ZTZjYjM3MjRhMTA5YzhlMTViZmQxNzlmM2UxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ2ZTA2NWM1ZTdhZjQ4ZjliMGZkMmQzNjFlZmFhZTJiLmJpbmRQb3B1cChwb3B1cF8xYzZjMWU1NjBhMmM0NTY2YmQ1MTBkNjA3N2I2MDJkNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YjAzOTJhYjU1ODY0YWNhOWI2ODlmNDhiMzBiZjNkNiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQ5MDIsLTc5LjM3NDcxNDA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBkN2M5MDVkYzU0MjQ4NThiZTU3MTM0ZTBkOGNkYWNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFlNWI2YmZmZWZhMzQyN2VhMTcxNTU5YWY0MTZjMzQ2ID0gJCgnPGRpdiBpZD0iaHRtbF8xZTViNmJmZmVmYTM0MjdlYTE3MTU1OWFmNDE2YzM0NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscywgU2lsdmVyIEhpbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wZDdjOTA1ZGM1NDI0ODU4YmU1NzEzNGUwZDhjZGFjZi5zZXRDb250ZW50KGh0bWxfMWU1YjZiZmZlZmEzNDI3ZWExNzE1NTlhZjQxNmMzNDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2IwMzkyYWI1NTg2NGFjYTliNjg5ZjQ4YjMwYmYzZDYuYmluZFBvcHVwKHBvcHVwXzBkN2M5MDVkYzU0MjQ4NThiZTU3MTM0ZTBkOGNkYWNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQwYzIxY2IyY2IzNTRlYzQ4YmU4YjNkZmI5YzQzYmQ0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5MDE0NiwtNzkuNTA2OTQzNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYjAxNGRhOTg3MjM0N2FlYmI4Y2I3NjIzZjFlNzJhOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMzcwMTI0NGFlNDg0NjExYjk2MjJmZGMwOWJiN2MzMCA9ICQoJzxkaXYgaWQ9Imh0bWxfYTM3MDEyNDRhZTQ4NDYxMWI5NjIyZmRjMDliYjdjMzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2IwMTRkYTk4NzIzNDdhZWJiOGNiNzYyM2YxZTcyYTguc2V0Q29udGVudChodG1sX2EzNzAxMjQ0YWU0ODQ2MTFiOTYyMmZkYzA5YmI3YzMwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQwYzIxY2IyY2IzNTRlYzQ4YmU4YjNkZmI5YzQzYmQ0LmJpbmRQb3B1cChwb3B1cF8zYjAxNGRhOTg3MjM0N2FlYmI4Y2I3NjIzZjFlNzJhOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMmQ3YjAwMTZiMGY0ODcyODJhODVkNTQ3OWQ4MDU0NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QxMzU3Y2I0NDRmOTRhNmNhODQ1MjEyZGUzYjI5NTJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y3ODM4MzM0ODYxNjRhYmE4M2NjN2M5MjRkZDRiZjA5ID0gJCgnPGRpdiBpZD0iaHRtbF9mNzgzODMzNDg2MTY0YWJhODNjYzdjOTI0ZGQ0YmYwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QxMzU3Y2I0NDRmOTRhNmNhODQ1MjEyZGUzYjI5NTJiLnNldENvbnRlbnQoaHRtbF9mNzgzODMzNDg2MTY0YWJhODNjYzdjOTI0ZGQ0YmYwOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMmQ3YjAwMTZiMGY0ODcyODJhODVkNTQ3OWQ4MDU0NC5iaW5kUG9wdXAocG9wdXBfZDEzNTdjYjQ0NGY5NGE2Y2E4NDUyMTJkZTNiMjk1MmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2ViNjM1MTM4ZjEyNDNlNGIwNDYzNjIzMGE0YjIyNGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMGM2MmRjNzhlY2Y0ZWNmOTBmYjI2ZGY1OTQ0NGM4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNTVhMGU2NWIyNDU0MmJiYTZiMzUxZDgxNzFkNWY1MyA9ICQoJzxkaXYgaWQ9Imh0bWxfZTU1YTBlNjViMjQ1NDJiYmE2YjM1MWQ4MTcxZDVmNTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTBjNjJkYzc4ZWNmNGVjZjkwZmIyNmRmNTk0NDRjODIuc2V0Q29udGVudChodG1sX2U1NWEwZTY1YjI0NTQyYmJhNmIzNTFkODE3MWQ1ZjUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdlYjYzNTEzOGYxMjQzZTRiMDQ2MzYyMzBhNGIyMjRlLmJpbmRQb3B1cChwb3B1cF9hMGM2MmRjNzhlY2Y0ZWNmOTBmYjI2ZGY1OTQ0NGM4Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMWQ1YWNmMDViNGU0NzYyODExYmVkNDEzNGQ0NTEwZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMzc1NjIwMDAwMDAwNiwtNzkuNDkwMDczOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZDM0OWE0M2M4NWY0OWEzOTczODA0MmQwMTEwMjZiZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iOTE2MzlmY2Q2YTc0NWQ4ODBiNzJhMzE1ZjQxZTA2YiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjkxNjM5ZmNkNmE3NDVkODgwYjcyYTMxNWY0MWUwNmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRoIFBhcmssIE1hcGxlIExlYWYgUGFyaywgVXB3b29kIFBhcmssIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkMzQ5YTQzYzg1ZjQ5YTM5NzM4MDQyZDAxMTAyNmJlLnNldENvbnRlbnQoaHRtbF9iOTE2MzlmY2Q2YTc0NWQ4ODBiNzJhMzE1ZjQxZTA2Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wMWQ1YWNmMDViNGU0NzYyODExYmVkNDEzNGQ0NTEwZC5iaW5kUG9wdXAocG9wdXBfZGQzNDlhNDNjODVmNDlhMzk3MzgwNDJkMDExMDI2YmUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzQ0MmMwYzhmMzE1NGU0MmFlOTg5N2JlNjFhMzI4ODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTYzMDMzLC03OS41NjU5NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MGFlZTk5YTNkOTA0M2JlYmJhMmQ4NGM0YzFjMDIwNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMTA4MjhiYWQwMDg0ZjYwYTFlMGI5ZDAzN2MzOGQ2MSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDEwODI4YmFkMDA4NGY2MGExZTBiOWQwMzdjMzhkNjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlciBTdW1taXQsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYwYWVlOTlhM2Q5MDQzYmViYmEyZDg0YzRjMWMwMjA1LnNldENvbnRlbnQoaHRtbF9kMTA4MjhiYWQwMDg0ZjYwYTFlMGI5ZDAzN2MzOGQ2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83NDQyYzBjOGYzMTU0ZTQyYWU5ODk3YmU2MWEzMjg4Ny5iaW5kUG9wdXAocG9wdXBfNjBhZWU5OWEzZDkwNDNiZWJiYTJkODRjNGMxYzAyMDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTdmMjkyNzBlM2ZhNGQ0ZTgwZmVkZWE1NmE0NTg4MDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTYzMTYsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhkMTgxMDMzZWY4NDRlMWE4MTY5YjYzNjFhNTVhMWQxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUyMGZiODAyYzk2OTQ4ZWZiZjM5YjY4MTE0MTViZThjID0gJCgnPGRpdiBpZD0iaHRtbF81MjBmYjgwMmM5Njk0OGVmYmYzOWI2ODExNDE1YmU4YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xpZmZzaWRlLCBDbGlmZmNyZXN0LCBTY2FyYm9yb3VnaCBWaWxsYWdlIFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZDE4MTAzM2VmODQ0ZTFhODE2OWI2MzYxYTU1YTFkMS5zZXRDb250ZW50KGh0bWxfNTIwZmI4MDJjOTY5NDhlZmJmMzliNjgxMTQxNWJlOGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTdmMjkyNzBlM2ZhNGQ0ZTgwZmVkZWE1NmE0NTg4MDMuYmluZFBvcHVwKHBvcHVwXzhkMTgxMDMzZWY4NDRlMWE4MTY5YjYzNjFhNTVhMWQxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U3NWY4YjAzYjcwNDRlZjJiYzkwZjM4NGMxY2VlMzk5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg5MDUzLC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81YjNmMjgyNjA3M2M0ZGQ1OWIzMTkyNGI4MmU0Nzk2MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mYjhiYTM5YjhjZjM0MWE5YTNlZmY4ZjRlODEyNTQyMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZmI4YmEzOWI4Y2YzNDFhOWEzZWZmOGY0ZTgxMjU0MjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIE5ld3RvbmJyb29rLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81YjNmMjgyNjA3M2M0ZGQ1OWIzMTkyNGI4MmU0Nzk2My5zZXRDb250ZW50KGh0bWxfZmI4YmEzOWI4Y2YzNDFhOWEzZWZmOGY0ZTgxMjU0MjApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTc1ZjhiMDNiNzA0NGVmMmJjOTBmMzg0YzFjZWUzOTkuYmluZFBvcHVwKHBvcHVwXzViM2YyODI2MDczYzRkZDU5YjMxOTI0YjgyZTQ3OTYzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVmYjVjNjg0MjA0MTRhMDRhMTc5Yjc0NGMzMTJhNTkyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4NDk2NCwtNzkuNDk1Njk3NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDk4NTBhMzIxZGI0NDkzYmFmNjJhMTc1NGZlZDY2NGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTJhNmY2YmRmNTA3NDI2YWE1NzAwOTQzMTU3ODI5YjMgPSAkKCc8ZGl2IGlkPSJodG1sX2UyYTZmNmJkZjUwNzQyNmFhNTcwMDk0MzE1NzgyOWIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA5ODUwYTMyMWRiNDQ5M2JhZjYyYTE3NTRmZWQ2NjRhLnNldENvbnRlbnQoaHRtbF9lMmE2ZjZiZGY1MDc0MjZhYTU3MDA5NDMxNTc4MjliMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZmI1YzY4NDIwNDE0YTA0YTE3OWI3NDRjMzEyYTU5Mi5iaW5kUG9wdXAocG9wdXBfMDk4NTBhMzIxZGI0NDkzYmFmNjJhMTc1NGZlZDY2NGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTk0N2IyYjhhYTFkNDc1Mjk4MTUwZmZjMDhkYWQ2NjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWE4MmY1NDRjMDBhNDljM2IyZjJhYjA4NDU3MGUxZmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTAyYWYyYWI4MzhmNGIxMGFjZTUxMzNlNTQzODMxOWUgPSAkKCc8ZGl2IGlkPSJodG1sXzkwMmFmMmFiODM4ZjRiMTBhY2U1MTMzZTU0MzgzMTllIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZWE4MmY1NDRjMDBhNDljM2IyZjJhYjA4NDU3MGUxZmEuc2V0Q29udGVudChodG1sXzkwMmFmMmFiODM4ZjRiMTBhY2U1MTMzZTU0MzgzMTllKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U5NDdiMmI4YWExZDQ3NTI5ODE1MGZmYzA4ZGFkNjY5LmJpbmRQb3B1cChwb3B1cF9lYTgyZjU0NGMwMGE0OWMzYjJmMmFiMDg0NTcwZTFmYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wOGIzNWIyOGUwYzQ0NmU1OGY3ZDY4NGIyZDM3Y2RjNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczMzI4MjUsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTExZTRkMDcwNmQ2NDA4YWIzOTY2YTg4ZjFlM2EzODkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2M0ZjgwNzcxYzFmNDViMTkyZTkzOTJhMmNkNGU1YTcgPSAkKCc8ZGl2IGlkPSJodG1sXzdjNGY4MDc3MWMxZjQ1YjE5MmU5MzkyYTJjZDRlNWE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZWRmb3JkIFBhcmssIExhd3JlbmNlIE1hbm9yIEVhc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UxMWU0ZDA3MDZkNjQwOGFiMzk2NmE4OGYxZTNhMzg5LnNldENvbnRlbnQoaHRtbF83YzRmODA3NzFjMWY0NWIxOTJlOTM5MmEyY2Q0ZTVhNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wOGIzNWIyOGUwYzQ0NmU1OGY3ZDY4NGIyZDM3Y2RjNS5iaW5kUG9wdXAocG9wdXBfZTExZTRkMDcwNmQ2NDA4YWIzOTY2YTg4ZjFlM2EzODkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzdlZjdjZDllNmU1NDdjY2EyYTk5Zjg1NDI2ZGViZTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTExMTU4LC03OS40NzYwMTMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYzlmODY1OGM4ODk0OTdlOGRiMWJmYmU0NWY3NjBkZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTc4YjhjODYxOWI0ODU4ODRlNmM5NjliODU4NGJkMSA9ICQoJzxkaXYgaWQ9Imh0bWxfN2E3OGI4Yzg2MTliNDg1ODg0ZTZjOTY5Yjg1ODRiZDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRlbCBSYXksIE1vdW50IERlbm5pcywgS2VlbHNkYWxlIGFuZCBTaWx2ZXJ0aG9ybiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M5Zjg2NThjODg5NDk3ZThkYjFiZmJlNDVmNzYwZGYuc2V0Q29udGVudChodG1sXzdhNzhiOGM4NjE5YjQ4NTg4NGU2Yzk2OWI4NTg0YmQxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M3ZWY3Y2Q5ZTZlNTQ3Y2NhMmE5OWY4NTQyNmRlYmU2LmJpbmRQb3B1cChwb3B1cF9jYzlmODY1OGM4ODk0OTdlOGRiMWJmYmU0NWY3NjBkZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMjE0M2VlNzQyMzM0ZjlhYTQ4YWRiNzZlN2RkMGRkZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNDc2NTksLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NjODE5MjkwZThjZTRlMTRiZTU2NTU4ZTUyMDc1MDVlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3M2IwOGI4NGY5OTRkY2JiYzRkZjIyMjUwMmJjMTQyID0gJCgnPGRpdiBpZD0iaHRtbF8wNzNiMDhiODRmOTk0ZGNiYmM0ZGYyMjI1MDJiYzE0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SHVtYmVybGVhLCBFbWVyeSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M4MTkyOTBlOGNlNGUxNGJlNTY1NThlNTIwNzUwNWUuc2V0Q29udGVudChodG1sXzA3M2IwOGI4NGY5OTRkY2JiYzRkZjIyMjUwMmJjMTQyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIyMTQzZWU3NDIzMzRmOWFhNDhhZGI3NmU3ZGQwZGRlLmJpbmRQb3B1cChwb3B1cF9jYzgxOTI5MGU4Y2U0ZTE0YmU1NjU1OGU1MjA3NTA1ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNmI0YTJhYzU4YWI0NmJmYjkxNjcxNmNkNjVjYjcwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MjY1NzAwMDAwMDAwNCwtNzkuMjY0ODQ4MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZDg0N2Y5YzNmNDI0OWNjOGZmZTIwZjhhNTZkYjM3ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YzcyMWJhYWExMDc0NWI0OGVkZWIwMjU0MDcxZGUyMyA9ICQoJzxkaXYgaWQ9Imh0bWxfN2M3MjFiYWFhMTA3NDViNDhlZGViMDI1NDA3MWRlMjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJpcmNoIENsaWZmLCBDbGlmZnNpZGUgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVkODQ3ZjljM2Y0MjQ5Y2M4ZmZlMjBmOGE1NmRiMzdkLnNldENvbnRlbnQoaHRtbF83YzcyMWJhYWExMDc0NWI0OGVkZWIwMjU0MDcxZGUyMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNmI0YTJhYzU4YWI0NmJmYjkxNjcxNmNkNjVjYjcwMC5iaW5kUG9wdXAocG9wdXBfNWQ4NDdmOWMzZjQyNDljYzhmZmUyMGY4YTU2ZGIzN2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTg5YWVlN2Q3OTVjNGU5MmE3Mjc0NWZlODgxZWE0MTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzAxMTk5LC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NGIwZTU5YzJhNjM0YTFkODUxNmMzMzlhOTI3NDk3YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZTVkZDk2MGU0NDQ0ODNhOTM1ZWVmMWMwZTc2MWM1YiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWU1ZGQ5NjBlNDQ0NDgzYTkzNWVlZjFjMGU3NjFjNWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIFdpbGxvd2RhbGUgRWFzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjRiMGU1OWMyYTYzNGExZDg1MTZjMzM5YTkyNzQ5N2Euc2V0Q29udGVudChodG1sXzllNWRkOTYwZTQ0NDQ4M2E5MzVlZWYxYzBlNzYxYzViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2U4OWFlZTdkNzk1YzRlOTJhNzI3NDVmZTg4MWVhNDExLmJpbmRQb3B1cChwb3B1cF82NGIwZTU5YzJhNjM0YTFkODUxNmMzMzlhOTI3NDk3YSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MGI4Mjg0NjRjNWU0OGE5OWI0MTc1YmQ5OGRlZGY5MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MTYzMTMsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NiYzZmOGViZmI3NTQzNmJiODM3NjQ5Y2FjMGFkMDE1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI5MmQ4ZGNkZmQ1OTQ0ZmE5MzIwZTBkYWRmMzZlZDEzID0gJCgnPGRpdiBpZD0iaHRtbF8yOTJkOGRjZGZkNTk0NGZhOTMyMGUwZGFkZjM2ZWQxMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYmM2ZjhlYmZiNzU0MzZiYjgzNzY0OWNhYzBhZDAxNS5zZXRDb250ZW50KGh0bWxfMjkyZDhkY2RmZDU5NDRmYTkzMjBlMGRhZGYzNmVkMTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTBiODI4NDY0YzVlNDhhOTliNDE3NWJkOThkZWRmOTMuYmluZFBvcHVwKHBvcHVwX2NiYzZmOGViZmI3NTQzNmJiODM3NjQ5Y2FjMGFkMDE1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmOGZjYzNjNmQ2NTRlYjc5Y2I0N2E4NDg2NDc1YTlhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83N2QyNDNiYmQ2MjU0YzE5OWNkN2JmZmY2MTMzODM0NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjMyZDc2MzE0NmQ0MjA0YmNkZGUyMTAyN2Q2ZjNmMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMmYzMmQ3NjMxNDZkNDIwNGJjZGRlMjEwMjdkNmYzZjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzdkMjQzYmJkNjI1NGMxOTljZDdiZmZmNjEzMzgzNDcuc2V0Q29udGVudChodG1sXzJmMzJkNzYzMTQ2ZDQyMDRiY2RkZTIxMDI3ZDZmM2YzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmOGZjYzNjNmQ2NTRlYjc5Y2I0N2E4NDg2NDc1YTlhLmJpbmRQb3B1cChwb3B1cF83N2QyNDNiYmQ2MjU0YzE5OWNkN2JmZmY2MTMzODM0Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOGUzNjhhZmQ4YWQ0ODdhOWJkNzE0NDI0NThmNGM0NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTY5NDgsLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNhMThkYThkNGM2MjQ2YzI5OTQ0OTNkY2JkYWI5MTY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdjNGY5ZTM1OWMxNzQwN2Q4MzY1NzZhODYyZmExZDk4ID0gJCgnPGRpdiBpZD0iaHRtbF83YzRmOWUzNTljMTc0MDdkODM2NTc2YTg2MmZhMWQ5OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWxhd24sIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2ExOGRhOGQ0YzYyNDZjMjk5NDQ5M2RjYmRhYjkxNjUuc2V0Q29udGVudChodG1sXzdjNGY5ZTM1OWMxNzQwN2Q4MzY1NzZhODYyZmExZDk4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q4ZTM2OGFmZDhhZDQ4N2E5YmQ3MTQ0MjQ1OGY0YzQ0LmJpbmRQb3B1cChwb3B1cF8zYTE4ZGE4ZDRjNjI0NmMyOTk0NDkzZGNiZGFiOTE2NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iYThkYmM3ZTE5Mjk0ZTAyYTkwZjkwZDg0NmEyOTVkYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MzE4NTI5OTk5OTk5LC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84OTY3YzgxMjZhOGY0YTVkYTJkOWUyY2IwMjBlNjQ0ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NThjMDBlNjY3ZjI0OWEyYjlkMDMxYzJiZTI1MzU0YSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTU4YzAwZTY2N2YyNDlhMmI5ZDAzMWMyYmUyNTM1NGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1bm55bWVkZSwgVGhlIEp1bmN0aW9uIE5vcnRoLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84OTY3YzgxMjZhOGY0YTVkYTJkOWUyY2IwMjBlNjQ0ZC5zZXRDb250ZW50KGh0bWxfOTU4YzAwZTY2N2YyNDlhMmI5ZDAzMWMyYmUyNTM1NGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmE4ZGJjN2UxOTI5NGUwMmE5MGY5MGQ4NDZhMjk1ZGEuYmluZFBvcHVwKHBvcHVwXzg5NjdjODEyNmE4ZjRhNWRhMmQ5ZTJjYjAyMGU2NDRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzc3ZGEzMmY0ZDQ3MDRmZDRiYWI5MzRhOTlkNjZiYzllID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LC03OS41MTgxODg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YmMxNjMzMTE3NWY0NGNlYmNkNTJhNTNkMzAxMmViZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMmFhN2Q4MDA5NDY0MWU4OWM2YmZlMzM4ZTU1NjExOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDJhYTdkODAwOTQ2NDFlODljNmJmZTMzOGU1NTYxMTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3RvbiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGJjMTYzMzExNzVmNDRjZWJjZDUyYTUzZDMwMTJlYmYuc2V0Q29udGVudChodG1sXzAyYWE3ZDgwMDk0NjQxZTg5YzZiZmUzMzhlNTU2MTE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc3ZGEzMmY0ZDQ3MDRmZDRiYWI5MzRhOTlkNjZiYzllLmJpbmRQb3B1cChwb3B1cF84YmMxNjMzMTE3NWY0NGNlYmNkNTJhNTNkMzAxMmViZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMWVmODAyY2NmZjE0YTA4OTFjYzY1NzhiMDkzY2I2NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQwOTYsLTc5LjI3MzMwNDAwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2M1MDdiZDU0MTNlZDRjYjA5NTcxZGExZDVlOTVjNmY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdlNWI4ZjQ4ZWNmZDRhZWY4MjFlZWUzMDJkMGFkM2Y5ID0gJCgnPGRpdiBpZD0iaHRtbF83ZTViOGY0OGVjZmQ0YWVmODIxZWVlMzAyZDBhZDNmOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9yc2V0IFBhcmssIFdleGZvcmQgSGVpZ2h0cywgU2NhcmJvcm91Z2ggVG93biBDZW50cmUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNTA3YmQ1NDEzZWQ0Y2IwOTU3MWRhMWQ1ZTk1YzZmNS5zZXRDb250ZW50KGh0bWxfN2U1YjhmNDhlY2ZkNGFlZjgyMWVlZTMwMmQwYWQzZjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTFlZjgwMmNjZmYxNGEwODkxY2M2NTc4YjA5M2NiNjYuYmluZFBvcHVwKHBvcHVwX2M1MDdiZDU0MTNlZDRjYjA5NTcxZGExZDVlOTVjNmY1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YzOGExOGFiMjJkNTQyNDhhMGNhN2Q5MTRjYmI5MTAyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUyNzU4Mjk5OTk5OTk2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JlMWJlYzcxYjI5ZjQ4NzI5YjE2MzI5NzE5ZmQ0YjFkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JmNjc5NGVmMjMwMzQzMGViZjgyNjNhZTFhOTRiZDE2ID0gJCgnPGRpdiBpZD0iaHRtbF9iZjY3OTRlZjIzMDM0MzBlYmY4MjYzYWUxYTk0YmQxNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscyBXZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZTFiZWM3MWIyOWY0ODcyOWIxNjMyOTcxOWZkNGIxZC5zZXRDb250ZW50KGh0bWxfYmY2Nzk0ZWYyMzAzNDMwZWJmODI2M2FlMWE5NGJkMTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjM4YTE4YWIyMmQ1NDI0OGEwY2E3ZDkxNGNiYjkxMDIuYmluZFBvcHVwKHBvcHVwX2JlMWJlYzcxYjI5ZjQ4NzI5YjE2MzI5NzE5ZmQ0YjFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I2ZWQ2ZjYzOTJlZDQ1OWJiMWRiMzJlYTNmYjYxMWRhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84ZTU2M2VkY2RkYjU0YjhiOTRmM2NmYjQxODJiMWE0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kM2RkNjNiZDA3MDY0NzM5OTM3YzZjZmY0NGY2NjFjYyA9ICQoJzxkaXYgaWQ9Imh0bWxfZDNkZDYzYmQwNzA2NDczOTkzN2M2Y2ZmNDRmNjYxY2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGgsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGU1NjNlZGNkZGI1NGI4Yjk0ZjNjZmI0MTgyYjFhNGMuc2V0Q29udGVudChodG1sX2QzZGQ2M2JkMDcwNjQ3Mzk5MzdjNmNmZjQ0ZjY2MWNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I2ZWQ2ZjYzOTJlZDQ1OWJiMWRiMzJlYTNmYjYxMWRhLmJpbmRQb3B1cChwb3B1cF84ZTU2M2VkY2RkYjU0YjhiOTRmM2NmYjQxODJiMWE0Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNzU2ZDUyNWU4MmQ0NDU0YTA3YzFjOWFjYmE0ZTQzYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc1MjY4YmUwMGQyYTRmZWJiNWE2MDAwNWYzNjdkOTU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzY1MmY3OGZmOGQxMzRjYTg5Y2ZjODRjZDdkZTE2OTIxID0gJCgnPGRpdiBpZD0iaHRtbF82NTJmNzhmZjhkMTM0Y2E4OWNmYzg0Y2Q3ZGUxNjkyMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgTm9ydGggJmFtcDsgV2VzdCwgRm9yZXN0IEhpbGwgUm9hZCBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc1MjY4YmUwMGQyYTRmZWJiNWE2MDAwNWYzNjdkOTU0LnNldENvbnRlbnQoaHRtbF82NTJmNzhmZjhkMTM0Y2E4OWNmYzg0Y2Q3ZGUxNjkyMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNzU2ZDUyNWU4MmQ0NDU0YTA3YzFjOWFjYmE0ZTQzYy5iaW5kUG9wdXAocG9wdXBfNzUyNjhiZTAwZDJhNGZlYmI1YTYwMDA1ZjM2N2Q5NTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWZhNmZiYWFkZTkwNDU5NzgzMDAxZTc3ZGUzNjBmOTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYTVkNGY3M2JmZjk0Y2NkODZhNThmYzRlMzZiMjhjNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YjQwZjBhM2Y2MDY0ZTk3OWRiOTcwYWJmZDM4MzdiNCA9ICQoJzxkaXYgaWQ9Imh0bWxfOGI0MGYwYTNmNjA2NGU5NzlkYjk3MGFiZmQzODM3YjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2ggUGFyaywgVGhlIEp1bmN0aW9uIFNvdXRoLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FhNWQ0ZjczYmZmOTRjY2Q4NmE1OGZjNGUzNmIyOGM2LnNldENvbnRlbnQoaHRtbF84YjQwZjBhM2Y2MDY0ZTk3OWRiOTcwYWJmZDM4MzdiNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZmE2ZmJhYWRlOTA0NTk3ODMwMDFlNzdkZTM2MGY5MS5iaW5kUG9wdXAocG9wdXBfYWE1ZDRmNzNiZmY5NGNjZDg2YTU4ZmM0ZTM2YjI4YzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDRlZTJlZWI2ZjVmNDExMGEzZmEyNzIyMzlhYzc1NDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTYzMTksLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI2YWVlNTk1YTE4YTRjNDM5MGU3YTg2ODViOWQwYjQwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VlMTFiMTIxNGMxNDQ0MTFiOTc1MDlmYTNkODg4MjdiID0gJCgnPGRpdiBpZD0iaHRtbF9lZTExYjEyMTRjMTQ0NDExYjk3NTA5ZmEzZDg4ODI3YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG1vdW50LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI2YWVlNTk1YTE4YTRjNDM5MGU3YTg2ODViOWQwYjQwLnNldENvbnRlbnQoaHRtbF9lZTExYjEyMTRjMTQ0NDExYjk3NTA5ZmEzZDg4ODI3Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80NGVlMmVlYjZmNWY0MTEwYTNmYTI3MjIzOWFjNzU0Ni5iaW5kUG9wdXAocG9wdXBfMjZhZWU1OTVhMThhNGM0MzkwZTdhODY4NWI5ZDBiNDApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjhmYmVkMzU5OGEwNDZiOWFmNzNlZWZiMDliOTMxOTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTAwNzE1MDAwMDAwMDQsLTc5LjI5NTg0OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjczZTI1YWQ5NzlkNDhkMDlmMjYyZTMxNDZmZDIyNDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTJjZGVlMzYxNGMwNGE1NmJjZTUxOWY5MzkxYTA5ZDQgPSAkKCc8ZGl2IGlkPSJodG1sX2UyY2RlZTM2MTRjMDRhNTZiY2U1MTlmOTM5MWEwOWQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXhmb3JkLCBNYXJ5dmFsZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY3M2UyNWFkOTc5ZDQ4ZDA5ZjI2MmUzMTQ2ZmQyMjQzLnNldENvbnRlbnQoaHRtbF9lMmNkZWUzNjE0YzA0YTU2YmNlNTE5ZjkzOTFhMDlkNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mOGZiZWQzNTk4YTA0NmI5YWY3M2VlZmIwOWI5MzE5Ny5iaW5kUG9wdXAocG9wdXBfNjczZTI1YWQ5NzlkNDhkMDlmMjYyZTMxNDZmZDIyNDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDcwY2RmZWZkZjczNGE2N2JlMWE1MDg2YzU1YTQ1ZGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODI3MzY0LC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YwOTc3YWNhNzk3MjRiNDBiNDZiZWIyZDZhOWNiN2EwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlmNDYyYmM5MDFkNzQyZGRiY2EzYjMxMmZhOWVjMDNlID0gJCgnPGRpdiBpZD0iaHRtbF85ZjQ2MmJjOTAxZDc0MmRkYmNhM2IzMTJmYTllYzAzZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSwgV2lsbG93ZGFsZSBXZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMDk3N2FjYTc5NzI0YjQwYjQ2YmViMmQ2YTljYjdhMC5zZXRDb250ZW50KGh0bWxfOWY0NjJiYzkwMWQ3NDJkZGJjYTNiMzEyZmE5ZWMwM2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDcwY2RmZWZkZjczNGE2N2JlMWE1MDg2YzU1YTQ1ZGQuYmluZFBvcHVwKHBvcHVwX2YwOTc3YWNhNzk3MjRiNDBiNDZiZWIyZDZhOWNiN2EwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIwYzI0YTE1ZTliMDQyNjdiZTdhMmM5YzdiNGRlZWQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWFhNjM3YTAxYmViNDVjMDgwNzI4OGI1OThmZTUxYjEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTc0YTlmODMwYWRhNGE2N2IyNTQzYjRlZWIzZjI4ODMgPSAkKCc8ZGl2IGlkPSJodG1sXzE3NGE5ZjgzMGFkYTRhNjdiMjU0M2I0ZWViM2YyODgzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsICBMYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFhYTYzN2EwMWJlYjQ1YzA4MDcyODhiNTk4ZmU1MWIxLnNldENvbnRlbnQoaHRtbF8xNzRhOWY4MzBhZGE0YTY3YjI1NDNiNGVlYjNmMjg4Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMGMyNGExNWU5YjA0MjY3YmU3YTJjOWM3YjRkZWVkMi5iaW5kUG9wdXAocG9wdXBfMWFhNjM3YTAxYmViNDVjMDgwNzI4OGI1OThmZTUxYjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2YzZGIzNTlhN2I1NDNhZDk5Zjg0N2E3YjM3NTVkODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMDE5MmRlYWZlNWI0NTQ3OTUxN2UwMGE2NGFjYmVhZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MThhY2VmODU2MmY0YzgxYTNlNDNiNGJhMjIxNGRkZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNDE4YWNlZjg1NjJmNGM4MWEzZTQzYjRiYTIyMTRkZGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwMTkyZGVhZmU1YjQ1NDc5NTE3ZTAwYTY0YWNiZWFlLnNldENvbnRlbnQoaHRtbF80MThhY2VmODU2MmY0YzgxYTNlNDNiNGJhMjIxNGRkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZjNkYjM1OWE3YjU0M2FkOTlmODQ3YTdiMzc1NWQ4Ny5iaW5kUG9wdXAocG9wdXBfYzAxOTJkZWFmZTViNDU0Nzk1MTdlMDBhNjRhY2JlYWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjA5Y2UwODIwNTdlNGU2MzkwZDk3ZTUyMWM4ZDNjZDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg5NTk3LC03OS40NTYzMjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjcwNzE2MjYzYzM5NDcwMzk5YTg1ODIxZjBjZjBjMDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2IxYjVlNWVjM2NmNDYzMDgwNDQ4MjhiYjcyNGViMzUgPSAkKCc8ZGl2IGlkPSJodG1sXzNiMWI1ZTVlYzNjZjQ2MzA4MDQ0ODI4YmI3MjRlYjM1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSwgUm9uY2VzdmFsbGVzLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI3MDcxNjI2M2MzOTQ3MDM5OWE4NTgyMWYwY2YwYzAxLnNldENvbnRlbnQoaHRtbF8zYjFiNWU1ZWMzY2Y0NjMwODA0NDgyOGJiNzI0ZWIzNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMDljZTA4MjA1N2U0ZTYzOTBkOTdlNTIxYzhkM2NkOS5iaW5kUG9wdXAocG9wdXBfMjcwNzE2MjYzYzM5NDcwMzk5YTg1ODIxZjBjZjBjMDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWZlODZlNDU0ZGMzNDdhNGJmY2NkYzM1YWMwY2ExM2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY5NjU2LC03OS42MTU4MTg5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMGMyMTQ4ZWI3ZWU0MGZjYjJiYTE1MGNhMzgwN2E2ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jZTAyNGFhZmY4OTU0YmRlYjhhNTg3MDJmNmEzZTI2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfY2UwMjRhYWZmODk1NGJkZWI4YTU4NzAyZjZhM2UyNmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbmFkYSBQb3N0IEdhdGV3YXkgUHJvY2Vzc2luZyBDZW50cmUsIE1pc3Npc3NhdWdhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMGMyMTQ4ZWI3ZWU0MGZjYjJiYTE1MGNhMzgwN2E2ZC5zZXRDb250ZW50KGh0bWxfY2UwMjRhYWZmODk1NGJkZWI4YTU4NzAyZjZhM2UyNmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWZlODZlNDU0ZGMzNDdhNGJmY2NkYzM1YWMwY2ExM2MuYmluZFBvcHVwKHBvcHVwXzEwYzIxNDhlYjdlZTQwZmNiMmJhMTUwY2EzODA3YTZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdjYjk3YzVjMzdhMTQwNjNiYTkyZWY5ZjI1MjQxYzJkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDdiNGU3YTUxYjM0NDk3YTk4NTViNjhkNTQwYzQwNjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDAyMDExNTQ2M2UzNDk2NmE3Y2Y1NWU1MDM1NWNmMzYgPSAkKCc8ZGl2IGlkPSJodG1sX2QwMjAxMTU0NjNlMzQ5NjZhN2NmNTVlNTAzNTVjZjM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3ZpZXcgVmlsbGFnZSwgU3QuIFBoaWxsaXBzLCBNYXJ0aW4gR3JvdmUgR2FyZGVucywgUmljaHZpZXcgR2FyZGVucywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wN2I0ZTdhNTFiMzQ0OTdhOTg1NWI2OGQ1NDBjNDA2Mi5zZXRDb250ZW50KGh0bWxfZDAyMDExNTQ2M2UzNDk2NmE3Y2Y1NWU1MDM1NWNmMzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2NiOTdjNWMzN2ExNDA2M2JhOTJlZjlmMjUyNDFjMmQuYmluZFBvcHVwKHBvcHVwXzA3YjRlN2E1MWIzNDQ5N2E5ODU1YjY4ZDU0MGM0MDYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzczMTM2Y2I2MzllYzRmYmFhMTk2OGE2NWRhOTAzZDQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk0MjAwMywtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjcyMTIwNzY3NzNlNDg0ZTkyZGU4YmE3NThkNGY0YjkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTQyNTU4YThhMWEzNGI3MmFlYjlhYmU3OWJhMzM3MjkgPSAkKCc8ZGl2IGlkPSJodG1sXzE0MjU1OGE4YTFhMzRiNzJhZWI5YWJlNzliYTMzNzI5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZ2luY291cnQsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNzIxMjA3Njc3M2U0ODRlOTJkZThiYTc1OGQ0ZjRiOS5zZXRDb250ZW50KGh0bWxfMTQyNTU4YThhMWEzNGI3MmFlYjlhYmU3OWJhMzM3MjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzMxMzZjYjYzOWVjNGZiYWExOTY4YTY1ZGE5MDNkNDIuYmluZFBvcHVwKHBvcHVwXzI3MjEyMDc2NzczZTQ4NGU5MmRlOGJhNzU4ZDRmNGI5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QyMDYzZTY2NmU0ZDQzZjY5YWU5YjgyNDFhZTFkNzUzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MjY5NzViMmY3Yzg0YjgyOTk1NmQwNTg5OWQyNWMxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85Y2ViNWU1ZmJjNGY0YjA2OWRjZDE4OTlmYzI0YTVlZiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWNlYjVlNWZiYzRmNGIwNjlkY2QxODk5ZmMyNGE1ZWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDI2OTc1YjJmN2M4NGI4Mjk5NTZkMDU4OTlkMjVjMWYuc2V0Q29udGVudChodG1sXzljZWI1ZTVmYmM0ZjRiMDY5ZGNkMTg5OWZjMjRhNWVmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QyMDYzZTY2NmU0ZDQzZjY5YWU5YjgyNDFhZTFkNzUzLmJpbmRQb3B1cChwb3B1cF80MjY5NzViMmY3Yzg0YjgyOTk1NmQwNTg5OWQyNWMxZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZDc5YjdmNjI4MDc0ODJmYWFkNGJjN2U1OTBhM2RjYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjQwYzJkYzFhMThiNDYxNWI2YWRhNTkxOWRkZjQ3OWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTgyODU3ZjcyZmNkNDgzOGJjYTY1ZmIzZTI0ZTFkMDEgPSAkKCc8ZGl2IGlkPSJodG1sX2E4Mjg1N2Y3MmZjZDQ4MzhiY2E2NWZiM2UyNGUxZDAxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY0MGMyZGMxYTE4YjQ2MTViNmFkYTU5MTlkZGY0NzljLnNldENvbnRlbnQoaHRtbF9hODI4NTdmNzJmY2Q0ODM4YmNhNjVmYjNlMjRlMWQwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZDc5YjdmNjI4MDc0ODJmYWFkNGJjN2U1OTBhM2RjYS5iaW5kUG9wdXAocG9wdXBfNjQwYzJkYzFhMThiNDYxNWI2YWRhNTkxOWRkZjQ3OWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmZiNmZmYzExOGEwNDg4MzkxOTJjZmU5YjJkZTAyNDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RjNzk2ZjI0NTBlZjQzYjc4NmI4MTdkYTdlYjA0MDM5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2IwMjAzMjI3MDZlYzQ4YzBiOGI4YjY3ZWM1YjhhYjZhID0gJCgnPGRpdiBpZD0iaHRtbF9iMDIwMzIyNzA2ZWM0OGMwYjhiOGI2N2VjNWI4YWI2YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RjNzk2ZjI0NTBlZjQzYjc4NmI4MTdkYTdlYjA0MDM5LnNldENvbnRlbnQoaHRtbF9iMDIwMzIyNzA2ZWM0OGMwYjhiOGI2N2VjNWI4YWI2YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yZmI2ZmZjMTE4YTA0ODgzOTE5MmNmZTliMmRlMDI0MC5iaW5kUG9wdXAocG9wdXBfZGM3OTZmMjQ1MGVmNDNiNzg2YjgxN2RhN2ViMDQwMzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjg3NjE2NTQ0OGFlNGUyZDgzZGQxMDBjNTM4OTUwY2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODE2Mzc1LC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MwZTZmMTdjMzI2NDRkMTliNTVjNTFmNjhjNGI2M2MzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdiZmRjNjczMzlhYTRhOWRhOWYxNTE4NmU5MmVhNTZmID0gJCgnPGRpdiBpZD0iaHRtbF83YmZkYzY3MzM5YWE0YTlkYTlmMTUxODZlOTJlYTU2ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xhcmtzIENvcm5lcnMsIFRhbSBPJiMzOTtTaGFudGVyLCBTdWxsaXZhbiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwZTZmMTdjMzI2NDRkMTliNTVjNTFmNjhjNGI2M2MzLnNldENvbnRlbnQoaHRtbF83YmZkYzY3MzM5YWE0YTlkYTlmMTUxODZlOTJlYTU2Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ODc2MTY1NDQ4YWU0ZTJkODNkZDEwMGM1Mzg5NTBjZC5iaW5kUG9wdXAocG9wdXBfYzBlNmYxN2MzMjY0NGQxOWI1NWM1MWY2OGM0YjYzYzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGIyMDYxMjcxZDJjNGZjZThkYTkwNmM0M2MxZWE2ZGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83M2Y5YWU3OTNhYTI0MWFhOWExMjQ1OWNjNTEwODdmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNGNjMjViMWFmZWU0MGVjOWI2ZmEzZTUyZDVjMGZjMiA9ICQoJzxkaXYgaWQ9Imh0bWxfYTRjYzI1YjFhZmVlNDBlYzliNmZhM2U1MmQ1YzBmYzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83M2Y5YWU3OTNhYTI0MWFhOWExMjQ1OWNjNTEwODdmMS5zZXRDb250ZW50KGh0bWxfYTRjYzI1YjFhZmVlNDBlYzliNmZhM2U1MmQ1YzBmYzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGIyMDYxMjcxZDJjNGZjZThkYTkwNmM0M2MxZWE2ZGYuYmluZFBvcHVwKHBvcHVwXzczZjlhZTc5M2FhMjQxYWE5YTEyNDU5Y2M1MTA4N2YxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UxNmRhNWM2NzkwMTRhMGViYzk2MTk1YjE1Y2Y2MTQwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZjEyYmMzZTA3YWE0Mjc2OTBhMDE2MmE1NTdhNmRmOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMDkwYTE4Zjc2MTM0YzhkODA3NWZkMjdhOTRjNjBmNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTA5MGExOGY3NjEzNGM4ZDgwNzVmZDI3YTk0YzYwZjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbnNpbmd0b24gTWFya2V0LCBDaGluYXRvd24sIEdyYW5nZSBQYXJrLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZjEyYmMzZTA3YWE0Mjc2OTBhMDE2MmE1NTdhNmRmOS5zZXRDb250ZW50KGh0bWxfMTA5MGExOGY3NjEzNGM4ZDgwNzVmZDI3YTk0YzYwZjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTE2ZGE1YzY3OTAxNGEwZWJjOTYxOTViMTVjZjYxNDAuYmluZFBvcHVwKHBvcHVwXzlmMTJiYzNlMDdhYTQyNzY5MGEwMTYyYTU1N2E2ZGY5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FlYjMzZWUyZTU0ZjQxY2JhNmVjOTdmZTAyNDdmZTFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODE1MjUyMiwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zZDJkMjdhNWFjNWI0NjgzODBmMmJhNGY1ZmIyY2ZlMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zN2ZjNmUxNjdhNDM0MjM1OTE2ODBlOTA2Nzc1YzZlYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMzdmYzZlMTY3YTQzNDIzNTkxNjgwZTkwNjc3NWM2ZWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1pbGxpa2VuLCBBZ2luY291cnQgTm9ydGgsIFN0ZWVsZXMgRWFzdCwgTCYjMzk7QW1vcmVhdXggRWFzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNkMmQyN2E1YWM1YjQ2ODM4MGYyYmE0ZjVmYjJjZmUxLnNldENvbnRlbnQoaHRtbF8zN2ZjNmUxNjdhNDM0MjM1OTE2ODBlOTA2Nzc1YzZlYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZWIzM2VlMmU1NGY0MWNiYTZlYzk3ZmUwMjQ3ZmUxZi5iaW5kUG9wdXAocG9wdXBfM2QyZDI3YTVhYzViNDY4MzgwZjJiYTRmNWZiMmNmZTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWRjZDEzN2JlMWEwNGU5Y2I5MDAxYzYzMzViZjgxZGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MmFlMjY1ZTRkZTg0Y2UwYTllYmUzOGM3MGY3YTVmMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNjgzODU1NTc0ZWU0YWVlYjg5YzRiM2E4ZjFjZDRlZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzY4Mzg1NTU3NGVlNGFlZWI4OWM0YjNhOGYxY2Q0ZWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkyYWUyNjVlNGRlODRjZTBhOWViZTM4YzcwZjdhNWYzLnNldENvbnRlbnQoaHRtbF8zNjgzODU1NTc0ZWU0YWVlYjg5YzRiM2E4ZjFjZDRlZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hZGNkMTM3YmUxYTA0ZTljYjkwMDFjNjMzNWJmODFkZS5iaW5kUG9wdXAocG9wdXBfOTJhZTI2NWU0ZGU4NGNlMGE5ZWJlMzhjNzBmN2E1ZjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzQzNTFhMjQzNmVjNDdhYzk3N2M0NDhiMzY0NGI2NzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhjZGY0NDU1NmZlMDQyZTY4ZDYzODRjMjYzMTVmM2MwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I0MGI2MTFlODE0ZDQwYWZiM2RmNjgzZGEyNDg1YmRkID0gJCgnPGRpdiBpZD0iaHRtbF9iNDBiNjExZTgxNGQ0MGFmYjNkZjY4M2RhMjQ4NWJkZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q04gVG93ZXIsIEtpbmcgYW5kIFNwYWRpbmEsIFJhaWx3YXkgTGFuZHMsIEhhcmJvdXJmcm9udCBXZXN0LCBCYXRodXJzdCBRdWF5LCBTb3V0aCBOaWFnYXJhLCBJc2xhbmQgYWlycG9ydCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGNkZjQ0NTU2ZmUwNDJlNjhkNjM4NGMyNjMxNWYzYzAuc2V0Q29udGVudChodG1sX2I0MGI2MTFlODE0ZDQwYWZiM2RmNjgzZGEyNDg1YmRkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M0MzUxYTI0MzZlYzQ3YWM5NzdjNDQ4YjM2NDRiNjc0LmJpbmRQb3B1cChwb3B1cF84Y2RmNDQ1NTZmZTA0MmU2OGQ2Mzg0YzI2MzE1ZjNjMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNjczYWUwNzQxZTc0NDFjOTczZTNhNGMwZWEzMzM0NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwNTY0NjYsLTc5LjUwMTMyMDcwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UyN2E2NDI1YTFlNjRiMGM4OTNlN2VlMzlhNzY0ZmI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE3YjFiYTdjMmUyNzQ5OTM5ODI2MjIwZGNiNDYyMzY0ID0gJCgnPGRpdiBpZD0iaHRtbF8xN2IxYmE3YzJlMjc0OTkzOTgyNjIyMGRjYjQ2MjM2NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IFRvcm9udG8sIE1pbWljbyBTb3V0aCwgSHVtYmVyIEJheSBTaG9yZXMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTI3YTY0MjVhMWU2NGIwYzg5M2U3ZWUzOWE3NjRmYjkuc2V0Q29udGVudChodG1sXzE3YjFiYTdjMmUyNzQ5OTM5ODI2MjIwZGNiNDYyMzY0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E2NzNhZTA3NDFlNzQ0MWM5NzNlM2E0YzBlYTMzMzQ2LmJpbmRQb3B1cChwb3B1cF9lMjdhNjQyNWExZTY0YjBjODkzZTdlZTM5YTc2NGZiOSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iZWJhYzFiYmM2MTg0Y2QxYjI1MzQyN2M5MzRiYmJiMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNzAyYWI3ZmEzY2M0M2Y1OTc4MTlhMTdjODg0MjFhMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xM2JjNWI5MjAwYzg0N2ZmOTNkZjZhZTE0MGMxZDlkOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTNiYzViOTIwMGM4NDdmZjkzZGY2YWUxNDBjMWQ5ZDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvdXRoIFN0ZWVsZXMsIFNpbHZlcnN0b25lLCBIdW1iZXJnYXRlLCBKYW1lc3Rvd24sIE1vdW50IE9saXZlLCBCZWF1bW9uZCBIZWlnaHRzLCBUaGlzdGxldG93biwgQWxiaW9uIEdhcmRlbnMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjcwMmFiN2ZhM2NjNDNmNTk3ODE5YTE3Yzg4NDIxYTAuc2V0Q29udGVudChodG1sXzEzYmM1YjkyMDBjODQ3ZmY5M2RmNmFlMTQwYzFkOWQ4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2JlYmFjMWJiYzYxODRjZDFiMjUzNDI3YzkzNGJiYmIzLmJpbmRQb3B1cChwb3B1cF8yNzAyYWI3ZmEzY2M0M2Y1OTc4MTlhMTdjODg0MjFhMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80OGQ0YTA2ZDU3YTc0NmIyYWI0OTkxY2Y3ZTFjZDUzNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc5OTUyNTIwMDAwMDAwNSwtNzkuMzE4Mzg4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMDllZjljOGM5YmI0MTA1ODNlYzQ5YWY2ZTQ4MjIxMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NGUzY2UwNDExMGM0OWJhYWVmMDEyZmYyZDAyNmQ0NyA9ICQoJzxkaXYgaWQ9Imh0bWxfOTRlM2NlMDQxMTBjNDliYWFlZjAxMmZmMmQwMjZkNDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0ZWVsZXMgV2VzdCwgTCYjMzk7QW1vcmVhdXggV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UwOWVmOWM4YzliYjQxMDU4M2VjNDlhZjZlNDgyMjExLnNldENvbnRlbnQoaHRtbF85NGUzY2UwNDExMGM0OWJhYWVmMDEyZmYyZDAyNmQ0Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80OGQ0YTA2ZDU3YTc0NmIyYWI0OTkxY2Y3ZTFjZDUzNy5iaW5kUG9wdXAocG9wdXBfZTA5ZWY5YzhjOWJiNDEwNTgzZWM0OWFmNmU0ODIyMTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2M2MDE5N2FjZjM0NDM4MDkzOTAyMTdkZDg4ZjQzMTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNGQ0NGRmZDg0YTQ0OGUwOGU1ODg1YWZkOWQ4YTY5ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hOWNkNmIyNjIzMmI0NzY0YjA4ZDRiMDBlODI5MTZlMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTljZDZiMjYyMzJiNDc2NGIwOGQ0YjAwZTgyOTE2ZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VkYWxlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNGQ0NGRmZDg0YTQ0OGUwOGU1ODg1YWZkOWQ4YTY5ZS5zZXRDb250ZW50KGh0bWxfYTljZDZiMjYyMzJiNDc2NGIwOGQ0YjAwZTgyOTE2ZTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2M2MDE5N2FjZjM0NDM4MDkzOTAyMTdkZDg4ZjQzMTcuYmluZFBvcHVwKHBvcHVwX2M0ZDQ0ZGZkODRhNDQ4ZTA4ZTU4ODVhZmQ5ZDhhNjllKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRlNjAzYWVkYmJkNDRjN2FiNTNhYzAyMDQxMWNiYmU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTA4YTNlN2Q0YTdjNDUwZjhkYzNjMzg0YmU5Y2UxZjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDA4ZWMzZjI1NjBlNDA2ODg2MWEzZTc1ODNjYWE3YmQgPSAkKCc8ZGl2IGlkPSJodG1sX2QwOGVjM2YyNTYwZTQwNjg4NjFhM2U3NTgzY2FhN2JkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTA4YTNlN2Q0YTdjNDUwZjhkYzNjMzg0YmU5Y2UxZjQuc2V0Q29udGVudChodG1sX2QwOGVjM2YyNTYwZTQwNjg4NjFhM2U3NTgzY2FhN2JkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRlNjAzYWVkYmJkNDRjN2FiNTNhYzAyMDQxMWNiYmU0LmJpbmRQb3B1cChwb3B1cF8xMDhhM2U3ZDRhN2M0NTBmOGRjM2MzODRiZTljZTFmNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZTc0ZTVmMDMzYTg0ZGVjYmQ0YjE5NzkyZWU5MzI5MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwMjQxMzcwMDAwMDAxLC03OS41NDM0ODQwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMzk1NDA1MDgxNjQ0ZTFiOGRiMDAxNWQ2MWUyN2IwZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNTJkMDZlZmExY2U0YWYzODU3YjJjNTIyNjg4NWQ4NiA9ICQoJzxkaXYgaWQ9Imh0bWxfMTUyZDA2ZWZhMWNlNGFmMzg1N2IyYzUyMjY4ODVkODYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsZGVyd29vZCwgTG9uZyBCcmFuY2gsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjM5NTQwNTA4MTY0NGUxYjhkYjAwMTVkNjFlMjdiMGYuc2V0Q29udGVudChodG1sXzE1MmQwNmVmYTFjZTRhZjM4NTdiMmM1MjI2ODg1ZDg2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NlNzRlNWYwMzNhODRkZWNiZDRiMTk3OTJlZTkzMjkzLmJpbmRQb3B1cChwb3B1cF9iMzk1NDA1MDgxNjQ0ZTFiOGRiMDAxNWQ2MWUyN2IwZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YWMxOTlkZTQ1ZDM0ZDM5YjkzOTAxMWNmM2U1NzViZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjc0ODI5OTk5OTk5NCwtNzkuNTk0MDU0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMGIwNTBmODlhZTA0ZWVkODMzMjdjOThkNjM3N2JlZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZmYyYjI0MDYyMWQ0OWMyYjZiZjhlZTgwM2JjMGMxNyA9ICQoJzxkaXYgaWQ9Imh0bWxfOGZmMmIyNDA2MjFkNDljMmI2YmY4ZWU4MDNiYzBjMTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod2VzdCwgV2VzdCBIdW1iZXIgLSBDbGFpcnZpbGxlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IwYjA1MGY4OWFlMDRlZWQ4MzMyN2M5OGQ2Mzc3YmVlLnNldENvbnRlbnQoaHRtbF84ZmYyYjI0MDYyMWQ0OWMyYjZiZjhlZTgwM2JjMGMxNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YWMxOTlkZTQ1ZDM0ZDM5YjkzOTAxMWNmM2U1NzViZi5iaW5kUG9wdXAocG9wdXBfYjBiMDUwZjg5YWUwNGVlZDgzMzI3Yzk4ZDYzNzdiZWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTMwNjU0YzM3NGRkNDZhMWFjN2M0MDc2ZWVkNWQ2MjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MzYxMjQ3MDAwMDAwMDYsLTc5LjIwNTYzNjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzAzZjVmZmQxZTQ5ZjRlNjU4MTNkZDA5MzI1YzdiM2U1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M4ZTdmZmFhY2NlNDRkODE4ZmM2NGUzYWJlZjAyODdiID0gJCgnPGRpdiBpZD0iaHRtbF9jOGU3ZmZhYWNjZTQ0ZDgxOGZjNjRlM2FiZWYwMjg3YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VXBwZXIgUm91Z2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wM2Y1ZmZkMWU0OWY0ZTY1ODEzZGQwOTMyNWM3YjNlNS5zZXRDb250ZW50KGh0bWxfYzhlN2ZmYWFjY2U0NGQ4MThmYzY0ZTNhYmVmMDI4N2IpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTMwNjU0YzM3NGRkNDZhMWFjN2M0MDc2ZWVkNWQ2MjcuYmluZFBvcHVwKHBvcHVwXzAzZjVmZmQxZTQ5ZjRlNjU4MTNkZDA5MzI1YzdiM2U1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VmNmVjYzYwNzk2MTQyODRiYjY2NWMwZDdkNzk0NTU3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3OTY3LC03OS4zNjc2NzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q3YjNmNWFiOGYyNjQ5ZjlhOTY3YWJkZmJkOGJjNjRjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZiMzJjNjI2NjA0MDQ3ZDBhNjFmMGRmMjdiM2MwYmI2ID0gJCgnPGRpdiBpZD0iaHRtbF9mYjMyYzYyNjYwNDA0N2QwYTYxZjBkZjI3YjNjMGJiNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIENhYmJhZ2V0b3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kN2IzZjVhYjhmMjY0OWY5YTk2N2FiZGZiZDhiYzY0Yy5zZXRDb250ZW50KGh0bWxfZmIzMmM2MjY2MDQwNDdkMGE2MWYwZGYyN2IzYzBiYjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZWY2ZWNjNjA3OTYxNDI4NGJiNjY1YzBkN2Q3OTQ1NTcuYmluZFBvcHVwKHBvcHVwX2Q3YjNmNWFiOGYyNjQ5ZjlhOTY3YWJkZmJkOGJjNjRjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NkYzcwZWI5MzM0NzQzZjBiYTFjMTU0YTA3ZDg1Mjg4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMmQ1M2Q0YWRhZDk0YzAxOWY0M2I4M2QxMWY0M2YxMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMzgwNmRiOWIyYTY0N2QzOTQyNzFhNGMyYjc2NjhlNSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjM4MDZkYjliMmE2NDdkMzk0MjcxYTRjMmI3NjY4ZTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMmQ1M2Q0YWRhZDk0YzAxOWY0M2I4M2QxMWY0M2YxMC5zZXRDb250ZW50KGh0bWxfZjM4MDZkYjliMmE2NDdkMzk0MjcxYTRjMmI3NjY4ZTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2RjNzBlYjkzMzQ3NDNmMGJhMWMxNTRhMDdkODUyODguYmluZFBvcHVwKHBvcHVwX2UyZDUzZDRhZGFkOTRjMDE5ZjQzYjgzZDExZjQzZjEwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBlMTllOTliMDgyZTQyMjY5M2U0YmE4ZDBhM2E4NmJiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzNjUzNjAwMDAwMDA1LC03OS41MDY5NDM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUwMzNlNzkzZmE3ZDQzZDRiMjRkZDdkMzY4ZmE5MTgwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JiZmVlZDIzMjJkMDQ3MTE4OGI1NTA4OWU0ZDI2MTNlID0gJCgnPGRpdiBpZD0iaHRtbF9iYmZlZWQyMzIyZDA0NzExODhiNTUwODllNGQyNjEzZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEtpbmdzd2F5LCBNb250Z29tZXJ5IFJvYWQsIE9sZCBNaWxsIE5vcnRoLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUwMzNlNzkzZmE3ZDQzZDRiMjRkZDdkMzY4ZmE5MTgwLnNldENvbnRlbnQoaHRtbF9iYmZlZWQyMzIyZDA0NzExODhiNTUwODllNGQyNjEzZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZTE5ZTk5YjA4MmU0MjI2OTNlNGJhOGQwYTNhODZiYi5iaW5kUG9wdXAocG9wdXBfNTAzM2U3OTNmYTdkNDNkNGIyNGRkN2QzNjhmYTkxODApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzJmYzE5NTJmZGMxNDMyZjlhNGRkNzE1YWIzNGUzNGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMzU5ODQ2ZTE5YTA0YmU5YTUwYzZmZGNjYzM5MThmOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZWIxMmQzZjhhZmU0Nzk1ODI2MDgxMWY1YzJmNjY0NSA9ICQoJzxkaXYgaWQ9Imh0bWxfMGViMTJkM2Y4YWZlNDc5NTgyNjA4MTFmNWMyZjY2NDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMzU5ODQ2ZTE5YTA0YmU5YTUwYzZmZGNjYzM5MThmOS5zZXRDb250ZW50KGh0bWxfMGViMTJkM2Y4YWZlNDc5NTgyNjA4MTFmNWMyZjY2NDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzJmYzE5NTJmZGMxNDMyZjlhNGRkNzE1YWIzNGUzNGUuYmluZFBvcHVwKHBvcHVwXzEzNTk4NDZlMTlhMDRiZTlhNTBjNmZkY2NjMzkxOGY5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg3Zjk0MDZjYWY4ZDRhNGI4YWMwNjE5YjRlM2VlZDE3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzEwMjRjOGIxNTljZTQxY2NiNzg2N2UzZTczMDViNTQwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JkMmEzM2I2N2YxZjQ5MDhhOGZmYzNkNmU2MTMyOGM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M5ZTFkM2RhYTg0ZTQ1NWRhZmQ5NTQ0NmIxNWE2OTQ1ID0gJCgnPGRpdiBpZD0iaHRtbF9jOWUxZDNkYWE4NGU0NTVkYWZkOTU0NDZiMTVhNjk0NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgcmVwbHkgbWFpbCBQcm9jZXNzaW5nIENlbnRyZSwgU291dGggQ2VudHJhbCBMZXR0ZXIgUHJvY2Vzc2luZyBQbGFudCBUb3JvbnRvLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JkMmEzM2I2N2YxZjQ5MDhhOGZmYzNkNmU2MTMyOGM0LnNldENvbnRlbnQoaHRtbF9jOWUxZDNkYWE4NGU0NTVkYWZkOTU0NDZiMTVhNjk0NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84N2Y5NDA2Y2FmOGQ0YTRiOGFjMDYxOWI0ZTNlZWQxNy5iaW5kUG9wdXAocG9wdXBfYmQyYTMzYjY3ZjFmNDkwOGE4ZmZjM2Q2ZTYxMzI4YzQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjMyMWJjNzQ3YzNhNDE4MTkzMzkyNTc2MjM3OTgzYzYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzYyNTc5LC03OS40OTg1MDkwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8xMDI0YzhiMTU5Y2U0MWNjYjc4NjdlM2U3MzA1YjU0MCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kYTk0MGRmOGNhZTY0NTZhYmMzYjE2YmJkYjJmNWNmMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNmJlMDk4Mjg5ODk0NDQ3YTgzZmU0YzM3ZjE4ODg5MSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzZiZTA5ODI4OTg5NDQ0N2E4M2ZlNGMzN2YxODg4OTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9sZCBNaWxsIFNvdXRoLCBLaW5nJiMzOTtzIE1pbGwgUGFyaywgU3VubnlsZWEsIEh1bWJlciBCYXksIE1pbWljbyBORSwgVGhlIFF1ZWVuc3dheSBFYXN0LCBSb3lhbCBZb3JrIFNvdXRoIEVhc3QsIEtpbmdzd2F5IFBhcmsgU291dGggRWFzdCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYTk0MGRmOGNhZTY0NTZhYmMzYjE2YmJkYjJmNWNmMi5zZXRDb250ZW50KGh0bWxfMzZiZTA5ODI4OTg5NDQ0N2E4M2ZlNGMzN2YxODg4OTEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjMyMWJjNzQ3YzNhNDE4MTkzMzkyNTc2MjM3OTgzYzYuYmluZFBvcHVwKHBvcHVwX2RhOTQwZGY4Y2FlNjQ1NmFiYzNiMTZiYmRiMmY1Y2YyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzA2ZDhhMGE0MTgwNjQzOWM5ZTVmODRjNDY2ZTRjNzliID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4ODQwOCwtNzkuNTIwOTk5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMTAyNGM4YjE1OWNlNDFjY2I3ODY3ZTNlNzMwNWI1NDApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjkzYTMwZGE5OTJiNDJmZTgwNTgwOTg4YTlkMjhjNWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWNkYWMxNDc5MTZjNGYyMDgyNGIyMjNiMzU1N2I4YWUgPSAkKCc8ZGl2IGlkPSJodG1sXzFjZGFjMTQ3OTE2YzRmMjA4MjRiMjIzYjM1NTdiOGFlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaW1pY28gTlcsIFRoZSBRdWVlbnN3YXkgV2VzdCwgU291dGggb2YgQmxvb3IsIEtpbmdzd2F5IFBhcmsgU291dGggV2VzdCwgUm95YWwgWW9yayBTb3V0aCBXZXN0LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI5M2EzMGRhOTkyYjQyZmU4MDU4MDk4OGE5ZDI4YzVmLnNldENvbnRlbnQoaHRtbF8xY2RhYzE0NzkxNmM0ZjIwODI0YjIyM2IzNTU3YjhhZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNmQ4YTBhNDE4MDY0MzljOWU1Zjg0YzQ2NmU0Yzc5Yi5iaW5kUG9wdXAocG9wdXBfMjkzYTMwZGE5OTJiNDJmZTgwNTgwOTg4YTlkMjhjNWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>




```python
map_toronto = folium.Map(location = [latitude_toronto, longitude_toronto], zoom_start = 10)

for lat, lng, borough, neighborhood in zip(ds['Latitude'], ds['Longitude'], ds['Borough'], ds['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzNDgxNywtNzkuMzgzOTM0N10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMCwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfNWVmNWE3ZjQwNTAxNGUwZGI4ZjUwYTg0MWQ2NGNhZDkgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhmMDNlMDlhMDA5YzQzZjRiYjQwNjQxNzA4Y2JjYjkwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUzMjU4NiwtNzkuMzI5NjU2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNDk3MTcyZDQ3ZTA0NTUyOWYxMzhjNmU0MWVkYWJmNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMDI5ZjlkY2JmYTQ0YzZhYjA0MGY3YzAyMzMxZWE5OSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTAyOWY5ZGNiZmE0NGM2YWIwNDBmN2MwMjMzMWVhOTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmt3b29kcywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzQ5NzE3MmQ0N2UwNDU1MjlmMTM4YzZlNDFlZGFiZjYuc2V0Q29udGVudChodG1sX2UwMjlmOWRjYmZhNDRjNmFiMDQwZjdjMDIzMzFlYTk5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhmMDNlMDlhMDA5YzQzZjRiYjQwNjQxNzA4Y2JjYjkwLmJpbmRQb3B1cChwb3B1cF9jNDk3MTcyZDQ3ZTA0NTUyOWYxMzhjNmU0MWVkYWJmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iOGUyNjViZmI5NzU0ODQ4YjI2YjQ1ZjAxN2QzZTM0NyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg4MjI5OTk5OTk5NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTI4MDc3MTlmNTRiNDQ4MThiODE1N2QxM2JmODY5YTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzM5NGZjYjU2YjY0NGJlMjliZDY3YzNhZDM1NTk1NzAgPSAkKCc8ZGl2IGlkPSJodG1sXzczOTRmY2I1NmI2NDRiZTI5YmQ2N2MzYWQzNTU5NTcwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5WaWN0b3JpYSBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMjgwNzcxOWY1NGI0NDgxOGI4MTU3ZDEzYmY4NjlhNy5zZXRDb250ZW50KGh0bWxfNzM5NGZjYjU2YjY0NGJlMjliZDY3YzNhZDM1NTk1NzApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjhlMjY1YmZiOTc1NDg0OGIyNmI0NWYwMTdkM2UzNDcuYmluZFBvcHVwKHBvcHVwX2UyODA3NzE5ZjU0YjQ0ODE4YjgxNTdkMTNiZjg2OWE3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UxY2RkYjdlZDY2ZDQ3YWE4ZDVhYmFhYWIwNWVjYjc4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMmNmYzcxMTZlYmU0OTY2YTg2NGYyZDMxYmI1ZjZkMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MTYzNzNkMWMxOTY0ZGFlOWQwNzhmZmE2MmQzNDY5MiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzE2MzczZDFjMTk2NGRhZTlkMDc4ZmZhNjJkMzQ2OTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZ2VudCBQYXJrLCBIYXJib3VyZnJvbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEyY2ZjNzExNmViZTQ5NjZhODY0ZjJkMzFiYjVmNmQzLnNldENvbnRlbnQoaHRtbF83MTYzNzNkMWMxOTY0ZGFlOWQwNzhmZmE2MmQzNDY5Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMWNkZGI3ZWQ2NmQ0N2FhOGQ1YWJhYWFiMDVlY2I3OC5iaW5kUG9wdXAocG9wdXBfMTJjZmM3MTE2ZWJlNDk2NmE4NjRmMmQzMWJiNWY2ZDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmU3ODM4OThhMDA3NDBmNmI0Nzc2YjA5OWNmZDRmNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTg1MTc5OTk5OTk5OTYsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MyMjU1MGE5OTgyZDRhZDE4MGViNzIwYmY1MDQzYmQxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q4OGVmYTEzMmRiMDRhNjdhMmEyYTk2ZjMzMzVmZWNjID0gJCgnPGRpdiBpZD0iaHRtbF9kODhlZmExMzJkYjA0YTY3YTJhMmE5NmYzMzM1ZmVjYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgTWFub3IsIExhd3JlbmNlIEhlaWdodHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MyMjU1MGE5OTgyZDRhZDE4MGViNzIwYmY1MDQzYmQxLnNldENvbnRlbnQoaHRtbF9kODhlZmExMzJkYjA0YTY3YTJhMmE5NmYzMzM1ZmVjYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ZTc4Mzg5OGEwMDc0MGY2YjQ3NzZiMDk5Y2ZkNGY2OC5iaW5kUG9wdXAocG9wdXBfYzIyNTUwYTk5ODJkNGFkMTgwZWI3MjBiZjUwNDNiZDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2NmYmE2OWM2ZjI0NDQ5Y2EyZjExNWYwZjJhYzkwNDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjIzMDE1LC03OS4zODk0OTM4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZmZjdlYTJlNzVmNTQyODZhM2U0MzE1Mzc4NmY1NzY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M5OTU1OTA4MDBlOTQ5MTI4ZGQ4YzI2MTIzYjg3ZDkxID0gJCgnPGRpdiBpZD0iaHRtbF9jOTk1NTkwODAwZTk0OTEyOGRkOGMyNjEyM2I4N2Q5MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UXVlZW4mIzM5O3MgUGFyaywgT250YXJpbyBQcm92aW5jaWFsIEdvdmVybm1lbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZmZjdlYTJlNzVmNTQyODZhM2U0MzE1Mzc4NmY1NzY3LnNldENvbnRlbnQoaHRtbF9jOTk1NTkwODAwZTk0OTEyOGRkOGMyNjEyM2I4N2Q5MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83Y2ZiYTY5YzZmMjQ0NDljYTJmMTE1ZjBmMmFjOTA0Ny5iaW5kUG9wdXAocG9wdXBfNmZmN2VhMmU3NWY1NDI4NmEzZTQzMTUzNzg2ZjU3NjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2ZlODNkZTVmM2NiNDIzZGE3MjI5M2YwZTY2MGVjZmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc4NTU2LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82Y2NjMWZhOGYwYjg0YTQyOTZmYmQ4ZmQ5ZmI0YmUwMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81YjM3ZDliOTQ3MGI0NGRmYTczOTNiNTgzMDVkNmFkMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNWIzN2Q5Yjk0NzBiNDRkZmE3MzkzYjU4MzA1ZDZhZDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPklzbGluZ3RvbiBBdmVudWUsIEh1bWJlciBWYWxsZXkgVmlsbGFnZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82Y2NjMWZhOGYwYjg0YTQyOTZmYmQ4ZmQ5ZmI0YmUwMy5zZXRDb250ZW50KGh0bWxfNWIzN2Q5Yjk0NzBiNDRkZmE3MzkzYjU4MzA1ZDZhZDMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2ZlODNkZTVmM2NiNDIzZGE3MjI5M2YwZTY2MGVjZmUuYmluZFBvcHVwKHBvcHVwXzZjY2MxZmE4ZjBiODRhNDI5NmZiZDhmZDlmYjRiZTAzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdhZGU1MmI4NWEzZDRhMDU5ZmRiNWZlNTUwNGIyNzIxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMmEyNjlmMzE0ODc0OTk1ODc3OTIwZmU4ZWIxNmE5ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYzI5OTFjODAwNjk0ODllOTc2ODYxNTgyMTJiYjllZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmMyOTkxYzgwMDY5NDg5ZTk3Njg2MTU4MjEyYmI5ZWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hbHZlcm4sIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjJhMjY5ZjMxNDg3NDk5NTg3NzkyMGZlOGViMTZhOWUuc2V0Q29udGVudChodG1sXzJjMjk5MWM4MDA2OTQ4OWU5NzY4NjE1ODIxMmJiOWVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdhZGU1MmI4NWEzZDRhMDU5ZmRiNWZlNTUwNGIyNzIxLmJpbmRQb3B1cChwb3B1cF9mMmEyNjlmMzE0ODc0OTk1ODc3OTIwZmU4ZWIxNmE5ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZjJkOTY3NThhYjI0M2JlYTI3N2NhOTkyYzZkZjYxMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc0NTkwNTc5OTk5OTk5NiwtNzkuMzUyMTg4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUwODg0NDYyNDMyZDQ2MzU5ZmYzN2JlYjdkOWFjMDZjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNiMzBjOGYzZDFhMTQ5OTM4NDY5YTRlYjE1MTNlNjQyID0gJCgnPGRpdiBpZD0iaHRtbF8zYjMwYzhmM2QxYTE0OTkzODQ2OWE0ZWIxNTEzZTY0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MDg4NDQ2MjQzMmQ0NjM1OWZmMzdiZWI3ZDlhYzA2Yy5zZXRDb250ZW50KGh0bWxfM2IzMGM4ZjNkMWExNDk5Mzg0NjlhNGViMTUxM2U2NDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWYyZDk2NzU4YWIyNDNiZWEyNzdjYTk5MmM2ZGY2MTMuYmluZFBvcHVwKHBvcHVwXzUwODg0NDYyNDMyZDQ2MzU5ZmYzN2JlYjdkOWFjMDZjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI4Y2E5NTZkNzNhMDRlN2JhOTk3MTQwMmU4M2ExY2JiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2Mzk3MiwtNzkuMzA5OTM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2VhMWQ1ZGJiYzI3NDRmZjNiN2FkMWVlMWI2MjU3YjllID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUzNDUwMzMyZTA4MDRhMzViZWYwNWU5MzI4YWYyOTQwID0gJCgnPGRpdiBpZD0iaHRtbF81MzQ1MDMzMmUwODA0YTM1YmVmMDVlOTMyOGFmMjk0MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UGFya3ZpZXcgSGlsbCwgV29vZGJpbmUgR2FyZGVucywgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYTFkNWRiYmMyNzQ0ZmYzYjdhZDFlZTFiNjI1N2I5ZS5zZXRDb250ZW50KGh0bWxfNTM0NTAzMzJlMDgwNGEzNWJlZjA1ZTkzMjhhZjI5NDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjhjYTk1NmQ3M2EwNGU3YmE5OTcxNDAyZTgzYTFjYmIuYmluZFBvcHVwKHBvcHVwX2VhMWQ1ZGJiYzI3NDRmZjNiN2FkMWVlMWI2MjU3YjllKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2JhMDUzNmZjYWIxODQ2MjA5MjE3ZWRiZDBkMmE3NTg0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTA4ZDIzMjdiNGY4NGRlOGFkNWM1ZGFmNDgyMTZmYTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjcwOGE5ZDM0YTQ4NGJmYzg1OGFhYjVjZWVhZDA2M2MgPSAkKCc8ZGl2IGlkPSJodG1sX2I3MDhhOWQzNGE0ODRiZmM4NThhYWI1Y2VlYWQwNjNjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5HYXJkZW4gRGlzdHJpY3QsIFJ5ZXJzb24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkwOGQyMzI3YjRmODRkZThhZDVjNWRhZjQ4MjE2ZmE2LnNldENvbnRlbnQoaHRtbF9iNzA4YTlkMzRhNDg0YmZjODU4YWFiNWNlZWFkMDYzYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iYTA1MzZmY2FiMTg0NjIwOTIxN2VkYmQwZDJhNzU4NC5iaW5kUG9wdXAocG9wdXBfOTA4ZDIzMjdiNGY4NGRlOGFkNWM1ZGFmNDgyMTZmYTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzY4MGE2MmJmN2I3NGZkNzk2N2U3NTc4ZjJmNGIzMzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDk1NzcsLTc5LjQ0NTA3MjU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2IyMzZlYmQ4YWRlYTQ0MDliZWRhNGM2NWIxNmVmN2Q4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UwMTlkOWU2YjA1MDRhMTg4ZGU1ZTZmYmNmZTIxYjE3ID0gJCgnPGRpdiBpZD0iaHRtbF9lMDE5ZDllNmIwNTA0YTE4OGRlNWU2ZmJjZmUyMWIxNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2xlbmNhaXJuLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMjM2ZWJkOGFkZWE0NDA5YmVkYTRjNjViMTZlZjdkOC5zZXRDb250ZW50KGh0bWxfZTAxOWQ5ZTZiMDUwNGExODhkZTVlNmZiY2ZlMjFiMTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzY4MGE2MmJmN2I3NGZkNzk2N2U3NTc4ZjJmNGIzMzguYmluZFBvcHVwKHBvcHVwX2IyMzZlYmQ4YWRlYTQ0MDliZWRhNGM2NWIxNmVmN2Q4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVjZTEwZGQ3MjhlZjQyZDY4NTlkNjRlM2QyYWUxZTUxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwOTQzMiwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTdkZjIwY2M4YmYxNDY4ZTg4NjFjY2Y1YjcxZGNjMTkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzJlMWY4ZGRlNGM0NGEzYzgwMmZmOWZmZGM1MTZlZTIgPSAkKCc8ZGl2IGlkPSJodG1sXzMyZTFmOGRkZTRjNDRhM2M4MDJmZjlmZmRjNTE2ZWUyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IERlYW5lIFBhcmssIFByaW5jZXNzIEdhcmRlbnMsIE1hcnRpbiBHcm92ZSwgSXNsaW5ndG9uLCBDbG92ZXJkYWxlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU3ZGYyMGNjOGJmMTQ2OGU4ODYxY2NmNWI3MWRjYzE5LnNldENvbnRlbnQoaHRtbF8zMmUxZjhkZGU0YzQ0YTNjODAyZmY5ZmZkYzUxNmVlMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81Y2UxMGRkNzI4ZWY0MmQ2ODU5ZDY0ZTNkMmFlMWU1MS5iaW5kUG9wdXAocG9wdXBfNTdkZjIwY2M4YmYxNDY4ZTg4NjFjY2Y1YjcxZGNjMTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmQwNDQzYjQ3OWI1NDRkYTllYjZhZTQ5YzhkMDU4NWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODQ1MzUxLC03OS4xNjA0OTcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYzgxZjRmNDNlMmU0ODhhOGYxOTQ0NTcwNzVjMjM3MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YjZhN2E4N2FmYjc0YWYzYWQ1ZTU4YzY3YTYyNjRmNCA9ICQoJzxkaXYgaWQ9Imh0bWxfOGI2YTdhODdhZmI3NGFmM2FkNWU1OGM2N2E2MjY0ZjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvdWdlIEhpbGwsIFBvcnQgVW5pb24sIEhpZ2hsYW5kIENyZWVrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M4MWY0ZjQzZTJlNDg4YThmMTk0NDU3MDc1YzIzNzEuc2V0Q29udGVudChodG1sXzhiNmE3YTg3YWZiNzRhZjNhZDVlNThjNjdhNjI2NGY0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZkMDQ0M2I0NzliNTQ0ZGE5ZWI2YWU0OWM4ZDA1ODVjLmJpbmRQb3B1cChwb3B1cF9jYzgxZjRmNDNlMmU0ODhhOGYxOTQ0NTcwNzVjMjM3MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80OWVmYmNlMTZmNDM0ZTIyOTEyMmU5MzQ4ZGUwYWM0MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg5OTcwMDAwMDAxLC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmM3ZWU2NDA4NDhlNDc1ZjljYWRiMDdhZTMyNmFkZDggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjg2Y2FmY2E5MTI4NGRhZWIyMWFmMWMyZjUzZWE2NzcgPSAkKCc8ZGl2IGlkPSJodG1sX2I4NmNhZmNhOTEyODRkYWViMjFhZjFjMmY1M2VhNjc3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb24gTWlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZjN2VlNjQwODQ4ZTQ3NWY5Y2FkYjA3YWUzMjZhZGQ4LnNldENvbnRlbnQoaHRtbF9iODZjYWZjYTkxMjg0ZGFlYjIxYWYxYzJmNTNlYTY3Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80OWVmYmNlMTZmNDM0ZTIyOTEyMmU5MzQ4ZGUwYWM0MC5iaW5kUG9wdXAocG9wdXBfNmM3ZWU2NDA4NDhlNDc1ZjljYWRiMDdhZTMyNmFkZDgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmJjMWJmMzc4ODM5NDBjMmJjZGE3YWNlNDgyYjk2YjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTUzNDM5MDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDRjNjM2MmNjOWE0NDU4Mzk2MjBlMDE0Y2JhY2ZhMmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWRkMThkZTIwNDUwNGRkYjk5MWE0YWQyMzA4ZWVhODAgPSAkKCc8ZGl2IGlkPSJodG1sX2FkZDE4ZGUyMDQ1MDRkZGI5OTFhNGFkMjMwOGVlYTgwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb29kYmluZSBIZWlnaHRzLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ0YzYzNjJjYzlhNDQ1ODM5NjIwZTAxNGNiYWNmYTJhLnNldENvbnRlbnQoaHRtbF9hZGQxOGRlMjA0NTA0ZGRiOTkxYTRhZDIzMDhlZWE4MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYmMxYmYzNzg4Mzk0MGMyYmNkYTdhY2U0ODJiOTZiMy5iaW5kUG9wdXAocG9wdXBfNDRjNjM2MmNjOWE0NDU4Mzk2MjBlMDE0Y2JhY2ZhMmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDMxMzhhY2M2ZjhmNDllYmFmNWEyMjQxYWJkZDY3ZGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NlMGNlMGNlMTY5ZDQ5YzQ5OTdhMTU5ZDY2ZDcxOTk2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVjZTgxNGU0YjU0MTRkYzU4YWE0YTVjMDRmNzAxZjYxID0gJCgnPGRpdiBpZD0iaHRtbF81Y2U4MTRlNGI1NDE0ZGM1OGFhNGE1YzA0ZjcwMWY2MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NlMGNlMGNlMTY5ZDQ5YzQ5OTdhMTU5ZDY2ZDcxOTk2LnNldENvbnRlbnQoaHRtbF81Y2U4MTRlNGI1NDE0ZGM1OGFhNGE1YzA0ZjcwMWY2MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMzEzOGFjYzZmOGY0OWViYWY1YTIyNDFhYmRkNjdkYi5iaW5kUG9wdXAocG9wdXBfY2UwY2UwY2UxNjlkNDljNDk5N2ExNTlkNjZkNzE5OTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjgyNzUzMDZmMjFjNDMwOWJhYzhmMzYzZTJiMTFiNDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTM3ODEzLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ZGU4MWY4YTg4MmQ0YmUzYjQ0NDI3NTljODI0M2ZiZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jODNjMjhlZmRlZGM0ZmI3YWM0ZWU4YWFmYTY1ZDg3NCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzgzYzI4ZWZkZWRjNGZiN2FjNGVlOGFhZmE2NWQ4NzQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWV3b29kLUNlZGFydmFsZSwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWRlODFmOGE4ODJkNGJlM2I0NDQyNzU5YzgyNDNmYmYuc2V0Q29udGVudChodG1sX2M4M2MyOGVmZGVkYzRmYjdhYzRlZThhYWZhNjVkODc0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I4Mjc1MzA2ZjIxYzQzMDliYWM4ZjM2M2UyYjExYjQzLmJpbmRQb3B1cChwb3B1cF81ZGU4MWY4YTg4MmQ0YmUzYjQ0NDI3NTljODI0M2ZiZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81YTA0NzhmNjkxODk0YjQzOGE3MjdhMjBhOTUyYmYxNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MzUxNTIsLTc5LjU3NzIwMDc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QzNmI4MzFkMGU5ZDRmZThhMjg1YzRmYTViZDVkZWEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M5MTFlZTM3MTgyODRiYzBhZDgxNzAwNzFhNjczNjUyID0gJCgnPGRpdiBpZD0iaHRtbF9jOTExZWUzNzE4Mjg0YmMwYWQ4MTcwMDcxYTY3MzY1MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RXJpbmdhdGUsIEJsb29yZGFsZSBHYXJkZW5zLCBPbGQgQnVybmhhbXRob3JwZSwgTWFya2xhbmQgV29vZCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMzZiODMxZDBlOWQ0ZmU4YTI4NWM0ZmE1YmQ1ZGVhMC5zZXRDb250ZW50KGh0bWxfYzkxMWVlMzcxODI4NGJjMGFkODE3MDA3MWE2NzM2NTIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWEwNDc4ZjY5MTg5NGI0MzhhNzI3YTIwYTk1MmJmMTQuYmluZFBvcHVwKHBvcHVwX2QzNmI4MzFkMGU5ZDRmZThhMjg1YzRmYTViZDVkZWEwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U4ZGUzNmIwODVjZjQ0YTRiYmRlYTgwZGZlODE0MTU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzYzNTcyNiwtNzkuMTg4NzExNV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NjEwNDY5MDQ3YTE0ZWE1YTFhMWQxNDA4YTVjODI0OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NWIwZjY5MTNmNzM0ZWJkODA5OTFhOTA3Y2JiYmQ2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfODViMGY2OTEzZjczNGViZDgwOTkxYTkwN2NiYmJkNmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1aWxkd29vZCwgTW9ybmluZ3NpZGUsIFdlc3QgSGlsbCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk2MTA0NjkwNDdhMTRlYTVhMWExZDE0MDhhNWM4MjQ5LnNldENvbnRlbnQoaHRtbF84NWIwZjY5MTNmNzM0ZWJkODA5OTFhOTA3Y2JiYmQ2ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lOGRlMzZiMDg1Y2Y0NGE0YmJkZWE4MGRmZTgxNDE1OC5iaW5kUG9wdXAocG9wdXBfOTYxMDQ2OTA0N2ExNGVhNWExYTFkMTQwOGE1YzgyNDkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzBiZjZlZDA2YzExNDI4MjhhNWQyOGE3MjIwMGI4ZTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzYzNTczOTk5OTk5OSwtNzkuMjkzMDMxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wYTgxNjMxZjY2MDk0NDg1OWYyOGRkZWNjNjYwNjkyNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMjZiMWI5Mzc0NjE0YTEzYTYzZmQwYjZkMzM2ODY0ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzI2YjFiOTM3NDYxNGExM2E2M2ZkMGI2ZDMzNjg2NGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBhODE2MzFmNjYwOTQ0ODU5ZjI4ZGRlY2M2NjA2OTI0LnNldENvbnRlbnQoaHRtbF8zMjZiMWI5Mzc0NjE0YTEzYTYzZmQwYjZkMzM2ODY0ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MGJmNmVkMDZjMTE0MjgyOGE1ZDI4YTcyMjAwYjhlOC5iaW5kUG9wdXAocG9wdXBfMGE4MTYzMWY2NjA5NDQ4NTlmMjhkZGVjYzY2MDY5MjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjgyMmNiMDc5YjBhNDA2MDgwNWJjZmQxNDkxZDhiNmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzdiMzBmZWNiNmYzNGMzMWIzNjllMDU1ZGE0NzQ1ZTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTYyOTkxMGUwYzExNDViMzkwOTNmZjliNWNhZmY4OGIgPSAkKCc8ZGl2IGlkPSJodG1sXzU2Mjk5MTBlMGMxMTQ1YjM5MDkzZmY5YjVjYWZmODhiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzdiMzBmZWNiNmYzNGMzMWIzNjllMDU1ZGE0NzQ1ZTUuc2V0Q29udGVudChodG1sXzU2Mjk5MTBlMGMxMTQ1YjM5MDkzZmY5YjVjYWZmODhiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY4MjJjYjA3OWIwYTQwNjA4MDViY2ZkMTQ5MWQ4YjZkLmJpbmRQb3B1cChwb3B1cF8zN2IzMGZlY2I2ZjM0YzMxYjM2OWUwNTVkYTQ3NDVlNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMTdjMjNlNjhlZTk0MTcwYWE2NDJmN2I2N2RhM2RmNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4OTAyNTYsLTc5LjQ1MzUxMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85Njg3ZTQzOWFiYmE0MzczYjYzZWRkMGQ3ODM2NDlhZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kN2RiYTQyMDUxYzQ0ZjE4ODY3ZTJmNzkwMjQyMmMxYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDdkYmE0MjA1MWM0NGYxODg2N2UyZjc5MDI0MjJjMWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbGVkb25pYS1GYWlyYmFua3MsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk2ODdlNDM5YWJiYTQzNzNiNjNlZGQwZDc4MzY0OWFmLnNldENvbnRlbnQoaHRtbF9kN2RiYTQyMDUxYzQ0ZjE4ODY3ZTJmNzkwMjQyMmMxYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lMTdjMjNlNjhlZTk0MTcwYWE2NDJmN2I2N2RhM2RmNy5iaW5kUG9wdXAocG9wdXBfOTY4N2U0MzlhYmJhNDM3M2I2M2VkZDBkNzgzNjQ5YWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjEyZTRkYzU2N2M4NGJmMzljMTQ3ZjYwZjY5MGJhNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzA5OTIxLC03OS4yMTY5MTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84MzI3ODJlNzQwYzY0MjAxYTMyZWRkYjE4MGZhZWQ0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYzVmYjVlN2I2Nzk0MTczYjBhNzgxMGI5N2ViNGZkZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZWM1ZmI1ZTdiNjc5NDE3M2IwYTc4MTBiOTdlYjRmZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvYnVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgzMjc4MmU3NDBjNjQyMDFhMzJlZGRiMTgwZmFlZDRjLnNldENvbnRlbnQoaHRtbF9lYzVmYjVlN2I2Nzk0MTczYjBhNzgxMGI5N2ViNGZkZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82MTJlNGRjNTY3Yzg0YmYzOWMxNDdmNjBmNjkwYmE1NS5iaW5kUG9wdXAocG9wdXBfODMyNzgyZTc0MGM2NDIwMWEzMmVkZGIxODBmYWVkNGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTM5Y2YyZDNmODRhNGNjODlhNjc5NDVhYTkxZDgyNzEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDkwNjA0LC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E2Y2I2ZTZlYzMzZTQxNzViMGY0MGEyODA4MDhlOTZlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzljN2ZhMThjNTA4ZDQ2YjNiZDhjN2Q0MWJiMjdlOTU5ID0gJCgnPGRpdiBpZD0iaHRtbF85YzdmYTE4YzUwOGQ0NmIzYmQ4YzdkNDFiYjI3ZTk1OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGVhc2lkZSwgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNmNiNmU2ZWMzM2U0MTc1YjBmNDBhMjgwODA4ZTk2ZS5zZXRDb250ZW50KGh0bWxfOWM3ZmExOGM1MDhkNDZiM2JkOGM3ZDQxYmIyN2U5NTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTM5Y2YyZDNmODRhNGNjODlhNjc5NDVhYTkxZDgyNzEuYmluZFBvcHVwKHBvcHVwX2E2Y2I2ZTZlYzMzZTQxNzViMGY0MGEyODA4MDhlOTZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkzNmQ4ZDZjOGYxNTRkMjc4NjhiZTQyYjU4ZWNmNWMwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NmQ4OGYyZDE0YTQ0MDM3YmIyYTkzNGZjZmEyY2E2MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NTI3ZWJmYzI4Njg0MDRhOGIwODVlOTllMzBlMTFlNCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTUyN2ViZmMyODY4NDA0YThiMDg1ZTk5ZTMwZTExZTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjZkODhmMmQxNGE0NDAzN2JiMmE5MzRmY2ZhMmNhNjIuc2V0Q29udGVudChodG1sXzU1MjdlYmZjMjg2ODQwNGE4YjA4NWU5OWUzMGUxMWU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkzNmQ4ZDZjOGYxNTRkMjc4NjhiZTQyYjU4ZWNmNWMwLmJpbmRQb3B1cChwb3B1cF82NmQ4OGYyZDE0YTQ0MDM3YmIyYTkzNGZjZmEyY2E2Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMmRlMjNkNmQxZmM0OWZiYTJmZmM3ODQwMWNlZjgzMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ODU4ZWNhMzYyNWI0NGQ1OGNlN2Q3NjEzMWY1NGFmZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjI3MzY1MjkxMTI0YzFmOTdjYTYzYWNiMjRlODRiNiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjYyNzM2NTI5MTEyNGMxZjk3Y2E2M2FjYjI0ZTg0YjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ODU4ZWNhMzYyNWI0NGQ1OGNlN2Q3NjEzMWY1NGFmZS5zZXRDb250ZW50KGh0bWxfYjYyNzM2NTI5MTEyNGMxZjk3Y2E2M2FjYjI0ZTg0YjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzJkZTIzZDZkMWZjNDlmYmEyZmZjNzg0MDFjZWY4MzAuYmluZFBvcHVwKHBvcHVwXzQ4NThlY2EzNjI1YjQ0ZDU4Y2U3ZDc2MTMxZjU0YWZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY2ZGIwYTI5NjdlMjQyNjVhNDkxOGUwMDU3NDY0ZjQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzczMTM2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kMTY4ZThmZDU2MmY0MDIyOTgxMjNlNmY3NTgxYjAyMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZjJlMGM5MzNhZTM0YmFkYWNlMTU3NzA3NDBhNWU1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfM2YyZTBjOTMzYWUzNGJhZGFjZTE1NzcwNzQwYTVlNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlZGFyYnJhZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QxNjhlOGZkNTYyZjQwMjI5ODEyM2U2Zjc1ODFiMDIyLnNldENvbnRlbnQoaHRtbF8zZjJlMGM5MzNhZTM0YmFkYWNlMTU3NzA3NDBhNWU1YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NmRiMGEyOTY3ZTI0MjY1YTQ5MThlMDA1NzQ2NGY0Mi5iaW5kUG9wdXAocG9wdXBfZDE2OGU4ZmQ1NjJmNDAyMjk4MTIzZTZmNzU4MWIwMjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTRiNTRhZDllOGY0NDVhODlhOTk1NWI5ZTM4YzVlNzQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MDM3NjIyLC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZhN2ZhOTdjNzE4NDRmMTlhODBmMzQ1YmFlZGMxNjViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcwMGJhZmFlMjE0NTRlMDg4NDNmMWYxZGY0OTYzMzg1ID0gJCgnPGRpdiBpZD0iaHRtbF83MDBiYWZhZTIxNDU0ZTA4ODQzZjFmMWRmNDk2MzM4NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGlsbGNyZXN0IFZpbGxhZ2UsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZhN2ZhOTdjNzE4NDRmMTlhODBmMzQ1YmFlZGMxNjViLnNldENvbnRlbnQoaHRtbF83MDBiYWZhZTIxNDU0ZTA4ODQzZjFmMWRmNDk2MzM4NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xNGI1NGFkOWU4ZjQ0NWE4OWE5OTU1YjllMzhjNWU3NC5iaW5kUG9wdXAocG9wdXBfZmE3ZmE5N2M3MTg0NGYxOWE4MGYzNDViYWVkYzE2NWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2MxN2QzZTEzYjY2NGNkNGEzZGNmMjM1YWVkMTRjOWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTQzMjgzLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhhZDhiNTI5OWJiNjQ0ZGE4MTVmMTdmOTMxMzE5Zjc0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I1ZmUzNzljMmRlYjQ3MDI4MWQxYjQ4YzYwYTAyNzBmID0gJCgnPGRpdiBpZD0iaHRtbF9iNWZlMzc5YzJkZWI0NzAyODFkMWI0OGM2MGEwMjcwZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIFdpbHNvbiBIZWlnaHRzLCBEb3duc3ZpZXcgTm9ydGgsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhhZDhiNTI5OWJiNjQ0ZGE4MTVmMTdmOTMxMzE5Zjc0LnNldENvbnRlbnQoaHRtbF9iNWZlMzc5YzJkZWI0NzAyODFkMWI0OGM2MGEwMjcwZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83YzE3ZDNlMTNiNjY0Y2Q0YTNkY2YyMzVhZWQxNGM5ZS5iaW5kUG9wdXAocG9wdXBfOGFkOGI1Mjk5YmI2NDRkYTgxNWYxN2Y5MzEzMTlmNzQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGM5ODM5ODRlODk4NDhhN2I2NGM5MmQ5MWIyOWFjMWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDUzNjg5LC03OS4zNDkzNzE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNmRhNTViNWNjOTE0YjI3YWQzZGJhOTE2NTgyMzAwMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kOTdmZGEwNTkyNzM0YmU3OTcwMjlkODE0NGMxNzAwZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDk3ZmRhMDU5MjczNGJlNzk3MDI5ZDgxNDRjMTcwMGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRob3JuY2xpZmZlIFBhcmssIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzZkYTU1YjVjYzkxNGIyN2FkM2RiYTkxNjU4MjMwMDIuc2V0Q29udGVudChodG1sX2Q5N2ZkYTA1OTI3MzRiZTc5NzAyOWQ4MTQ0YzE3MDBkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRjOTgzOTg0ZTg5ODQ4YTdiNjRjOTJkOTFiMjlhYzFlLmJpbmRQb3B1cChwb3B1cF9jNmRhNTViNWNjOTE0YjI3YWQzZGJhOTE2NTgyMzAwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZDI1ZDMyYWE4ZTI0YTI5OWJlNGRiZGU4NDBkNGNiZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZlZjExZmU3ODJhMjQ5MjY4ODFkMjI3Y2I0YjJiNDgxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVmYjZlNWMyYzBhYTRmY2FiZmEzM2IxNjgwMDM2YTc1ID0gJCgnPGRpdiBpZD0iaHRtbF81ZmI2ZTVjMmMwYWE0ZmNhYmZhMzNiMTY4MDAzNmE3NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaG1vbmQsIEFkZWxhaWRlLCBLaW5nLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ZWYxMWZlNzgyYTI0OTI2ODgxZDIyN2NiNGIyYjQ4MS5zZXRDb250ZW50KGh0bWxfNWZiNmU1YzJjMGFhNGZjYWJmYTMzYjE2ODAwMzZhNzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2QyNWQzMmFhOGUyNGEyOTliZTRkYmRlODQwZDRjYmYuYmluZFBvcHVwKHBvcHVwXzZlZjExZmU3ODJhMjQ5MjY4ODFkMjI3Y2I0YjJiNDgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBkMWJlZTk2NjBiNDQ0YTc4NWFjMTQzOWJjNDU3ZWEyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY5MDA1MTAwMDAwMDEsLTc5LjQ0MjI1OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTMyZGJjMTMwOWEzNGU0OTliZmJmMzgxYTMyNDMxMmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDk5OWU0ZGVhNjY1NDkwYzhjYjFhMDNmMzkyOTlmZGUgPSAkKCc8ZGl2IGlkPSJodG1sXzA5OTllNGRlYTY2NTQ5MGM4Y2IxYTAzZjM5Mjk5ZmRlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EdWZmZXJpbiwgRG92ZXJjb3VydCBWaWxsYWdlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEzMmRiYzEzMDlhMzRlNDk5YmZiZjM4MWEzMjQzMTJmLnNldENvbnRlbnQoaHRtbF8wOTk5ZTRkZWE2NjU0OTBjOGNiMWEwM2YzOTI5OWZkZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZDFiZWU5NjYwYjQ0NGE3ODVhYzE0MzliYzQ1N2VhMi5iaW5kUG9wdXAocG9wdXBfMTMyZGJjMTMwOWEzNGU0OTliZmJmMzgxYTMyNDMxMmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjA3ZTg1ZjQ1NmUzNDlmZjhiOWExMGRkMzA1NjdlYjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NDQ3MzQyLC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOTBhM2RiN2IwZmU0MDJjYmE0ZGI2NmJlODIxZmFiMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjNjYTU4NGVjNWU0OTcxOTYyMTdiYWU4OTFhNmFlZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYjYzY2E1ODRlYzVlNDk3MTk2MjE3YmFlODkxYTZhZWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNjYXJib3JvdWdoIFZpbGxhZ2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOTBhM2RiN2IwZmU0MDJjYmE0ZGI2NmJlODIxZmFiMy5zZXRDb250ZW50KGh0bWxfYjYzY2E1ODRlYzVlNDk3MTk2MjE3YmFlODkxYTZhZWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjA3ZTg1ZjQ1NmUzNDlmZjhiOWExMGRkMzA1NjdlYjYuYmluZFBvcHVwKHBvcHVwXzM5MGEzZGI3YjBmZTQwMmNiYTRkYjY2YmU4MjFmYWIzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhjOTM3ZmQ2YjRhODQyZGNhODFiMmZlZDUyYTM4MDM2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzc4NTE3NSwtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MTcxZTI2NGIyN2Y0YmFlODA4NWU2OTg4NzhhMmY3NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMDdkOThkMzRjOTA0MWUwOTNlYzdhNTU5ZjAyMTVkZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzA3ZDk4ZDM0YzkwNDFlMDkzZWM3YTU1OWYwMjE1ZGUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZhaXJ2aWV3LCBIZW5yeSBGYXJtLCBPcmlvbGUsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkxNzFlMjY0YjI3ZjRiYWU4MDg1ZTY5ODg3OGEyZjc2LnNldENvbnRlbnQoaHRtbF9jMDdkOThkMzRjOTA0MWUwOTNlYzdhNTU5ZjAyMTVkZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YzkzN2ZkNmI0YTg0MmRjYTgxYjJmZWQ1MmEzODAzNi5iaW5kUG9wdXAocG9wdXBfOTE3MWUyNjRiMjdmNGJhZTgwODVlNjk4ODc4YTJmNzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzhiNDFjNmI5ZGJiNDU0ZGEyNmQ2OTA0NTYwOTEzZjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Njc5ODAzLC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNmFlYzFhNWY5NDI0MTA1OTJkNmRhOTllMzRmNDM3OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYTExNTI0ZTMyMDk0ZWViYTBmODY3NTg2MzQ4ZjVjZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmExMTUyNGUzMjA5NGVlYmEwZjg2NzU4NjM0OGY1Y2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod29vZCBQYXJrLCBZb3JrIFVuaXZlcnNpdHksIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I2YWVjMWE1Zjk0MjQxMDU5MmQ2ZGE5OWUzNGY0Mzc5LnNldENvbnRlbnQoaHRtbF8yYTExNTI0ZTMyMDk0ZWViYTBmODY3NTg2MzQ4ZjVjZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83OGI0MWM2YjlkYmI0NTRkYTI2ZDY5MDQ1NjA5MTNmNy5iaW5kUG9wdXAocG9wdXBfYjZhZWMxYTVmOTQyNDEwNTkyZDZkYTk5ZTM0ZjQzNzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDk0ZWJiOGJiNzEzNGU4MWIyZGRiMmEzMjFiZGYwNjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODUzNDcsLTc5LjMzODEwNjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmZhNjdkZWE4YWQ4NGY4YmFhZDAyMDdiYjhlOTNkODUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjJmYzdjN2Q3MWRmNDA3ZWFhYzQ2ZGQ0YzUzYjVjOGYgPSAkKCc8ZGl2IGlkPSJodG1sXzIyZmM3YzdkNzFkZjQwN2VhYWM0NmRkNGM1M2I1YzhmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5FYXN0IFRvcm9udG8sIEJyb2FkdmlldyBOb3J0aCAoT2xkIEVhc3QgWW9yayksIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMmZhNjdkZWE4YWQ4NGY4YmFhZDAyMDdiYjhlOTNkODUuc2V0Q29udGVudChodG1sXzIyZmM3YzdkNzFkZjQwN2VhYWM0NmRkNGM1M2I1YzhmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q5NGViYjhiYjcxMzRlODFiMmRkYjJhMzIxYmRmMDYzLmJpbmRQb3B1cChwb3B1cF8yZmE2N2RlYThhZDg0ZjhiYWFkMDIwN2JiOGU5M2Q4NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YjY3YWNiNzFjNzY0OWNiOTFmOTQ1OWY4NmMxMmM5MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNmN2U5YzVhM2QzMjQyMTA5ZGM4YzBiNmI0MTM0NGU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E2YmIwY2UyNWMzZDRhM2VhNWNiNDhhNWVhOWM4NDdmID0gJCgnPGRpdiBpZD0iaHRtbF9hNmJiMGNlMjVjM2Q0YTNlYTVjYjQ4YTVlYTljODQ3ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIFVuaW9uIFN0YXRpb24sIFRvcm9udG8gSXNsYW5kcywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2Y3ZTljNWEzZDMyNDIxMDlkYzhjMGI2YjQxMzQ0ZTQuc2V0Q29udGVudChodG1sX2E2YmIwY2UyNWMzZDRhM2VhNWNiNDhhNWVhOWM4NDdmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZiNjdhY2I3MWM3NjQ5Y2I5MWY5NDU5Zjg2YzEyYzkwLmJpbmRQb3B1cChwb3B1cF8zZjdlOWM1YTNkMzI0MjEwOWRjOGMwYjZiNDEzNDRlNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZDhiMzczNmU4NjY0MDhkODhjNmFlYzFiZWUwYmQ2YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzkyNjcwMDAwMDAwNiwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZDA5N2E1YmMwMDQ0ZjQ2YmI1MDJkMGY1ZmUzZTg1NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMDFmYThlODNhNmQ0MTM1YTFhZDlhY2E2OWNkNzVkNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjAxZmE4ZTgzYTZkNDEzNWExYWQ5YWNhNjljZDc1ZDUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgVHJpbml0eSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZDA5N2E1YmMwMDQ0ZjQ2YmI1MDJkMGY1ZmUzZTg1NS5zZXRDb250ZW50KGh0bWxfMjAxZmE4ZTgzYTZkNDEzNWExYWQ5YWNhNjljZDc1ZDUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmQ4YjM3MzZlODY2NDA4ZDg4YzZhZWMxYmVlMGJkNmIuYmluZFBvcHVwKHBvcHVwXzlkMDk3YTViYzAwNDRmNDZiYjUwMmQwZjVmZTNlODU1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBiZGE2MjY0MzRhNjRkNzVhMzk4ZWFhY2QxZDc3YTcwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI3OTI5MiwtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfM2JjNWJjOWUzNGViNDVlZGI5M2FjMjMyNDYzMWZjNGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTJhNWI1ZjJiZTg2NDllN2E0NmU3NzZmODYyYTVmMjYgPSAkKCc8ZGl2IGlkPSJodG1sXzEyYTViNWYyYmU4NjQ5ZTdhNDZlNzc2Zjg2MmE1ZjI2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5uZWR5IFBhcmssIElvbnZpZXcsIEVhc3QgQmlyY2htb3VudCBQYXJrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2JjNWJjOWUzNGViNDVlZGI5M2FjMjMyNDYzMWZjNGMuc2V0Q29udGVudChodG1sXzEyYTViNWYyYmU4NjQ5ZTdhNDZlNzc2Zjg2MmE1ZjI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzBiZGE2MjY0MzRhNjRkNzVhMzk4ZWFhY2QxZDc3YTcwLmJpbmRQb3B1cChwb3B1cF8zYmM1YmM5ZTM0ZWI0NWVkYjkzYWMyMzI0NjMxZmM0Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NWY5ZDIzY2I0N2U0NGI5YmUxYjFlNGY5NTY4MzEwNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4Njk0NzMsLTc5LjM4NTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMTNlNGQ4OTk5YWY0Y2QwOGE4YWUyYjRlMDQwODAxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMjkwNDE2YmRmNWE0ZjgxODI1NjBkZjU0N2Y0ZjNiZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjI5MDQxNmJkZjVhNGY4MTgyNTYwZGY1NDdmNGYzYmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJheXZpZXcgVmlsbGFnZSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDEzZTRkODk5OWFmNGNkMDhhOGFlMmI0ZTA0MDgwMWYuc2V0Q29udGVudChodG1sX2YyOTA0MTZiZGY1YTRmODE4MjU2MGRmNTQ3ZjRmM2JlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzU1ZjlkMjNjYjQ3ZTQ0YjliZTFiMWU0Zjk1NjgzMTA0LmJpbmRQb3B1cChwb3B1cF8wMTNlNGQ4OTk5YWY0Y2QwOGE4YWUyYjRlMDQwODAxZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jN2I4ZDBhNTQ3NjA0NDIwOGJmYTBhZjc0NDVmNjQ2YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczNzQ3MzIwMDAwMDAwNCwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGUwYTBmNjcwYmY2NDQzZGE4YTQ3MGY1YjQ4OWRiMDMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjA5ODAyN2UzOGYyNDQzNmI3MWE4OTBmOTVhZmRmNDQgPSAkKCc8ZGl2IGlkPSJodG1sX2YwOTgwMjdlMzhmMjQ0MzZiNzFhODkwZjk1YWZkZjQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRlMGEwZjY3MGJmNjQ0M2RhOGE0NzBmNWI0ODlkYjAzLnNldENvbnRlbnQoaHRtbF9mMDk4MDI3ZTM4ZjI0NDM2YjcxYTg5MGY5NWFmZGY0NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jN2I4ZDBhNTQ3NjA0NDIwOGJmYTBhZjc0NDVmNjQ2Yi5iaW5kUG9wdXAocG9wdXBfNGUwYTBmNjcwYmY2NDQzZGE4YTQ3MGY1YjQ4OWRiMDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTJkYjY3Yjc1OGYzNGYxODlhNTdiYWE4YzI0YjYwZTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNmRhOGFjMzkzYzAxNDYxYWJjYTI3NGE0ZDBlYmFmMTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzRiMmMwNWY2Nzk1NDdlYWIxNjE2MzFlYTU1MjQ5MjcgPSAkKCc8ZGl2IGlkPSJodG1sX2M0YjJjMDVmNjc5NTQ3ZWFiMTYxNjMxZWE1NTI0OTI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgRGFuZm9ydGggV2VzdCwgUml2ZXJkYWxlLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZkYThhYzM5M2MwMTQ2MWFiY2EyNzRhNGQwZWJhZjE0LnNldENvbnRlbnQoaHRtbF9jNGIyYzA1ZjY3OTU0N2VhYjE2MTYzMWVhNTUyNDkyNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMmRiNjdiNzU4ZjM0ZjE4OWE1N2JhYThjMjRiNjBlNS5iaW5kUG9wdXAocG9wdXBfNmRhOGFjMzkzYzAxNDYxYWJjYTI3NGE0ZDBlYmFmMTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2Y1M2NkNmJlNTRmNDA1ZWJjYTkxNWIxZmNkMGJjNTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMDA1ZGJkMjk2MGE0OTYxYTU2NzlmMjJjNmViZDg4NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NDNjZmZlZmUxNjk0OTdjYmRjNGI5MmFlOTJjZWIzMCA9ICQoJzxkaXYgaWQ9Imh0bWxfNjQzY2ZmZWZlMTY5NDk3Y2JkYzRiOTJhZTkyY2ViMzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2UsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IwMDVkYmQyOTYwYTQ5NjFhNTY3OWYyMmM2ZWJkODg1LnNldENvbnRlbnQoaHRtbF82NDNjZmZlZmUxNjk0OTdjYmRjNGI5MmFlOTJjZWIzMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZjUzY2Q2YmU1NGY0MDVlYmNhOTE1YjFmY2QwYmM1MC5iaW5kUG9wdXAocG9wdXBfYjAwNWRiZDI5NjBhNDk2MWE1Njc5ZjIyYzZlYmQ4ODUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODZhMGM2OTk1OTEwNGQ2NDlmYzc2YmI1YmUzNzFlNDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY4NDcyLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83OGRkN2FjYWEwMzk0N2MwOWIwNzNkODQ4ZDEyOTg0YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kNTNmMDkyYWRmMjQ0Y2Q3OWEwNmRhM2FmMTJlNGM5MSA9ICQoJzxkaXYgaWQ9Imh0bWxfZDUzZjA5MmFkZjI0NGNkNzlhMDZkYTNhZjEyZTRjOTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJyb2NrdG9uLCBQYXJrZGFsZSBWaWxsYWdlLCBFeGhpYml0aW9uIFBsYWNlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc4ZGQ3YWNhYTAzOTQ3YzA5YjA3M2Q4NDhkMTI5ODRhLnNldENvbnRlbnQoaHRtbF9kNTNmMDkyYWRmMjQ0Y2Q3OWEwNmRhM2FmMTJlNGM5MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NmEwYzY5OTU5MTA0ZDY0OWZjNzZiYjViZTM3MWU0Ni5iaW5kUG9wdXAocG9wdXBfNzhkZDdhY2FhMDM5NDdjMDliMDczZDg0OGQxMjk4NGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2ZlNjdiMTZiMzhiNGVkNmI5M2U1ZTA0YzljMjZjMjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTExMTE3MDAwMDAwMDQsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2IyNjA4NDZlMmNiNDQ1NjgzOGU1NDlmZGY2YzQyM2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmQ5ZDAxYjVlMzdlNDdmMTkzM2Y1NDE2OTVmZjljNjUgPSAkKCc8ZGl2IGlkPSJodG1sX2JkOWQwMWI1ZTM3ZTQ3ZjE5MzNmNTQxNjk1ZmY5YzY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Hb2xkZW4gTWlsZSwgQ2xhaXJsZWEsIE9ha3JpZGdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2IyNjA4NDZlMmNiNDQ1NjgzOGU1NDlmZGY2YzQyM2Iuc2V0Q29udGVudChodG1sX2JkOWQwMWI1ZTM3ZTQ3ZjE5MzNmNTQxNjk1ZmY5YzY1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NmZTY3YjE2YjM4YjRlZDZiOTNlNWUwNGM5YzI2YzI0LmJpbmRQb3B1cChwb3B1cF9jYjI2MDg0NmUyY2I0NDU2ODM4ZTU0OWZkZjZjNDIzYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZDNmOGVmYjJkOTY0OTE2Yjk4NDE0Mzc2MTBiNjg1MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQ5MDIsLTc5LjM3NDcxNDA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NlZmEwZGJiYjVmNjQzMWU4NTRhNTM4ZDg3MDM1YmYzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU1ZWE0ZDZiNGQyOTQ1ZjBhYmI2Njc2ODIxZDBiZjgyID0gJCgnPGRpdiBpZD0iaHRtbF81NWVhNGQ2YjRkMjk0NWYwYWJiNjY3NjgyMWQwYmY4MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscywgU2lsdmVyIEhpbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jZWZhMGRiYmI1ZjY0MzFlODU0YTUzOGQ4NzAzNWJmMy5zZXRDb250ZW50KGh0bWxfNTVlYTRkNmI0ZDI5NDVmMGFiYjY2NzY4MjFkMGJmODIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWQzZjhlZmIyZDk2NDkxNmI5ODQxNDM3NjEwYjY4NTAuYmluZFBvcHVwKHBvcHVwX2NlZmEwZGJiYjVmNjQzMWU4NTRhNTM4ZDg3MDM1YmYzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFjMjE0YzRhZGUxMzRhOGNhMjlhODkyNzg2NWNkNWQ3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5MDE0NiwtNzkuNTA2OTQzNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZmY3ZmVjMjAwMzU0NDYwOGQzYTQxNGJhNTNmYTJlNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xOTYzM2I3OWQ3YzI0MGM5YTgxMzE3ZjFiZWY5MTAyMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMTk2MzNiNzlkN2MyNDBjOWE4MTMxN2YxYmVmOTEwMjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvd25zdmlldywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGZmN2ZlYzIwMDM1NDQ2MDhkM2E0MTRiYTUzZmEyZTUuc2V0Q29udGVudChodG1sXzE5NjMzYjc5ZDdjMjQwYzlhODEzMTdmMWJlZjkxMDIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFjMjE0YzRhZGUxMzRhOGNhMjlhODkyNzg2NWNkNWQ3LmJpbmRQb3B1cChwb3B1cF80ZmY3ZmVjMjAwMzU0NDYwOGQzYTQxNGJhNTNmYTJlNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMzI0MTJlMmU4YTc0OTMwYTE4YjcxZDJkNGJiZjg5YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I2OGY3ZmJlNDJkNDQ4OTQ5MGJlYTc0MDZiNmFkY2YzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E3MzQ0Mjc5YmQyMTRhNDRiNmZlOWIwNGY3NjBlY2QyID0gJCgnPGRpdiBpZD0iaHRtbF9hNzM0NDI3OWJkMjE0YTQ0YjZmZTliMDRmNzYwZWNkMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBUaGUgQmVhY2hlcyBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I2OGY3ZmJlNDJkNDQ4OTQ5MGJlYTc0MDZiNmFkY2YzLnNldENvbnRlbnQoaHRtbF9hNzM0NDI3OWJkMjE0YTQ0YjZmZTliMDRmNzYwZWNkMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMzI0MTJlMmU4YTc0OTMwYTE4YjcxZDJkNGJiZjg5YS5iaW5kUG9wdXAocG9wdXBfYjY4ZjdmYmU0MmQ0NDg5NDkwYmVhNzQwNmI2YWRjZjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2I4MDg1YzkwYzY1NDAyNDg1NTA0NWI3ZmQwZTQ1MjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNGIxMWI2YTRhMjc0ZDQ5OGEyNDViNzI1ZDc1YzJjYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mOGNhNTc2MTdmODM0MjBiYmI5M2QxNDllN2RiMDY2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZjhjYTU3NjE3ZjgzNDIwYmJiOTNkMTQ5ZTdkYjA2NmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0LCBWaWN0b3JpYSBIb3RlbCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTRiMTFiNmE0YTI3NGQ0OThhMjQ1YjcyNWQ3NWMyY2Euc2V0Q29udGVudChodG1sX2Y4Y2E1NzYxN2Y4MzQyMGJiYjkzZDE0OWU3ZGIwNjZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNiODA4NWM5MGM2NTQwMjQ4NTUwNDViN2ZkMGU0NTI1LmJpbmRQb3B1cChwb3B1cF8xNGIxMWI2YTRhMjc0ZDQ5OGEyNDViNzI1ZDc1YzJjYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yOTdjMTg5NWM2ZTI0NTQ4OWIwY2U2MzU5OGFiNWI3NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMzc1NjIwMDAwMDAwNiwtNzkuNDkwMDczOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lNzYyMjczNzIwNGU0ZDhjYTk2MjNjYmQ4ZTgzY2I5ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81NTkzZDhiYzBiYTg0Y2IzYTIyOTdmZmI3MTNhMjY3YiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTU5M2Q4YmMwYmE4NGNiM2EyMjk3ZmZiNzEzYTI2N2IiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRoIFBhcmssIE1hcGxlIExlYWYgUGFyaywgVXB3b29kIFBhcmssIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2U3NjIyNzM3MjA0ZTRkOGNhOTYyM2NiZDhlODNjYjlmLnNldENvbnRlbnQoaHRtbF81NTkzZDhiYzBiYTg0Y2IzYTIyOTdmZmI3MTNhMjY3Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yOTdjMTg5NWM2ZTI0NTQ4OWIwY2U2MzU5OGFiNWI3NS5iaW5kUG9wdXAocG9wdXBfZTc2MjI3MzcyMDRlNGQ4Y2E5NjIzY2JkOGU4M2NiOWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDVmNjk5ZGU0ZGMxNDgzMmE5NTE5YThkZDg1OTlhZGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTYzMDMzLC03OS41NjU5NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNmFkNTI2MTIzZTA0MWM4YjYxYzZlZWY5N2FhY2Y5ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82YjYyNGQ0ZmYyYWY0NjUyOGZmNmU0ODE4NjVlM2QwZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNmI2MjRkNGZmMmFmNDY1MjhmZjZlNDgxODY1ZTNkMGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlciBTdW1taXQsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y2YWQ1MjYxMjNlMDQxYzhiNjFjNmVlZjk3YWFjZjlmLnNldENvbnRlbnQoaHRtbF82YjYyNGQ0ZmYyYWY0NjUyOGZmNmU0ODE4NjVlM2QwZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNWY2OTlkZTRkYzE0ODMyYTk1MTlhOGRkODU5OWFkYi5iaW5kUG9wdXAocG9wdXBfZjZhZDUyNjEyM2UwNDFjOGI2MWM2ZWVmOTdhYWNmOWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2I1OTMxODdmYmYzNGFhNDk3ODkxMGM1ODk3ODE2ZmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTYzMTYsLTc5LjIzOTQ3NjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU3NmJmYTkzOTUyZDQ5OGI4YTZkMmQyMDI4MzFhNGM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhNTMzN2NiZThkZDQ1MjU5ZWQyNmFmMjNlZDFmYTQ4ID0gJCgnPGRpdiBpZD0iaHRtbF85YTUzMzdjYmU4ZGQ0NTI1OWVkMjZhZjIzZWQxZmE0OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xpZmZzaWRlLCBDbGlmZmNyZXN0LCBTY2FyYm9yb3VnaCBWaWxsYWdlIFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NzZiZmE5Mzk1MmQ0OThiOGE2ZDJkMjAyODMxYTRjOC5zZXRDb250ZW50KGh0bWxfOWE1MzM3Y2JlOGRkNDUyNTllZDI2YWYyM2VkMWZhNDgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2I1OTMxODdmYmYzNGFhNDk3ODkxMGM1ODk3ODE2ZmYuYmluZFBvcHVwKHBvcHVwXzU3NmJmYTkzOTUyZDQ5OGI4YTZkMmQyMDI4MzFhNGM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2I4OGVjZjUyOTQzZjRkZGQ4Y2QzYTlhZTU1YjQ5ZjA3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg5MDUzLC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYjk3OGY3MWU0NGM0Y2NkYTc3MDI0NTUxY2E3ZWY3OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85OTY0ZmQzODk5ZDM0YTY4OTMyNTlmNDg1OGE2Y2I2YSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTk2NGZkMzg5OWQzNGE2ODkzMjU5ZjQ4NThhNmNiNmEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIE5ld3RvbmJyb29rLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jYjk3OGY3MWU0NGM0Y2NkYTc3MDI0NTUxY2E3ZWY3OC5zZXRDb250ZW50KGh0bWxfOTk2NGZkMzg5OWQzNGE2ODkzMjU5ZjQ4NThhNmNiNmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjg4ZWNmNTI5NDNmNGRkZDhjZDNhOWFlNTViNDlmMDcuYmluZFBvcHVwKHBvcHVwX2NiOTc4ZjcxZTQ0YzRjY2RhNzcwMjQ1NTFjYTdlZjc4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg2YjIzNWZiYmM1ZDQ5MjM5YTk0NmU4ZDQwOTQxZDdkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4NDk2NCwtNzkuNDk1Njk3NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzkxYjlmMjJmZDgxNDZhMmFiODRlZTBiYWNkYzNlYjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzJmZGJiMDY2NDQ0NGM5MGE4ODVjNzRmYjBiMjdmMWMgPSAkKCc8ZGl2IGlkPSJodG1sXzMyZmRiYjA2NjQ0NDRjOTBhODg1Yzc0ZmIwYjI3ZjFjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M5MWI5ZjIyZmQ4MTQ2YTJhYjg0ZWUwYmFjZGMzZWI4LnNldENvbnRlbnQoaHRtbF8zMmZkYmIwNjY0NDQ0YzkwYTg4NWM3NGZiMGIyN2YxYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NmIyMzVmYmJjNWQ0OTIzOWE5NDZlOGQ0MDk0MWQ3ZC5iaW5kUG9wdXAocG9wdXBfYzkxYjlmMjJmZDgxNDZhMmFiODRlZTBiYWNkYzNlYjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTVmNmUzZDI1Zjg3NDg1Y2I4MTliMDdkZWQ4MGRmYmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjVkMTEwOGYwNWM3NDQ4ZDk5YjQ2ZDE4NDBiMWFhMDIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjMxY2Q4ZTJlZjZiNGM5OGEzMmQ2N2RmMjFhM2FlZDMgPSAkKCc8ZGl2IGlkPSJodG1sXzIzMWNkOGUyZWY2YjRjOThhMzJkNjdkZjIxYTNhZWQzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdHVkaW8gRGlzdHJpY3QsIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjVkMTEwOGYwNWM3NDQ4ZDk5YjQ2ZDE4NDBiMWFhMDIuc2V0Q29udGVudChodG1sXzIzMWNkOGUyZWY2YjRjOThhMzJkNjdkZjIxYTNhZWQzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1ZjZlM2QyNWY4NzQ4NWNiODE5YjA3ZGVkODBkZmJhLmJpbmRQb3B1cChwb3B1cF9iNWQxMTA4ZjA1Yzc0NDhkOTliNDZkMTg0MGIxYWEwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZGYxOWEwNzk3YjA0YzE3YWUxYjdiOTkwN2RiYzcyZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczMzI4MjUsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWFlZDU5NzViN2MzNGQ1ZGExZTY1ZDlhMDY5MDk0N2UgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWNkZjQzNzgwYjRhNGE3ZTk4YWY3NzM4OWZlYWYyM2UgPSAkKCc8ZGl2IGlkPSJodG1sXzVjZGY0Mzc4MGI0YTRhN2U5OGFmNzczODlmZWFmMjNlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZWRmb3JkIFBhcmssIExhd3JlbmNlIE1hbm9yIEVhc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FhZWQ1OTc1YjdjMzRkNWRhMWU2NWQ5YTA2OTA5NDdlLnNldENvbnRlbnQoaHRtbF81Y2RmNDM3ODBiNGE0YTdlOThhZjc3Mzg5ZmVhZjIzZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZGYxOWEwNzk3YjA0YzE3YWUxYjdiOTkwN2RiYzcyZi5iaW5kUG9wdXAocG9wdXBfYWFlZDU5NzViN2MzNGQ1ZGExZTY1ZDlhMDY5MDk0N2UpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDJjZGUwZTUxMTAxNDYwZmFhOTk0YTgwMTE3MmRmYzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTExMTU4LC03OS40NzYwMTMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wMzMzM2EyNWY2ZTk0Nzc3YTYxODViOGQwN2YwMTc4NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mODY2ZjE2YmNiNDU0YjNkYjEwZGQzODE4NjA5YTIxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfZjg2NmYxNmJjYjQ1NGIzZGIxMGRkMzgxODYwOWEyMTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRlbCBSYXksIE1vdW50IERlbm5pcywgS2VlbHNkYWxlIGFuZCBTaWx2ZXJ0aG9ybiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDMzMzNhMjVmNmU5NDc3N2E2MTg1YjhkMDdmMDE3ODYuc2V0Q29udGVudChodG1sX2Y4NjZmMTZiY2I0NTRiM2RiMTBkZDM4MTg2MDlhMjEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQyY2RlMGU1MTEwMTQ2MGZhYTk5NGE4MDExNzJkZmMwLmJpbmRQb3B1cChwb3B1cF8wMzMzM2EyNWY2ZTk0Nzc3YTYxODViOGQwN2YwMTc4Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYWE5ODE1ZWVhNmU0YzkwOWVlZWE3MjI0YThhMjFkMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNDc2NTksLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q3NjliYzZlZWYzODRkNzM4MjgyNWI3NzY0ZWM4MGNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAwMjIyNWUwODYxMjQ2NzM4MTVkNmU5MDkzNjc2ZGJkID0gJCgnPGRpdiBpZD0iaHRtbF8wMDIyMjVlMDg2MTI0NjczODE1ZDZlOTA5MzY3NmRiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SHVtYmVybGVhLCBFbWVyeSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDc2OWJjNmVlZjM4NGQ3MzgyODI1Yjc3NjRlYzgwY2Iuc2V0Q29udGVudChodG1sXzAwMjIyNWUwODYxMjQ2NzM4MTVkNmU5MDkzNjc2ZGJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VhYTk4MTVlZWE2ZTRjOTA5ZWVlYTcyMjRhOGEyMWQxLmJpbmRQb3B1cChwb3B1cF9kNzY5YmM2ZWVmMzg0ZDczODI4MjViNzc2NGVjODBjYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81NGNmOTA0ODdiNTY0NDUxYmVlMzllYzI5YjliZWEzZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MjY1NzAwMDAwMDAwNCwtNzkuMjY0ODQ4MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80YzhmYjMzODhjNDQ0ZjE0OTQ5ZGQ3MDMxYjYyYjhjMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zMzY4NjExOTlhYWE0ZDQ5YTUzOWMwZDliNjc2ZjY3ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzM2ODYxMTk5YWFhNGQ0OWE1MzljMGQ5YjY3NmY2N2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJpcmNoIENsaWZmLCBDbGlmZnNpZGUgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRjOGZiMzM4OGM0NDRmMTQ5NDlkZDcwMzFiNjJiOGMxLnNldENvbnRlbnQoaHRtbF8zMzY4NjExOTlhYWE0ZDQ5YTUzOWMwZDliNjc2ZjY3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NGNmOTA0ODdiNTY0NDUxYmVlMzllYzI5YjliZWEzZi5iaW5kUG9wdXAocG9wdXBfNGM4ZmIzMzg4YzQ0NGYxNDk0OWRkNzAzMWI2MmI4YzEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmRhZmNlYmU0M2JjNDVlNTgwYTNiOTJhZGEwYzdlOTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzAxMTk5LC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMDYyYzQxMWJlNzE0ODM3YjhlM2NiMmUwOGY5ODBiZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NjFlYmFhNTY2YzU0ZjhkYTRiM2U1NTYxMmZhMzcxZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDYxZWJhYTU2NmM1NGY4ZGE0YjNlNTU2MTJmYTM3MWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIFdpbGxvd2RhbGUgRWFzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzA2MmM0MTFiZTcxNDgzN2I4ZTNjYjJlMDhmOTgwYmQuc2V0Q29udGVudChodG1sXzQ2MWViYWE1NjZjNTRmOGRhNGIzZTU1NjEyZmEzNzFmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZkYWZjZWJlNDNiYzQ1ZTU4MGEzYjkyYWRhMGM3ZTkyLmJpbmRQb3B1cChwb3B1cF9jMDYyYzQxMWJlNzE0ODM3YjhlM2NiMmUwOGY5ODBiZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMmQ4YzlmNWE0N2Y0MWExOTM3NTFkNGVjYTA2M2MzYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MTYzMTMsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I3OWE2OTRiMjlmZDQyOTNhYzUzMTNlNGFlZmZiYzA3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZjYTEyYjliYjZkNzQxMmNhNGUzNjFkZGJkMWI3NmIxID0gJCgnPGRpdiBpZD0iaHRtbF82Y2ExMmI5YmI2ZDc0MTJjYTRlMzYxZGRiZDFiNzZiMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNzlhNjk0YjI5ZmQ0MjkzYWM1MzEzZTRhZWZmYmMwNy5zZXRDb250ZW50KGh0bWxfNmNhMTJiOWJiNmQ3NDEyY2E0ZTM2MWRkYmQxYjc2YjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTJkOGM5ZjVhNDdmNDFhMTkzNzUxZDRlY2EwNjNjM2MuYmluZFBvcHVwKHBvcHVwX2I3OWE2OTRiMjlmZDQyOTNhYzUzMTNlNGFlZmZiYzA3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmOGMwYzQ1NjNjODRjODg5MWM0MmZkOTVjYTM5OTAzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNDI1YWI4NTEzNjE0ZmVmYjZkY2RlMTQ1MDc2Njg5NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZmU4NGQ3ZGRmNGE0ZmE3OWFiNTg1YWRiNjc0ODBiNSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWZlODRkN2RkZjRhNGZhNzlhYjU4NWFkYjY3NDgwYjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIFBhcmssIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDQyNWFiODUxMzYxNGZlZmI2ZGNkZTE0NTA3NjY4OTcuc2V0Q29udGVudChodG1sX2VmZTg0ZDdkZGY0YTRmYTc5YWI1ODVhZGI2NzQ4MGI1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmOGMwYzQ1NjNjODRjODg5MWM0MmZkOTVjYTM5OTAzLmJpbmRQb3B1cChwb3B1cF9kNDI1YWI4NTEzNjE0ZmVmYjZkY2RlMTQ1MDc2Njg5Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZmE3MDdmOGEwMzU0NjE0OGVjZjgxMzU3MDE5YTgwOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTY5NDgsLTc5LjQxNjkzNTU5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U4NjA5NTc0ZDY5MzQwMjdhOTA0NjY2MTA2MzI0OTgxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk2NmIxM2Q0Mjk2MDQxNDE5ZjUzYjE5Y2QxMDYwYWZjID0gJCgnPGRpdiBpZD0iaHRtbF85NjZiMTNkNDI5NjA0MTQxOWY1M2IxOWNkMTA2MGFmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWxhd24sIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTg2MDk1NzRkNjkzNDAyN2E5MDQ2NjYxMDYzMjQ5ODEuc2V0Q29udGVudChodG1sXzk2NmIxM2Q0Mjk2MDQxNDE5ZjUzYjE5Y2QxMDYwYWZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmYTcwN2Y4YTAzNTQ2MTQ4ZWNmODEzNTcwMTlhODA4LmJpbmRQb3B1cChwb3B1cF9lODYwOTU3NGQ2OTM0MDI3YTkwNDY2NjEwNjMyNDk4MSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xZGM4MTFjYzM0YWY0YWQ5YmRlZWFjZTU0YzI0YjY1OSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MzE4NTI5OTk5OTk5LC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81YmE0MTA5MTg4OGM0ZWNlODYyMzBhYTYyYmFjYjQ2YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMWY1ZTgzMjNiNWY0YWUwOTEwN2I3OTRlZjhkYzczZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMTFmNWU4MzIzYjVmNGFlMDkxMDdiNzk0ZWY4ZGM3M2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ1bm55bWVkZSwgVGhlIEp1bmN0aW9uIE5vcnRoLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81YmE0MTA5MTg4OGM0ZWNlODYyMzBhYTYyYmFjYjQ2YS5zZXRDb250ZW50KGh0bWxfMTFmNWU4MzIzYjVmNGFlMDkxMDdiNzk0ZWY4ZGM3M2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWRjODExY2MzNGFmNGFkOWJkZWVhY2U1NGMyNGI2NTkuYmluZFBvcHVwKHBvcHVwXzViYTQxMDkxODg4YzRlY2U4NjIzMGFhNjJiYWNiNDZhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E4NzhjNzdmYmI2ODQwZjc4NDQyOWVmMWE3NjRjMGU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LC03OS41MTgxODg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YjFiM2NiNzNjNzY0MDM0OTZlNWM4MDRjNmRiYmQ2ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xZmFmOWNmYmZiYjU0NTJhOWJkYmEyMjhlYjU2ZDAzYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMWZhZjljZmJmYmI1NDUyYTliZGJhMjI4ZWI1NmQwM2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3RvbiwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGIxYjNjYjczYzc2NDAzNDk2ZTVjODA0YzZkYmJkNmYuc2V0Q29udGVudChodG1sXzFmYWY5Y2ZiZmJiNTQ1MmE5YmRiYTIyOGViNTZkMDNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E4NzhjNzdmYmI2ODQwZjc4NDQyOWVmMWE3NjRjMGU0LmJpbmRQb3B1cChwb3B1cF84YjFiM2NiNzNjNzY0MDM0OTZlNWM4MDRjNmRiYmQ2Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kMmYxYWFlNTUwOTY0N2I4YjRlYzZlOWRlYzIxNzMyMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQwOTYsLTc5LjI3MzMwNDAwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNlNDMxMDBkYjEzOTQ1MWI4ZGYyMzdiMGNjNGM3MDUxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzM2ZmNiZGUxZmE0MDQ4OGI4ODVkOTQzYjQzMjk1YWM0ID0gJCgnPGRpdiBpZD0iaHRtbF8zNmZjYmRlMWZhNDA0ODhiODg1ZDk0M2I0MzI5NWFjNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9yc2V0IFBhcmssIFdleGZvcmQgSGVpZ2h0cywgU2NhcmJvcm91Z2ggVG93biBDZW50cmUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zZTQzMTAwZGIxMzk0NTFiOGRmMjM3YjBjYzRjNzA1MS5zZXRDb250ZW50KGh0bWxfMzZmY2JkZTFmYTQwNDg4Yjg4NWQ5NDNiNDMyOTVhYzQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDJmMWFhZTU1MDk2NDdiOGI0ZWM2ZTlkZWMyMTczMjIuYmluZFBvcHVwKHBvcHVwXzNlNDMxMDBkYjEzOTQ1MWI4ZGYyMzdiMGNjNGM3MDUxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFlNjRmMzAzYzk3YjRiMmI4NTUxZDc2NDgxOTkyZGNiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzUyNzU4Mjk5OTk5OTk2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I3MjkwMjQ3YmFmZTRlOGZiMDU5NDI0MjhlYzc1OTY0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U5N2FkZDg4ZWU5NjQyZWJhZmQ3YzgzZTg3NTE5NjM5ID0gJCgnPGRpdiBpZD0iaHRtbF9lOTdhZGQ4OGVlOTY0MmViYWZkN2M4M2U4NzUxOTYzOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscyBXZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNzI5MDI0N2JhZmU0ZThmYjA1OTQyNDI4ZWM3NTk2NC5zZXRDb250ZW50KGh0bWxfZTk3YWRkODhlZTk2NDJlYmFmZDdjODNlODc1MTk2MzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWU2NGYzMDNjOTdiNGIyYjg1NTFkNzY0ODE5OTJkY2IuYmluZFBvcHVwKHBvcHVwX2I3MjkwMjQ3YmFmZTRlOGZiMDU5NDI0MjhlYzc1OTY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzEyNWRhMTRjZmJmZDRlNTJhYmMzZjQ0MjkzNGVmMWRjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYzlhNWJiOTJlZmM0YjgwODcxNzhkMmU2M2FmMWEwNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hMGQ4ZWM3ZjZkZWQ0Mzg4YWQxMTZjOWU0NDM0MDc5MiA9ICQoJzxkaXYgaWQ9Imh0bWxfYTBkOGVjN2Y2ZGVkNDM4OGFkMTE2YzllNDQzNDA3OTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUgTm9ydGgsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWM5YTViYjkyZWZjNGI4MDg3MTc4ZDJlNjNhZjFhMDYuc2V0Q29udGVudChodG1sX2EwZDhlYzdmNmRlZDQzODhhZDExNmM5ZTQ0MzQwNzkyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEyNWRhMTRjZmJmZDRlNTJhYmMzZjQ0MjkzNGVmMWRjLmJpbmRQb3B1cChwb3B1cF8xYzlhNWJiOTJlZmM0YjgwODcxNzhkMmU2M2FmMWEwNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81ZTMzMTRlMjMzYmM0ZWE3YTNkN2JiYzNkNWU0ODhhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5Njk0NzYsLTc5LjQxMTMwNzIwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E2MmJhYjVmZGE1MDRmOTI4YTMxNjFhNWU5NjgwMmQ0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU4YWNhMGE3ODkzNDQ2ZWRhMjA3NjgxMTg3YzQwMzA5ID0gJCgnPGRpdiBpZD0iaHRtbF81OGFjYTBhNzg5MzQ0NmVkYTIwNzY4MTE4N2M0MDMwOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgTm9ydGggJmFtcDsgV2VzdCwgRm9yZXN0IEhpbGwgUm9hZCBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E2MmJhYjVmZGE1MDRmOTI4YTMxNjFhNWU5NjgwMmQ0LnNldENvbnRlbnQoaHRtbF81OGFjYTBhNzg5MzQ0NmVkYTIwNzY4MTE4N2M0MDMwOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZTMzMTRlMjMzYmM0ZWE3YTNkN2JiYzNkNWU0ODhhMi5iaW5kUG9wdXAocG9wdXBfYTYyYmFiNWZkYTUwNGY5MjhhMzE2MWE1ZTk2ODAyZDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTM5ZDNlYjgyMDY1NDA0MDljMjZmYTc5YzJmZDAyYjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjE2MDgzLC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MzExYzk3MDRmMGM0NDEzYTQ0MjNkYjIxMDM0ZGYxZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYjIzYzc3NDljN2E0ODNlODkxMTNjYzljOWFlNWU1MSA9ICQoJzxkaXYgaWQ9Imh0bWxfM2IyM2M3NzQ5YzdhNDgzZTg5MTEzY2M5YzlhZTVlNTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2ggUGFyaywgVGhlIEp1bmN0aW9uIFNvdXRoLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYzMTFjOTcwNGYwYzQ0MTNhNDQyM2RiMjEwMzRkZjFmLnNldENvbnRlbnQoaHRtbF8zYjIzYzc3NDljN2E0ODNlODkxMTNjYzljOWFlNWU1MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMzlkM2ViODIwNjU0MDQwOWMyNmZhNzljMmZkMDJiMS5iaW5kUG9wdXAocG9wdXBfNjMxMWM5NzA0ZjBjNDQxM2E0NDIzZGIyMTAzNGRmMWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjdiNjk0YTAzYjMyNGVjN2I4MDc5OGE3NTk3YTc4MDEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTYzMTksLTc5LjUzMjI0MjQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I4ZDIzYmQ1MGU2YTQ0OWNiNDA5ODhkNjBiODk5NmQzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Q2NjMwOTZhMjkwYzRiNTM5MzE5MmJlNWI0YTEzYzU4ID0gJCgnPGRpdiBpZD0iaHRtbF9kNjYzMDk2YTI5MGM0YjUzOTMxOTJiZTViNGExM2M1OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG1vdW50LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I4ZDIzYmQ1MGU2YTQ0OWNiNDA5ODhkNjBiODk5NmQzLnNldENvbnRlbnQoaHRtbF9kNjYzMDk2YTI5MGM0YjUzOTMxOTJiZTViNGExM2M1OCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82N2I2OTRhMDNiMzI0ZWM3YjgwNzk4YTc1OTdhNzgwMS5iaW5kUG9wdXAocG9wdXBfYjhkMjNiZDUwZTZhNDQ5Y2I0MDk4OGQ2MGI4OTk2ZDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzg5M2IxNWQ1Njc3NGFhY2E0NzZlMGQ4ZWI1N2I4MGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTAwNzE1MDAwMDAwMDQsLTc5LjI5NTg0OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNjg3MmY4OTZiNzg3NDZmZGJiNDdiMGEzZDA3OGZiNjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDk1ODM5MjExMWYwNDQ1ZWEzZDY0NDQ0ZTE4MjUzZDcgPSAkKCc8ZGl2IGlkPSJodG1sX2Q5NTgzOTIxMTFmMDQ0NWVhM2Q2NDQ0NGUxODI1M2Q3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXhmb3JkLCBNYXJ5dmFsZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY4NzJmODk2Yjc4NzQ2ZmRiYjQ3YjBhM2QwNzhmYjYzLnNldENvbnRlbnQoaHRtbF9kOTU4MzkyMTExZjA0NDVlYTNkNjQ0NDRlMTgyNTNkNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zODkzYjE1ZDU2Nzc0YWFjYTQ3NmUwZDhlYjU3YjgwYS5iaW5kUG9wdXAocG9wdXBfNjg3MmY4OTZiNzg3NDZmZGJiNDdiMGEzZDA3OGZiNjMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmZjZmMzMTBmZmFkNGNlYWEyZGQ0OTdkZTQzMzI5MDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODI3MzY0LC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM3ZjkyMGM4NzU3YjQ5MTFiYTlmODM5NTdiMTU4YWQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAyMDM2OGQ0ZmRmZjRiNDBiYjRkMTgzOTM2Mjg4ZTVjID0gJCgnPGRpdiBpZD0iaHRtbF8wMjAzNjhkNGZkZmY0YjQwYmI0ZDE4MzkzNjI4OGU1YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSwgV2lsbG93ZGFsZSBXZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zN2Y5MjBjODc1N2I0OTExYmE5ZjgzOTU3YjE1OGFkMi5zZXRDb250ZW50KGh0bWxfMDIwMzY4ZDRmZGZmNGI0MGJiNGQxODM5MzYyODhlNWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmZjZmMzMTBmZmFkNGNlYWEyZGQ0OTdkZTQzMzI5MDkuYmluZFBvcHVwKHBvcHVwXzM3ZjkyMGM4NzU3YjQ5MTFiYTlmODM5NTdiMTU4YWQyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMxZjExODJlMzhlMTQwMTM4MGNlMDFjYjA2MTc0YmEzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE1MzgzNCwtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOWRlMDAzOGFkYzk2NDI0MDgwMzIyZTk0ODJiZjE5ZTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTAwODFiMDBhMmRlNGYzNTlhNTYzZGFlZGViZmZjYTIgPSAkKCc8ZGl2IGlkPSJodG1sXzEwMDgxYjAwYTJkZTRmMzU5YTU2M2RhZWRlYmZmY2EyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aCBUb3JvbnRvIFdlc3QsICBMYXdyZW5jZSBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlkZTAwMzhhZGM5NjQyNDA4MDMyMmU5NDgyYmYxOWUyLnNldENvbnRlbnQoaHRtbF8xMDA4MWIwMGEyZGU0ZjM1OWE1NjNkYWVkZWJmZmNhMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMWYxMTgyZTM4ZTE0MDEzODBjZTAxY2IwNjE3NGJhMy5iaW5kUG9wdXAocG9wdXBfOWRlMDAzOGFkYzk2NDI0MDgwMzIyZTk0ODJiZjE5ZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDBmMTk3MjJjZTBhNDIwODg5NWVhYmE4OWY3NTQ1NDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84NDgzZTYwZjc5MjI0YzY0ODA1ZjYxZDU5MjYwMmE3ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NDRhYWNiNzBhZjA0ZDYxYWE5ZjljNzZmNWFhYmJjNyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDQ0YWFjYjcwYWYwNGQ2MWFhOWY5Yzc2ZjVhYWJiYzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBBbm5leCwgTm9ydGggTWlkdG93biwgWW9ya3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg0ODNlNjBmNzkyMjRjNjQ4MDVmNjFkNTkyNjAyYTdmLnNldENvbnRlbnQoaHRtbF80NDRhYWNiNzBhZjA0ZDYxYWE5ZjljNzZmNWFhYmJjNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MGYxOTcyMmNlMGE0MjA4ODk1ZWFiYTg5Zjc1NDU0NC5iaW5kUG9wdXAocG9wdXBfODQ4M2U2MGY3OTIyNGM2NDgwNWY2MWQ1OTI2MDJhN2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2UxODE1MTdjNWJkNDk0NThkNmI3Zjc3OTA0OWRmN2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg5NTk3LC03OS40NTYzMjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWVmZTFmMjJhY2VjNGExNGE5ODc1NTdlODFkYjMzOWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGRiZmM5MTBiM2M2NGU3MjkzMzQ4OGM2MzVhZDMxYTQgPSAkKCc8ZGl2IGlkPSJodG1sXzBkYmZjOTEwYjNjNjRlNzI5MzM0ODhjNjM1YWQzMWE0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSwgUm9uY2VzdmFsbGVzLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFlZmUxZjIyYWNlYzRhMTRhOTg3NTU3ZTgxZGIzMzliLnNldENvbnRlbnQoaHRtbF8wZGJmYzkxMGIzYzY0ZTcyOTMzNDg4YzYzNWFkMzFhNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZTE4MTUxN2M1YmQ0OTQ1OGQ2YjdmNzc5MDQ5ZGY3ZC5iaW5kUG9wdXAocG9wdXBfMWVmZTFmMjJhY2VjNGExNGE5ODc1NTdlODFkYjMzOWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmQ3MDZmN2MyYzhjNDFjZThmZTZlMDU2NzMxZjRmNzggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzY5NjU2LC03OS42MTU4MTg5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNWY1ZmNjZjc0YzI0ZDBlYjM4MzgwYTI1ZDYwYmIxZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMzFhYjllZWVmMWI0NDcyOWQ0MzcyOTdkZTI4ODJlYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDMxYWI5ZWVlZjFiNDQ3MjlkNDM3Mjk3ZGUyODgyZWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhbmFkYSBQb3N0IEdhdGV3YXkgUHJvY2Vzc2luZyBDZW50cmUsIE1pc3Npc3NhdWdhPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xNWY1ZmNjZjc0YzI0ZDBlYjM4MzgwYTI1ZDYwYmIxZC5zZXRDb250ZW50KGh0bWxfMDMxYWI5ZWVlZjFiNDQ3MjlkNDM3Mjk3ZGUyODgyZWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmQ3MDZmN2MyYzhjNDFjZThmZTZlMDU2NzMxZjRmNzguYmluZFBvcHVwKHBvcHVwXzE1ZjVmY2NmNzRjMjRkMGViMzgzODBhMjVkNjBiYjFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NiOTIwOGNkMDBmYjQyODRhMGEyMzk5MDg5YzYyN2VmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDQ4Y2I0ZGM4ZmMyNDAxOWFkNjk1MTNjN2I0NzU5N2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWQxOTE5ZTg3NmJiNGZhMzk0OTM3MmQ3YTgwNTYwOTkgPSAkKCc8ZGl2IGlkPSJodG1sXzlkMTkxOWU4NzZiYjRmYTM5NDkzNzJkN2E4MDU2MDk5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3ZpZXcgVmlsbGFnZSwgU3QuIFBoaWxsaXBzLCBNYXJ0aW4gR3JvdmUgR2FyZGVucywgUmljaHZpZXcgR2FyZGVucywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NDhjYjRkYzhmYzI0MDE5YWQ2OTUxM2M3YjQ3NTk3YS5zZXRDb250ZW50KGh0bWxfOWQxOTE5ZTg3NmJiNGZhMzk0OTM3MmQ3YTgwNTYwOTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2I5MjA4Y2QwMGZiNDI4NGEwYTIzOTkwODljNjI3ZWYuYmluZFBvcHVwKHBvcHVwXzQ0OGNiNGRjOGZjMjQwMTlhZDY5NTEzYzdiNDc1OTdhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzAxNDM1YmM5MGJiMzRlZWRiMzFlMTg2Y2Y4NDNhNDNjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk0MjAwMywtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTllNDQwNWIwZTBkNGVkM2IxY2E4NmYxZTg1YTU3Y2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGYwMjY2NWQ0YmFmNGZmMzk2ZjUyZDY5M2EyYjQxZGYgPSAkKCc8ZGl2IGlkPSJodG1sXzhmMDI2NjVkNGJhZjRmZjM5NmY1MmQ2OTNhMmI0MWRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZ2luY291cnQsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xOWU0NDA1YjBlMGQ0ZWQzYjFjYTg2ZjFlODVhNTdjYi5zZXRDb250ZW50KGh0bWxfOGYwMjY2NWQ0YmFmNGZmMzk2ZjUyZDY5M2EyYjQxZGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDE0MzViYzkwYmIzNGVlZGIzMWUxODZjZjg0M2E0M2MuYmluZFBvcHVwKHBvcHVwXzE5ZTQ0MDViMGUwZDRlZDNiMWNhODZmMWU4NWE1N2NiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EzODQ0YzgzNDI5YjQ3ZmI4YjliNTZlYzk4YjgyOTczID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA0MzI0NCwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yMjQ5ZGUxZTc3OWE0NGU3YjI5MGM3MjVmMzZjMzJiZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZWYyYTM5OTVlMGI0MTBhYmRlNzQyYzY2ODFiMDRkNyA9ICQoJzxkaXYgaWQ9Imh0bWxfZmVmMmEzOTk1ZTBiNDEwYWJkZTc0MmM2NjgxYjA0ZDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRhdmlzdmlsbGUsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjI0OWRlMWU3NzlhNDRlN2IyOTBjNzI1ZjM2YzMyYmYuc2V0Q29udGVudChodG1sX2ZlZjJhMzk5NWUwYjQxMGFiZGU3NDJjNjY4MWIwNGQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EzODQ0YzgzNDI5YjQ3ZmI4YjliNTZlYzk4YjgyOTczLmJpbmRQb3B1cChwb3B1cF8yMjQ5ZGUxZTc3OWE0NGU3YjI5MGM3MjVmMzZjMzJiZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NzUwODI2OTAyZGI0NWQ3YTI1MGQyN2Y0YTNkMGIzMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjhhY2ZmYTc2OTc4NDQ3NDgxYTMzNDMxZDBhNjVhMzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmMzNjU0OThkNDNhNGE2ZmJjZGRmMjQ1N2I1NDdjNzYgPSAkKCc8ZGl2IGlkPSJodG1sXzJjMzY1NDk4ZDQzYTRhNmZiY2RkZjI0NTdiNTQ3Yzc2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI4YWNmZmE3Njk3ODQ0NzQ4MWEzMzQzMWQwYTY1YTMzLnNldENvbnRlbnQoaHRtbF8yYzM2NTQ5OGQ0M2E0YTZmYmNkZGYyNDU3YjU0N2M3Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NzUwODI2OTAyZGI0NWQ3YTI1MGQyN2Y0YTNkMGIzMS5iaW5kUG9wdXAocG9wdXBfMjhhY2ZmYTc2OTc4NDQ3NDgxYTMzNDMxZDBhNjVhMzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNGYxMmFmZTAzYTk1NGIyYWE2NzJiZDQ0MGJjMTk1MTMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q0YTliZWRlNGMzMDRiOTliNzUzOTVkYzBmMzZmM2FkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRmYjA3YWJlNWU1OTQwOGVhNjZkYmZlNzg4OWI0OWI3ID0gJCgnPGRpdiBpZD0iaHRtbF80ZmIwN2FiZTVlNTk0MDhlYTY2ZGJmZTc4ODliNDliNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBTd2Fuc2VhLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q0YTliZWRlNGMzMDRiOTliNzUzOTVkYzBmMzZmM2FkLnNldENvbnRlbnQoaHRtbF80ZmIwN2FiZTVlNTk0MDhlYTY2ZGJmZTc4ODliNDliNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZjEyYWZlMDNhOTU0YjJhYTY3MmJkNDQwYmMxOTUxMy5iaW5kUG9wdXAocG9wdXBfZDRhOWJlZGU0YzMwNGI5OWI3NTM5NWRjMGYzNmYzYWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjYzOWUxZjM1OTJhNDYxZTkyMWNjNjlkMTY5OWJkNGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODE2Mzc1LC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhlMzI2YzM0M2RjOTQ4ZGJhYTk5MzI1OWJlOGE5MzRmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EwNGJmNWFlYTE1ODQyZWI4ZDhkNGNlNmUwNjVjMTQwID0gJCgnPGRpdiBpZD0iaHRtbF9hMDRiZjVhZWExNTg0MmViOGQ4ZDRjZTZlMDY1YzE0MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xhcmtzIENvcm5lcnMsIFRhbSBPJiMzOTtTaGFudGVyLCBTdWxsaXZhbiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhlMzI2YzM0M2RjOTQ4ZGJhYTk5MzI1OWJlOGE5MzRmLnNldENvbnRlbnQoaHRtbF9hMDRiZjVhZWExNTg0MmViOGQ4ZDRjZTZlMDY1YzE0MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mNjM5ZTFmMzU5MmE0NjFlOTIxY2M2OWQxNjk5YmQ0Zi5iaW5kUG9wdXAocG9wdXBfOGUzMjZjMzQzZGM5NDhkYmFhOTkzMjU5YmU4YTkzNGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjI0ZDdhYjQ0Yzc1NDEyYTkxNTI3YzY3M2M5Njk0OGMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNTdmNTNmYTIxODc0MDQ4OWMxNjhiYWEyNzQxNTY4ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NzM2MjMyYmMzNTA0YmQ1YmZiNzA0NmVjZWJiNDcyOSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDczNjIzMmJjMzUwNGJkNWJmYjcwNDZlY2ViYjQ3MjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1vb3JlIFBhcmssIFN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNTdmNTNmYTIxODc0MDQ4OWMxNjhiYWEyNzQxNTY4ZS5zZXRDb250ZW50KGh0bWxfNDczNjIzMmJjMzUwNGJkNWJmYjcwNDZlY2ViYjQ3MjkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjI0ZDdhYjQ0Yzc1NDEyYTkxNTI3YzY3M2M5Njk0OGMuYmluZFBvcHVwKHBvcHVwX2E1N2Y1M2ZhMjE4NzQwNDg5YzE2OGJhYTI3NDE1NjhlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdjOTI2OWU2ODNjYzQzZGM4ZDIwNzFjZTBhZGY4MTQwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MzIyMTY1N2YwMzE0MzdjYjkzMTFlMzg2NzNhYWY2OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MjUzZTdlMDNkYmY0ZDNkYWI5YzJhNDY0NGMyMWQyZCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTI1M2U3ZTAzZGJmNGQzZGFiOWMyYTQ2NDRjMjFkMmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbnNpbmd0b24gTWFya2V0LCBDaGluYXRvd24sIEdyYW5nZSBQYXJrLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MzIyMTY1N2YwMzE0MzdjYjkzMTFlMzg2NzNhYWY2OS5zZXRDb250ZW50KGh0bWxfOTI1M2U3ZTAzZGJmNGQzZGFiOWMyYTQ2NDRjMjFkMmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2M5MjY5ZTY4M2NjNDNkYzhkMjA3MWNlMGFkZjgxNDAuYmluZFBvcHVwKHBvcHVwXzkzMjIxNjU3ZjAzMTQzN2NiOTMxMWUzODY3M2FhZjY5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM2ZTBjNjUwODUyYTQwY2VhZGRjNGNmYTI4ZWM5YzIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODE1MjUyMiwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNjBhMTkyMjdiNWQ0OGI0YjdkOTNkYmVlYTM2ZDA3NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMDc1MTM3OGIyMGM0OGY5YmU3MmJmZGM1NzU0ZGM3YyA9ICQoJzxkaXYgaWQ9Imh0bWxfMTA3NTEzNzhiMjBjNDhmOWJlNzJiZmRjNTc1NGRjN2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1pbGxpa2VuLCBBZ2luY291cnQgTm9ydGgsIFN0ZWVsZXMgRWFzdCwgTCYjMzk7QW1vcmVhdXggRWFzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q2MGExOTIyN2I1ZDQ4YjRiN2Q5M2RiZWVhMzZkMDc3LnNldENvbnRlbnQoaHRtbF8xMDc1MTM3OGIyMGM0OGY5YmU3MmJmZGM1NzU0ZGM3Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNmUwYzY1MDg1MmE0MGNlYWRkYzRjZmEyOGVjOWMyMy5iaW5kUG9wdXAocG9wdXBfZDYwYTE5MjI3YjVkNDhiNGI3ZDkzZGJlZWEzNmQwNzcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzY1YzllOGZmMjQ1NDA4ZjlhZjdmNGRhY2NlZmY2YjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZmQxMjBiZjk4MTQ0MWViYjJmM2Q1M2FhMTc2ODAxZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MjJhNzM0OGJlYjg0MGExOTI3ZmYyMjUzYWRjODU0NCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTIyYTczNDhiZWI4NDBhMTkyN2ZmMjI1M2FkYzg1NDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgV2VzdCwgUmF0aG5lbGx5LCBTb3V0aCBIaWxsLCBGb3Jlc3QgSGlsbCBTRSwgRGVlciBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JmZDEyMGJmOTgxNDQxZWJiMmYzZDUzYWExNzY4MDFlLnNldENvbnRlbnQoaHRtbF85MjJhNzM0OGJlYjg0MGExOTI3ZmYyMjUzYWRjODU0NCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNjVjOWU4ZmYyNDU0MDhmOWFmN2Y0ZGFjY2VmZjZiMi5iaW5kUG9wdXAocG9wdXBfYmZkMTIwYmY5ODE0NDFlYmIyZjNkNTNhYTE3NjgwMWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzY1NTdkN2U4YTk3NDIwNDljNmI2YWFiZjdjMzFjNjEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA0NDAyYTFhNDA2OTQ1NDI4YTk4MzMzNzYzMWYzMTBkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzlhNDliMDRhMGMyMTQyMzBhYjQyMmRiZTNkYTQ1Zjg2ID0gJCgnPGRpdiBpZD0iaHRtbF85YTQ5YjA0YTBjMjE0MjMwYWI0MjJkYmUzZGE0NWY4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q04gVG93ZXIsIEtpbmcgYW5kIFNwYWRpbmEsIFJhaWx3YXkgTGFuZHMsIEhhcmJvdXJmcm9udCBXZXN0LCBCYXRodXJzdCBRdWF5LCBTb3V0aCBOaWFnYXJhLCBJc2xhbmQgYWlycG9ydCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDQ0MDJhMWE0MDY5NDU0MjhhOTgzMzM3NjMxZjMxMGQuc2V0Q29udGVudChodG1sXzlhNDliMDRhMGMyMTQyMzBhYjQyMmRiZTNkYTQ1Zjg2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M2NTU3ZDdlOGE5NzQyMDQ5YzZiNmFhYmY3YzMxYzYxLmJpbmRQb3B1cChwb3B1cF8wNDQwMmExYTQwNjk0NTQyOGE5ODMzMzc2MzFmMzEwZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jY2VjZjBhM2FiNjU0ZGFkODM5YWU1OTYxZTkyZTZhYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwNTY0NjYsLTc5LjUwMTMyMDcwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EwMzI0MWZjZDY0YjQxODk5NDhlZDRkZjMyMDFlOGFjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg2MDA1MmIxZDJmYzQwZGI5YTU2MjgxZmUxMzkwNjViID0gJCgnPGRpdiBpZD0iaHRtbF84NjAwNTJiMWQyZmM0MGRiOWE1NjI4MWZlMTM5MDY1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IFRvcm9udG8sIE1pbWljbyBTb3V0aCwgSHVtYmVyIEJheSBTaG9yZXMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTAzMjQxZmNkNjRiNDE4OTk0OGVkNGRmMzIwMWU4YWMuc2V0Q29udGVudChodG1sXzg2MDA1MmIxZDJmYzQwZGI5YTU2MjgxZmUxMzkwNjViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NjZWNmMGEzYWI2NTRkYWQ4MzlhZTU5NjFlOTJlNmFiLmJpbmRQb3B1cChwb3B1cF9hMDMyNDFmY2Q2NGI0MTg5OTQ4ZWQ0ZGYzMjAxZThhYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYWZiMmUxOTllZTk0YzllYTE0OGMwZjBjMDQ2MDUxZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZmI5NmQ5NzY0NWE0ZDVmYTM5N2ZkMTgwZjczNjQ4NCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZDA4YTM3OTYzYjg0NDc2OTk4N2Q5NzIzMDE1MmEwMSA9ICQoJzxkaXYgaWQ9Imh0bWxfOGQwOGEzNzk2M2I4NDQ3Njk5ODdkOTcyMzAxNTJhMDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNvdXRoIFN0ZWVsZXMsIFNpbHZlcnN0b25lLCBIdW1iZXJnYXRlLCBKYW1lc3Rvd24sIE1vdW50IE9saXZlLCBCZWF1bW9uZCBIZWlnaHRzLCBUaGlzdGxldG93biwgQWxiaW9uIEdhcmRlbnMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2ZiOTZkOTc2NDVhNGQ1ZmEzOTdmZDE4MGY3MzY0ODQuc2V0Q29udGVudChodG1sXzhkMDhhMzc5NjNiODQ0NzY5OTg3ZDk3MjMwMTUyYTAxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FhZmIyZTE5OWVlOTRjOWVhMTQ4YzBmMGMwNDYwNTFmLmJpbmRQb3B1cChwb3B1cF9jZmI5NmQ5NzY0NWE0ZDVmYTM5N2ZkMTgwZjczNjQ4NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ZmZmZTE5MDI3N2M0YjAxYmFjM2FiNDExYTc5MTg4MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc5OTUyNTIwMDAwMDAwNSwtNzkuMzE4Mzg4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83NmViOGY5NTNlMzQ0NDZhOWRkODkxNzgwZGViNDY4MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MjI5NmM3NzFiZTE0YWYzYTkwYTVhMGNjMzg5MmJhMyA9ICQoJzxkaXYgaWQ9Imh0bWxfODIyOTZjNzcxYmUxNGFmM2E5MGE1YTBjYzM4OTJiYTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0ZWVsZXMgV2VzdCwgTCYjMzk7QW1vcmVhdXggV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzc2ZWI4Zjk1M2UzNDQ0NmE5ZGQ4OTE3ODBkZWI0NjgzLnNldENvbnRlbnQoaHRtbF84MjI5NmM3NzFiZTE0YWYzYTkwYTVhMGNjMzg5MmJhMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84ZmZmZTE5MDI3N2M0YjAxYmFjM2FiNDExYTc5MTg4MS5iaW5kUG9wdXAocG9wdXBfNzZlYjhmOTUzZTM0NDQ2YTlkZDg5MTc4MGRlYjQ2ODMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjU1NTVlODk2NDYyNGMyYzhhYzQzNTU4NGM2MWFkNjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNDY5ZTQ5OTY1NjU0YzVlODIxYzg1NjdhMDU4NDY0YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMzM1Y2Q0MjFlNWQ0MzNhOTQ2Y2NhOTFlMGZkMTk2YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZTMzNWNkNDIxZTVkNDMzYTk0NmNjYTkxZTBmZDE5NmIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VkYWxlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNDY5ZTQ5OTY1NjU0YzVlODIxYzg1NjdhMDU4NDY0YS5zZXRDb250ZW50KGh0bWxfZTMzNWNkNDIxZTVkNDMzYTk0NmNjYTkxZTBmZDE5NmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjU1NTVlODk2NDYyNGMyYzhhYzQzNTU4NGM2MWFkNjguYmluZFBvcHVwKHBvcHVwXzI0NjllNDk5NjU2NTRjNWU4MjFjODU2N2EwNTg0NjRhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmZjgyYzdlM2VhYTQ0N2E4NDgwNDY0NTVkYmZlMTlmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzIxZjBhZWVjMTllNDYzYjg4NzJkMmFkNTA3ODEyYWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWMwYmUwYzc5OGI1NDlmNGJkMGQ3NzVlYjVkMjJmMzUgPSAkKCc8ZGl2IGlkPSJodG1sX2VjMGJlMGM3OThiNTQ5ZjRiZDBkNzc1ZWI1ZDIyZjM1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzIxZjBhZWVjMTllNDYzYjg4NzJkMmFkNTA3ODEyYWYuc2V0Q29udGVudChodG1sX2VjMGJlMGM3OThiNTQ5ZjRiZDBkNzc1ZWI1ZDIyZjM1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmZjgyYzdlM2VhYTQ0N2E4NDgwNDY0NTVkYmZlMTlmLmJpbmRQb3B1cChwb3B1cF83MjFmMGFlZWMxOWU0NjNiODg3MmQyYWQ1MDc4MTJhZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yODU0M2Y2Y2U3NTU0YzljYmRkMjg5ZTM1YTc0YTVlYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYwMjQxMzcwMDAwMDAxLC03OS41NDM0ODQwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMjI2MDg2OTE5ZmI0NjBlYWQ2N2Y0MDZjZmRjM2RjMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZDc0ZThlMzNhMDc0NzQ0OGI1ZjY0Y2RlMTRkMWJlMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNmQ3NGU4ZTMzYTA3NDc0NDhiNWY2NGNkZTE0ZDFiZTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsZGVyd29vZCwgTG9uZyBCcmFuY2gsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzIyNjA4NjkxOWZiNDYwZWFkNjdmNDA2Y2ZkYzNkYzIuc2V0Q29udGVudChodG1sXzZkNzRlOGUzM2EwNzQ3NDQ4YjVmNjRjZGUxNGQxYmUzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzI4NTQzZjZjZTc1NTRjOWNiZGQyODllMzVhNzRhNWViLmJpbmRQb3B1cChwb3B1cF8zMjI2MDg2OTE5ZmI0NjBlYWQ2N2Y0MDZjZmRjM2RjMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMGMyMDVhZjdiZTc0MjIxOWUyOTlhODRiZTZiNzM4YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNjc0ODI5OTk5OTk5NCwtNzkuNTk0MDU0NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yNDg3NTYzN2NhYTI0NTdlYmIxMDcwOGU0YzdjZGE0MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zYzUxN2ZkOTQzZGI0NTY0OTJlZDIyYmViNjBlODNkOCA9ICQoJzxkaXYgaWQ9Imh0bWxfM2M1MTdmZDk0M2RiNDU2NDkyZWQyMmJlYjYwZTgzZDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod2VzdCwgV2VzdCBIdW1iZXIgLSBDbGFpcnZpbGxlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI0ODc1NjM3Y2FhMjQ1N2ViYjEwNzA4ZTRjN2NkYTQxLnNldENvbnRlbnQoaHRtbF8zYzUxN2ZkOTQzZGI0NTY0OTJlZDIyYmViNjBlODNkOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMGMyMDVhZjdiZTc0MjIxOWUyOTlhODRiZTZiNzM4Yy5iaW5kUG9wdXAocG9wdXBfMjQ4NzU2MzdjYWEyNDU3ZWJiMTA3MDhlNGM3Y2RhNDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmNlMDlkMTljNWI1NDFlMmJmM2NlNmZjMTM2ZDE5NjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MzYxMjQ3MDAwMDAwMDYsLTc5LjIwNTYzNjA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM3MWJiNDQzMzdjZTRmNjFhMzQzMDUwMzJjYjcxZmNmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkzMTRlZGU0ZDNmNDQ0NDM4MjE2MzMzMWQ0MWE4MDNkID0gJCgnPGRpdiBpZD0iaHRtbF85MzE0ZWRlNGQzZjQ0NDQzODIxNjMzMzFkNDFhODAzZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VXBwZXIgUm91Z2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zNzFiYjQ0MzM3Y2U0ZjYxYTM0MzA1MDMyY2I3MWZjZi5zZXRDb250ZW50KGh0bWxfOTMxNGVkZTRkM2Y0NDQ0MzgyMTYzMzMxZDQxYTgwM2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmNlMDlkMTljNWI1NDFlMmJmM2NlNmZjMTM2ZDE5NjUuYmluZFBvcHVwKHBvcHVwXzM3MWJiNDQzMzdjZTRmNjFhMzQzMDUwMzJjYjcxZmNmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RkOWQxMDdhZWM1MTQ0MjhiOTllZmMxYzUyNmM4YzY1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY3OTY3LC03OS4zNjc2NzUzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EyZDA4NjVlOWM2NDQxZGFhMWY5OTYyOWM3ZWZmNDdkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QxM2YxMDE5YmVkZTQ2YjA5MTIxNDI1NWI3ZDNlZWJkID0gJCgnPGRpdiBpZD0iaHRtbF9kMTNmMTAxOWJlZGU0NmIwOTEyMTQyNTViN2QzZWViZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIENhYmJhZ2V0b3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMmQwODY1ZTljNjQ0MWRhYTFmOTk2MjljN2VmZjQ3ZC5zZXRDb250ZW50KGh0bWxfZDEzZjEwMTliZWRlNDZiMDkxMjE0MjU1YjdkM2VlYmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGQ5ZDEwN2FlYzUxNDQyOGI5OWVmYzFjNTI2YzhjNjUuYmluZFBvcHVwKHBvcHVwX2EyZDA4NjVlOWM2NDQxZGFhMWY5OTYyOWM3ZWZmNDdkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU0ZmZkNGQzZmFhODRhZTk5YzQzMTk3NDMyZDA2Y2RlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNWUxZDJjMDE3YTk0OTRmODliMDk4MzFkYmI3Nzc0NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZTkwN2ZmZGI4ZmY0Yjc3OTJlZGUyMDdkMTNmZGFhZCA9ICQoJzxkaXYgaWQ9Imh0bWxfOWU5MDdmZmRiOGZmNGI3NzkyZWRlMjA3ZDEzZmRhYWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kNWUxZDJjMDE3YTk0OTRmODliMDk4MzFkYmI3Nzc0Ni5zZXRDb250ZW50KGh0bWxfOWU5MDdmZmRiOGZmNGI3NzkyZWRlMjA3ZDEzZmRhYWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTRmZmQ0ZDNmYWE4NGFlOTljNDMxOTc0MzJkMDZjZGUuYmluZFBvcHVwKHBvcHVwX2Q1ZTFkMmMwMTdhOTQ5NGY4OWIwOTgzMWRiYjc3NzQ2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFhZDExYzUzMmIxMTRlNDJhZTFlMzAzYTc0MjM4OTVlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzNjUzNjAwMDAwMDA1LC03OS41MDY5NDM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg5ZTkxNzVkOGRjOTQ4NDRhODY3YzRkNjYwM2IzOWFmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk3ZWMzYjA1YTEzZjRhNDc4Yzc5MWM3NmYwZGIzMWQzID0gJCgnPGRpdiBpZD0iaHRtbF85N2VjM2IwNWExM2Y0YTQ3OGM3OTFjNzZmMGRiMzFkMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEtpbmdzd2F5LCBNb250Z29tZXJ5IFJvYWQsIE9sZCBNaWxsIE5vcnRoLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg5ZTkxNzVkOGRjOTQ4NDRhODY3YzRkNjYwM2IzOWFmLnNldENvbnRlbnQoaHRtbF85N2VjM2IwNWExM2Y0YTQ3OGM3OTFjNzZmMGRiMzFkMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYWQxMWM1MzJiMTE0ZTQyYWUxZTMwM2E3NDIzODk1ZS5iaW5kUG9wdXAocG9wdXBfODllOTE3NWQ4ZGM5NDg0NGE4NjdjNGQ2NjAzYjM5YWYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTJkMDYxZGZjZGY3NDMxZTkyMTQ4YmFlZDg5MmFmOTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NTkxZGIxYTEyNmU0OWQ2Yjg1Mzk1NTZhNTQ3YjZkMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNjliZmQ0MTk3NjM0ZDAwOGZhZDNjOGRkZGMxNjRjZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMzY5YmZkNDE5NzYzNGQwMDhmYWQzYzhkZGRjMTY0Y2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NTkxZGIxYTEyNmU0OWQ2Yjg1Mzk1NTZhNTQ3YjZkMi5zZXRDb250ZW50KGh0bWxfMzY5YmZkNDE5NzYzNGQwMDhmYWQzYzhkZGRjMTY0Y2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTJkMDYxZGZjZGY3NDMxZTkyMTQ4YmFlZDg5MmFmOTkuYmluZFBvcHVwKHBvcHVwXzU1OTFkYjFhMTI2ZTQ5ZDZiODUzOTU1NmE1NDdiNmQyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MzMzc3NmI4MTdjMjQ3OGFiMGNkMTQ1NjQ5ZTg0OTkyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNzQzOSwtNzkuMzIxNTU4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzdlYTljYTg2NTM0ZTQyODI5MWZiOWQzMGEwOTBiZGJiKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q2NWI5NDY3MmQzOTQwYTE4ZGM4N2U0ZTdmZThhOGZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFiYjcxYWJiZTU3ZTQxNjA5NDQ2NjFiNTEwMzY2NGVhID0gJCgnPGRpdiBpZD0iaHRtbF8xYmI3MWFiYmU1N2U0MTYwOTQ0NjYxYjUxMDM2NjRlYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QnVzaW5lc3MgcmVwbHkgbWFpbCBQcm9jZXNzaW5nIENlbnRyZSwgU291dGggQ2VudHJhbCBMZXR0ZXIgUHJvY2Vzc2luZyBQbGFudCBUb3JvbnRvLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q2NWI5NDY3MmQzOTQwYTE4ZGM4N2U0ZTdmZThhOGZmLnNldENvbnRlbnQoaHRtbF8xYmI3MWFiYmU1N2U0MTYwOTQ0NjYxYjUxMDM2NjRlYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMzM3NzZiODE3YzI0NzhhYjBjZDE0NTY0OWU4NDk5Mi5iaW5kUG9wdXAocG9wdXBfZDY1Yjk0NjcyZDM5NDBhMThkYzg3ZTRlN2ZlOGE4ZmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2MwNmU2MzYxNDA1NGNkNWJhMTcxMDBiMjEyMzg0OTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzYyNTc5LC03OS40OTg1MDkwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF83ZWE5Y2E4NjUzNGU0MjgyOTFmYjlkMzBhMDkwYmRiYik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNDU2YWVmNjUzM2Y0Y2U0YjY2YjdiNGEzNWUyZmYwMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YjFkZTQ3N2E1ODA0Y2RkYTc0YzFmNWI2M2Q1M2JlNCA9ICQoJzxkaXYgaWQ9Imh0bWxfOGIxZGU0NzdhNTgwNGNkZGE3NGMxZjViNjNkNTNiZTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9sZCBNaWxsIFNvdXRoLCBLaW5nJiMzOTtzIE1pbGwgUGFyaywgU3VubnlsZWEsIEh1bWJlciBCYXksIE1pbWljbyBORSwgVGhlIFF1ZWVuc3dheSBFYXN0LCBSb3lhbCBZb3JrIFNvdXRoIEVhc3QsIEtpbmdzd2F5IFBhcmsgU291dGggRWFzdCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jNDU2YWVmNjUzM2Y0Y2U0YjY2YjdiNGEzNWUyZmYwMC5zZXRDb250ZW50KGh0bWxfOGIxZGU0NzdhNTgwNGNkZGE3NGMxZjViNjNkNTNiZTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2MwNmU2MzYxNDA1NGNkNWJhMTcxMDBiMjEyMzg0OTkuYmluZFBvcHVwKHBvcHVwX2M0NTZhZWY2NTMzZjRjZTRiNjZiN2I0YTM1ZTJmZjAwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNkMGQ5NWQxN2YzNjQ4NjdiZjNlZmRmN2YyMGVlN2M3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4ODQwOCwtNzkuNTIwOTk5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfN2VhOWNhODY1MzRlNDI4MjkxZmI5ZDMwYTA5MGJkYmIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODA4MWM2MDgzYjQ3NDk4N2I1NTFjMzljYTY5ZGViMDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODNkMWI4NTVjZjE0NDhhOTkwNDBjNmIzOTViOTRjZTkgPSAkKCc8ZGl2IGlkPSJodG1sXzgzZDFiODU1Y2YxNDQ4YTk5MDQwYzZiMzk1Yjk0Y2U5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaW1pY28gTlcsIFRoZSBRdWVlbnN3YXkgV2VzdCwgU291dGggb2YgQmxvb3IsIEtpbmdzd2F5IFBhcmsgU291dGggV2VzdCwgUm95YWwgWW9yayBTb3V0aCBXZXN0LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgwODFjNjA4M2I0NzQ5ODdiNTUxYzM5Y2E2OWRlYjA3LnNldENvbnRlbnQoaHRtbF84M2QxYjg1NWNmMTQ0OGE5OTA0MGM2YjM5NWI5NGNlOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zZDBkOTVkMTdmMzY0ODY3YmYzZWZkZjdmMjBlZTdjNy5iaW5kUG9wdXAocG9wdXBfODA4MWM2MDgzYjQ3NDk4N2I1NTFjMzljYTY5ZGViMDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Map Downtown


```python
ds_downtown = ds[ds['Borough'] == 'Downtown Toronto'].reset_index(drop = True)
print(ds_downtown.shape)
ds_downtown.head(10)
```

    (19, 5)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Richmond, Adelaide, King</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
  </tbody>
</table>
</div>




```python
map_downtown = folium.Map(location=[latitude_downtown, longitude_downtown], zoom_start= 11)

for lat, lng, borough, neighborhood in zip(ds_downtown['Latitude'], ds_downtown['Longitude'], 
                                           ds_downtown['Borough'], ds_downtown['Neighbourhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='red',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_downtown)  
    
map_downtown
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfODcyNTUyOWI4NDMzNDljOGJiZWNmN2Q0MTliZTljZjIgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMiA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMicsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjU2MzIyMSwtNzkuMzgwOTE2MV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfZjE3NzBlNGFjZmQxNGZjNWEwYzk4ODM1NTU0YTYxMWMgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODcyNTUyOWI4NDMzNDljOGJiZWNmN2Q0MTliZTljZjIpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ5NzMzYmZjYjE4OTQyZjY5YjI3ZDBlZjM1OGZmNzk1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U0M2QwMGYyNzllODRkZDZhOWU2MWZhOTZjOTRkMTIxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I1ZjYxODE3ZWYyNzQ5NDQ5YTY0MTY2MTUyZWFkZTY5ID0gJCgnPGRpdiBpZD0iaHRtbF9iNWY2MTgxN2VmMjc0OTQ0OWE2NDE2NjE1MmVhZGU2OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmVnZW50IFBhcmssIEhhcmJvdXJmcm9udCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTQzZDAwZjI3OWU4NGRkNmE5ZTYxZmE5NmM5NGQxMjEuc2V0Q29udGVudChodG1sX2I1ZjYxODE3ZWYyNzQ5NDQ5YTY0MTY2MTUyZWFkZTY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ5NzMzYmZjYjE4OTQyZjY5YjI3ZDBlZjM1OGZmNzk1LmJpbmRQb3B1cChwb3B1cF9lNDNkMDBmMjc5ZTg0ZGQ2YTllNjFmYTk2Yzk0ZDEyMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iYzM4N2U4NDA4MTE0ZDQxYmZmMTExOTk4MTY2NjdjZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjMwMTUsLTc5LjM4OTQ5MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMjAxM2NiNDliNzM0MjlhOTJjYzk4OGIyZTMzZjE1YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NmE0MThjZGVkNmY0Yjc0OWE2YTcxNWU2MmRhNmQ3OSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDZhNDE4Y2RlZDZmNGI3NDlhNmE3MTVlNjJkYTZkNzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIE9udGFyaW8gUHJvdmluY2lhbCBHb3Zlcm5tZW50LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMjAxM2NiNDliNzM0MjlhOTJjYzk4OGIyZTMzZjE1YS5zZXRDb250ZW50KGh0bWxfNDZhNDE4Y2RlZDZmNGI3NDlhNmE3MTVlNjJkYTZkNzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmMzODdlODQwODExNGQ0MWJmZjExMTk5ODE2NjY3Y2UuYmluZFBvcHVwKHBvcHVwX2YyMDEzY2I0OWI3MzQyOWE5MmNjOTg4YjJlMzNmMTVhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZjMGVkOGJjZTQ1ZTQwZmU5Y2MwZDI1NjBiN2RmMGQ0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMjg1YTE3Y2MwOTI0YjRkYjhiYzVjOTg4MDkwYTEzNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZWUxYmRhM2VlYTA0ZjMwYjExZjdlNDRiMDg5N2I0MSA9ICQoJzxkaXYgaWQ9Imh0bWxfOWVlMWJkYTNlZWEwNGYzMGIxMWY3ZTQ0YjA4OTdiNDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCwgUnllcnNvbiwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTI4NWExN2NjMDkyNGI0ZGI4YmM1Yzk4ODA5MGExMzcuc2V0Q29udGVudChodG1sXzllZTFiZGEzZWVhMDRmMzBiMTFmN2U0NGIwODk3YjQxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZjMGVkOGJjZTQ1ZTQwZmU5Y2MwZDI1NjBiN2RmMGQ0LmJpbmRQb3B1cChwb3B1cF8xMjg1YTE3Y2MwOTI0YjRkYjhiYzVjOTg4MDkwYTEzNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNWZkMDgyM2ZmNDQ0NTljODNhNTA0MGJhY2Y0MWE2YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YzdmYmNmZWQ0NmE0NDhkYWFjM2Q4NDkzYmUzOTJiNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NWQ3MjJhMTU0YjM0MjBhOTZhYmU0ZmMyOWM4NWFkYiA9ICQoJzxkaXYgaWQ9Imh0bWxfNzVkNzIyYTE1NGIzNDIwYTk2YWJlNGZjMjljODVhZGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YzdmYmNmZWQ0NmE0NDhkYWFjM2Q4NDkzYmUzOTJiNi5zZXRDb250ZW50KGh0bWxfNzVkNzIyYTE1NGIzNDIwYTk2YWJlNGZjMjljODVhZGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzVmZDA4MjNmZjQ0NDU5YzgzYTUwNDBiYWNmNDFhNmIuYmluZFBvcHVwKHBvcHVwXzZjN2ZiY2ZlZDQ2YTQ0OGRhYWMzZDg0OTNiZTM5MmI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzYzA3ODUwZmI0YjQ0ZWJhOTliY2M3MDhhNjIwMWZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODcyNTUyOWI4NDMzNDljOGJiZWNmN2Q0MTliZTljZjIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTNkZTQwOGVlMDRiNDZhNWE1NmMxZGQwNGZkMWU0OWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODA0NjFiYzA1NjJjNGJhYzgyM2Q2MGRmZTBmZDA3NTQgPSAkKCc8ZGl2IGlkPSJodG1sXzgwNDYxYmMwNTYyYzRiYWM4MjNkNjBkZmUwZmQwNzU0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTNkZTQwOGVlMDRiNDZhNWE1NmMxZGQwNGZkMWU0OWUuc2V0Q29udGVudChodG1sXzgwNDYxYmMwNTYyYzRiYWM4MjNkNjBkZmUwZmQwNzU0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzYzA3ODUwZmI0YjQ0ZWJhOTliY2M3MDhhNjIwMWZiLmJpbmRQb3B1cChwb3B1cF9lM2RlNDA4ZWUwNGI0NmE1YTU2YzFkZDA0ZmQxZTQ5ZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZTUxMzA1MzJhODI0MzVhYWVhMThlNTVjOGFiNjRiMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1Nzk1MjQsLTc5LjM4NzM4MjZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYjYyMmUyNDA2ZDI0ZTdmYWFiZGEwNTBjMWI5NmVhYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMGIyY2U0YTM4YTg0NGQxOTcxNTAzNGY3YjVlNWNiZiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjBiMmNlNGEzOGE4NDRkMTk3MTUwMzRmN2I1ZTVjYmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2I2MjJlMjQwNmQyNGU3ZmFhYmRhMDUwYzFiOTZlYWEuc2V0Q29udGVudChodG1sX2IwYjJjZTRhMzhhODQ0ZDE5NzE1MDM0ZjdiNWU1Y2JmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJlNTEzMDUzMmE4MjQzNWFhZWExOGU1NWM4YWI2NGIwLmJpbmRQb3B1cChwb3B1cF9jYjYyMmUyNDA2ZDI0ZTdmYWFiZGEwNTBjMWI5NmVhYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NWQ3NTk2N2U1ZjU0ZjZjYTdjNDFmNmI2OTBiM2EyOCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVjYzdhODQ1ZDQ3NjRjNmQ5MGEyZjI0MTQ0Yjk5M2ExID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJhZDExNDZkMjhmYzQ3ZjVhYTMyMmY5MDYzMTA3NjBkID0gJCgnPGRpdiBpZD0iaHRtbF8yYWQxMTQ2ZDI4ZmM0N2Y1YWEzMjJmOTA2MzEwNzYwZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hyaXN0aWUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVjYzdhODQ1ZDQ3NjRjNmQ5MGEyZjI0MTQ0Yjk5M2ExLnNldENvbnRlbnQoaHRtbF8yYWQxMTQ2ZDI4ZmM0N2Y1YWEzMjJmOTA2MzEwNzYwZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NWQ3NTk2N2U1ZjU0ZjZjYTdjNDFmNmI2OTBiM2EyOC5iaW5kUG9wdXAocG9wdXBfNWNjN2E4NDVkNDc2NGM2ZDkwYTJmMjQxNDRiOTkzYTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjI3Nzc5YzE3N2M4NGYzNGE3MmJjODhlMmRiNTZmMGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdkZWJjYWE3MDBmZTQ4MWFiZDFlZTRmNjZhYWY2MmQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M5NzYzNDA5YmE3NzQwMzNhYThmMmZkMDIzYWI4ZmYxID0gJCgnPGRpdiBpZD0iaHRtbF9jOTc2MzQwOWJhNzc0MDMzYWE4ZjJmZDAyM2FiOGZmMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaG1vbmQsIEFkZWxhaWRlLCBLaW5nLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZGViY2FhNzAwZmU0ODFhYmQxZWU0ZjY2YWFmNjJkNy5zZXRDb250ZW50KGh0bWxfYzk3NjM0MDliYTc3NDAzM2FhOGYyZmQwMjNhYjhmZjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjI3Nzc5YzE3N2M4NGYzNGE3MmJjODhlMmRiNTZmMGYuYmluZFBvcHVwKHBvcHVwXzdkZWJjYWE3MDBmZTQ4MWFiZDFlZTRmNjZhYWY2MmQ3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2VjYWI2OGVkOGNkYTRhYTRiYjUzOGMyMDY1ZTZhMDBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MTdkZGIyODllMTc0OTUzOTYyYmRjOTQ4ZDUwYTZjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMzY1OThlMjY3MDM0ZTVkYmJlZWIzZjg1NzNkM2I1NyA9ICQoJzxkaXYgaWQ9Imh0bWxfZDM2NTk4ZTI2NzAzNGU1ZGJiZWViM2Y4NTczZDNiNTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBFYXN0LCBVbmlvbiBTdGF0aW9uLCBUb3JvbnRvIElzbGFuZHMsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkxN2RkYjI4OWUxNzQ5NTM5NjJiZGM5NDhkNTBhNmNkLnNldENvbnRlbnQoaHRtbF9kMzY1OThlMjY3MDM0ZTVkYmJlZWIzZjg1NzNkM2I1Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lY2FiNjhlZDhjZGE0YWE0YmI1MzhjMjA2NWU2YTAwZi5iaW5kUG9wdXAocG9wdXBfOTE3ZGRiMjg5ZTE3NDk1Mzk2MmJkYzk0OGQ1MGE2Y2QpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzVlNjExMTE4MzJiNDk1Y2I2NjY1Y2NkZjUzMmEwNDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y3OGQ3MTdmOTE1NDQ1ZDY5YmRjMjA0MGQzZDM5NGQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUyNWZjZDMwZWMzZDQyMzE5MWJlYjI3NTcyMjQ4NTlhID0gJCgnPGRpdiBpZD0iaHRtbF81MjVmY2QzMGVjM2Q0MjMxOTFiZWIyNzU3MjI0ODU5YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmUsIERlc2lnbiBFeGNoYW5nZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjc4ZDcxN2Y5MTU0NDVkNjliZGMyMDQwZDNkMzk0ZDcuc2V0Q29udGVudChodG1sXzUyNWZjZDMwZWMzZDQyMzE5MWJlYjI3NTcyMjQ4NTlhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M1ZTYxMTExODMyYjQ5NWNiNjY2NWNjZGY1MzJhMDQwLmJpbmRQb3B1cChwb3B1cF9mNzhkNzE3ZjkxNTQ0NWQ2OWJkYzIwNDBkM2QzOTRkNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNzdmMWEzYzM2MjA0NzU0YTNiYjJlMDBkZjA5NTE3MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODE5ODUsLTc5LjM3OTgxNjkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODcyNTUyOWI4NDMzNDljOGJiZWNmN2Q0MTliZTljZjIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzAxODMxZDc0ZjBkNDIwN2FhNzBkMzQzYjdmMWM0MWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzUzZDU3NzlkMWE1NDE3Yjg3ODcyMWQ1ODQyY2RkMzUgPSAkKCc8ZGl2IGlkPSJodG1sXzM1M2Q1Nzc5ZDFhNTQxN2I4Nzg3MjFkNTg0MmNkZDM1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21tZXJjZSBDb3VydCwgVmljdG9yaWEgSG90ZWwsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwMTgzMWQ3NGYwZDQyMDdhYTcwZDM0M2I3ZjFjNDFlLnNldENvbnRlbnQoaHRtbF8zNTNkNTc3OWQxYTU0MTdiODc4NzIxZDU4NDJjZGQzNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iNzdmMWEzYzM2MjA0NzU0YTNiYjJlMDBkZjA5NTE3MC5iaW5kUG9wdXAocG9wdXBfYzAxODMxZDc0ZjBkNDIwN2FhNzBkMzQzYjdmMWM0MWUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjBjNzMwOGUwMmFmNGFkZGFjMmY5OTRjZWJmZjg4NDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODcyNTUyOWI4NDMzNDljOGJiZWNmN2Q0MTliZTljZjIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmU1NzMwYzlhYWRhNDU4ZmI5NWZlODg5ZDZlMDYwZTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTFlZGJhZjhlOGMzNDNiOGIzMTk2YTFiMWZiMmNiNjAgPSAkKCc8ZGl2IGlkPSJodG1sX2UxZWRiYWY4ZThjMzQzYjhiMzE5NmExYjFmYjJjYjYwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8sIEhhcmJvcmQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJlNTczMGM5YWFkYTQ1OGZiOTVmZTg4OWQ2ZTA2MGU1LnNldENvbnRlbnQoaHRtbF9lMWVkYmFmOGU4YzM0M2I4YjMxOTZhMWIxZmIyY2I2MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMGM3MzA4ZTAyYWY0YWRkYWMyZjk5NGNlYmZmODg0Ny5iaW5kUG9wdXAocG9wdXBfMmU1NzMwYzlhYWRhNDU4ZmI5NWZlODg5ZDZlMDYwZTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzkzZGYzZmIzODY0NDU1ZTk3MWMxOTYyYjA3ZGEwMjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInJlZCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfODcyNTUyOWI4NDMzNDljOGJiZWNmN2Q0MTliZTljZjIpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmZhNjEzMmZiOWVlNDkxY2E2M2Q1OWFhZmEzOTBjMzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzFjZGVhMzAzNGI4NGYyM2JjN2ZmMzI2N2JkZmE4YTIgPSAkKCc8ZGl2IGlkPSJodG1sX2MxY2RlYTMwMzRiODRmMjNiYzdmZjMyNjdiZGZhOGEyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5zaW5ndG9uIE1hcmtldCwgQ2hpbmF0b3duLCBHcmFuZ2UgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmZhNjEzMmZiOWVlNDkxY2E2M2Q1OWFhZmEzOTBjMzMuc2V0Q29udGVudChodG1sX2MxY2RlYTMwMzRiODRmMjNiYzdmZjMyNjdiZGZhOGEyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc5M2RmM2ZiMzg2NDQ1NWU5NzFjMTk2MmIwN2RhMDIyLmJpbmRQb3B1cChwb3B1cF9iZmE2MTMyZmI5ZWU0OTFjYTYzZDU5YWFmYTM5MGMzMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTU3ODJkMDU2Nzg0YzcwOGI1NTA0MGZmODNmMzEzNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMGQwYjNjYTJmOTA0NDkxYTIzMDRjNDc4MjgxMTdkZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82MjA2MDA3YmY1ZGQ0MGVjOTNjZjBiYjhkMzI2OGNiZCA9ICQoJzxkaXYgaWQ9Imh0bWxfNjIwNjAwN2JmNWRkNDBlYzkzY2YwYmI4ZDMyNjhjYmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EwZDBiM2NhMmY5MDQ0OTFhMjMwNGM0NzgyODExN2RmLnNldENvbnRlbnQoaHRtbF82MjA2MDA3YmY1ZGQ0MGVjOTNjZjBiYjhkMzI2OGNiZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NTU3ODJkMDU2Nzg0YzcwOGI1NTA0MGZmODNmMzEzNC5iaW5kUG9wdXAocG9wdXBfYTBkMGIzY2EyZjkwNDQ5MWEyMzA0YzQ3ODI4MTE3ZGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTRkYjk2YjY1NzFmNDdjNTkzNmM4ZThjMTZmOGEwNjQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NjI2LC03OS4zNzc1Mjk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYyN2M5MGFmZDQzZTQwMjQ4YTQxYTQ4YmQ0NzYxMDBjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2U0NWU5OWM1YjNhMzRhMDA5ZTlmZThmMmU1ZWVlMmRhID0gJCgnPGRpdiBpZD0iaHRtbF9lNDVlOTljNWIzYTM0YTAwOWU5ZmU4ZjJlNWVlZTJkYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYyN2M5MGFmZDQzZTQwMjQ4YTQxYTQ4YmQ0NzYxMDBjLnNldENvbnRlbnQoaHRtbF9lNDVlOTljNWIzYTM0YTAwOWU5ZmU4ZjJlNWVlZTJkYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NGRiOTZiNjU3MWY0N2M1OTM2YzhlOGMxNmY4YTA2NC5iaW5kUG9wdXAocG9wdXBfNjI3YzkwYWZkNDNlNDAyNDhhNDFhNDhiZDQ3NjEwMGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTlmNTZiZTAxZGIzNDU0OGFiMDIyZDUwYjhiOGZmNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJyZWQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzg3MjU1MjliODQzMzQ5YzhiYmVjZjdkNDE5YmU5Y2YyKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNjYjU4OWU2ZDhiNTQyYTRiMjgxMjVkNDg1NzZkNjMzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2MxYjU5YTQ4ZjQzMDRjNTk5MThhNzY5M2QxZDhjZWE5ID0gJCgnPGRpdiBpZD0iaHRtbF9jMWI1OWE0OGY0MzA0YzU5OTE4YTc2OTNkMWQ4Y2VhOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3RuIEEgUE8gQm94ZXMsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNjYjU4OWU2ZDhiNTQyYTRiMjgxMjVkNDg1NzZkNjMzLnNldENvbnRlbnQoaHRtbF9jMWI1OWE0OGY0MzA0YzU5OTE4YTc2OTNkMWQ4Y2VhOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85OWY1NmJlMDFkYjM0NTQ4YWIwMjJkNTBiOGI4ZmY1NS5iaW5kUG9wdXAocG9wdXBfM2NiNTg5ZTZkOGI1NDJhNGIyODEyNWQ0ODU3NmQ2MzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmY4YTI4ZGYxOWNkNGQ3YmJhODgwOWJkNGM5YTc1NjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kNDZmOTllYTdlYzU0YjExODVjNGQxN2NkOGVmNDZlMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80YWVjZGYxYTUxMTY0YjNiOTMxMTIzZTEzNWZjMGI2NSA9ICQoJzxkaXYgaWQ9Imh0bWxfNGFlY2RmMWE1MTE2NGIzYjkzMTEyM2UxMzVmYzBiNjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBDYWJiYWdldG93biwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDQ2Zjk5ZWE3ZWM1NGIxMTg1YzRkMTdjZDhlZjQ2ZTEuc2V0Q29udGVudChodG1sXzRhZWNkZjFhNTExNjRiM2I5MzExMjNlMTM1ZmMwYjY1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZmOGEyOGRmMTljZDRkN2JiYTg4MDliZDRjOWE3NTY4LmJpbmRQb3B1cChwb3B1cF9kNDZmOTllYTdlYzU0YjExODVjNGQxN2NkOGVmNDZlMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMzk1ZGNiZDY0OTg0YTQ5OWYzNWU5ZTNkYTNmNGE5ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODQyOTIsLTc5LjM4MjI4MDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NTlkZWU5ZjExY2Q0MWU2YjQ3YzQ3MDQwYzBiZTJkZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZGRjYTcxNzkyMzQ0MzExYjk4N2ZhNjhjNGI4MjgxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfMGRkY2E3MTc5MjM0NDMxMWI5ODdmYTY4YzRiODI4MWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlLCBVbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NTlkZWU5ZjExY2Q0MWU2YjQ3YzQ3MDQwYzBiZTJkZi5zZXRDb250ZW50KGh0bWxfMGRkY2E3MTc5MjM0NDMxMWI5ODdmYTY4YzRiODI4MWMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjM5NWRjYmQ2NDk4NGE0OTlmMzVlOWUzZGEzZjRhOWQuYmluZFBvcHVwKHBvcHVwXzU1OWRlZTlmMTFjZDQxZTZiNDdjNDcwNDBjMGJlMmRmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2RlYTYzMTQ0ZDdiOTQzY2M5YTJhOTkxMWUzYmNjYWY5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY1ODU5OSwtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAicmVkIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF84NzI1NTI5Yjg0MzM0OWM4YmJlY2Y3ZDQxOWJlOWNmMik7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84YjQ3NTI1OWNiZTU0MjA5OThmNjliYTUwYzM1NGIyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yOTcxYzljZjExYzc0YmJjYWRiNDZiNGRiODllZGU4YiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjk3MWM5Y2YxMWM3NGJiY2FkYjQ2YjRkYjg5ZWRlOGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84YjQ3NTI1OWNiZTU0MjA5OThmNjliYTUwYzM1NGIyYy5zZXRDb250ZW50KGh0bWxfMjk3MWM5Y2YxMWM3NGJiY2FkYjQ2YjRkYjg5ZWRlOGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZGVhNjMxNDRkN2I5NDNjYzlhMmE5OTExZTNiY2NhZjkuYmluZFBvcHVwKHBvcHVwXzhiNDc1MjU5Y2JlNTQyMDk5OGY2OWJhNTBjMzU0YjJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4= onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



## Step 8 - Passing Credentials Foursquare


```python
CLIENT_ID = 'FW4LQTK3S5SUAAOCARVEFH4SC3PKQKCZRBPXT02DBRIPVMOT' # your Foursquare ID
CLIENT_SECRET = 'MTU22W5KUID5MW1UK3QDVLOBOCBJ4QCJBFDLUPINJFK3ZSQA' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: FW4LQTK3S5SUAAOCARVEFH4SC3PKQKCZRBPXT02DBRIPVMOT
    CLIENT_SECRET:MTU22W5KUID5MW1UK3QDVLOBOCBJ4QCJBFDLUPINJFK3ZSQA


## Step 9 - Exploring Details of Donwtown Toronto   


```python
RADIUS = 500 # define radius
LIMIT = 100 # limit of number of venues returned by Foursquare API

url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    latitude_downtown, 
    longitude_downtown, 
    RADIUS, 
    LIMIT)
url
```




    'https://api.foursquare.com/v2/venues/explore?&client_id=FW4LQTK3S5SUAAOCARVEFH4SC3PKQKCZRBPXT02DBRIPVMOT&client_secret=MTU22W5KUID5MW1UK3QDVLOBOCBJ4QCJBFDLUPINJFK3ZSQA&v=20180605&ll=43.6563221,-79.3809161&radius=500&limit=100'



## Step 10 - Exposing Explore Datas in JSON


```python
results = requests.get(url).json()
results
```




    {'meta': {'code': 200, 'requestId': '5f9202c490fd6244ffd00331'},
     'response': {'suggestedFilters': {'header': 'Tap to show:',
       'filters': [{'name': 'Open now', 'key': 'openNow'}]},
      'headerLocation': 'Bay Street Corridor',
      'headerFullLocation': 'Bay Street Corridor, Toronto',
      'headerLocationGranularity': 'neighborhood',
      'totalResults': 108,
      'suggestedBounds': {'ne': {'lat': 43.6608221045, 'lng': -79.37470788695488},
       'sw': {'lat': 43.651822095499995, 'lng': -79.3871243130451}},
      'groups': [{'type': 'Recommended Places',
        'name': 'recommended',
        'items': [{'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '57eda381498ebe0e6ef40972',
           'name': 'UNIQLO ユニクロ',
           'location': {'address': '220 Yonge St',
            'crossStreet': 'at Dundas St W',
            'lat': 43.65591027779457,
            'lng': -79.38064099181345,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65591027779457,
              'lng': -79.38064099181345}],
            'distance': 50,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'neighborhood': 'Downtown Toronto',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St (at Dundas St W)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d103951735',
             'name': 'Clothing Store',
             'pluralName': 'Clothing Stores',
             'shortName': 'Apparel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-57eda381498ebe0e6ef40972-0'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c062f964a5200bf820e3',
           'name': 'Silver Snail Comics',
           'location': {'address': '329 Yonge St',
            'crossStreet': 'at Dundas St E',
            'lat': 43.65703137958407,
            'lng': -79.38140310220501,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65703137958407,
              'lng': -79.38140310220501}],
            'distance': 88,
            'postalCode': 'M5B 1R7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['329 Yonge St (at Dundas St E)',
             'Toronto ON M5B 1R7',
             'Canada']},
           'categories': [{'id': '52f2ab2ebcbc57f1066b8b18',
             'name': 'Comic Shop',
             'pluralName': 'Comic Shops',
             'shortName': 'Comic Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/comic_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c062f964a5200bf820e3-1'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c062f964a520c5f720e3',
           'name': 'Ed Mirvish Theatre',
           'location': {'address': '244 Victoria St.',
            'crossStreet': 'btwn Dundas St E and Shuter St',
            'lat': 43.655101567321054,
            'lng': -79.37976762131545,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.655101567321054,
              'lng': -79.37976762131545}],
            'distance': 164,
            'postalCode': 'M5B 1V8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['244 Victoria St. (btwn Dundas St E and Shuter St)',
             'Toronto ON M5B 1V8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d137941735',
             'name': 'Theater',
             'pluralName': 'Theaters',
             'shortName': 'Theater',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/performingarts_theater_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c062f964a520c5f720e3-2'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5615b6c4498e3c32c67ad78f',
           'name': 'Blaze Pizza',
           'location': {'address': '10 Dundas Street East, #124',
            'lat': 43.656518,
            'lng': -79.380015,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656518,
              'lng': -79.380015}],
            'distance': 75,
            'postalCode': 'M5B 2G9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['10 Dundas Street East, #124',
             'Toronto ON M5B 2G9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1ca941735',
             'name': 'Pizza Place',
             'pluralName': 'Pizza Places',
             'shortName': 'Pizza',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/pizza_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5615b6c4498e3c32c67ad78f-3'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad8cd16f964a520c91421e3',
           'name': 'Yonge-Dundas Square',
           'location': {'address': '1 Dundas St E',
            'crossStreet': 'at Yonge St',
            'lat': 43.65605389742188,
            'lng': -79.38049504264389,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65605389742188,
              'lng': -79.38049504264389}],
            'distance': 45,
            'postalCode': 'M5B 2R8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['1 Dundas St E (at Yonge St)',
             'Toronto ON M5B 2R8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d164941735',
             'name': 'Plaza',
             'pluralName': 'Plazas',
             'shortName': 'Plaza',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/plaza_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '68861986'}},
          'referralId': 'e-0-4ad8cd16f964a520c91421e3-4'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad77a12f964a520260b21e3',
           'name': 'CF Toronto Eaton Centre',
           'location': {'address': '220 Yonge St',
            'crossStreet': 'btwn Queen & Dundas',
            'lat': 43.654646177209955,
            'lng': -79.38113925615237,
            'distance': 187,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'neighborhood': 'Downtown Toronto, Toronto, ON',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St (btwn Queen & Dundas)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1fd941735',
             'name': 'Shopping Mall',
             'pluralName': 'Shopping Malls',
             'shortName': 'Mall',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/mall_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad77a12f964a520260b21e3-5'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ae3398ff964a520ed9121e3',
           'name': 'Red Lobster',
           'location': {'address': '20 Dundas Street West',
            'crossStreet': 'at Bay St',
            'lat': 43.656328,
            'lng': -79.383621,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656328,
              'lng': -79.383621}],
            'distance': 217,
            'postalCode': 'M5G 2C2',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['20 Dundas Street West (at Bay St)',
             'Toronto ON M5G 2C2',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1ce941735',
             'name': 'Seafood Restaurant',
             'pluralName': 'Seafood Restaurants',
             'shortName': 'Seafood',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/seafood_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ae3398ff964a520ed9121e3-6'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ab2b0b9f964a520e56b20e3',
           'name': 'The Queen and Beaver Public House',
           'location': {'address': '35 Elm St.',
            'crossStreet': 'at Bay St.',
            'lat': 43.65747228208784,
            'lng': -79.38352412327917,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65747228208784,
              'lng': -79.38352412327917}],
            'distance': 245,
            'postalCode': 'M5G 1H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['35 Elm St. (at Bay St.)',
             'Toronto ON M5G 1H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d155941735',
             'name': 'Gastropub',
             'pluralName': 'Gastropubs',
             'shortName': 'Gastropub',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/gastropub_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ab2b0b9f964a520e56b20e3-7'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '55a9bbf9498e00ffd7f4c71f',
           'name': 'Burrito Boyz',
           'location': {'address': '74 Dundas St E',
            'lat': 43.656265264409015,
            'lng': -79.37834318376771,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656265264409015,
              'lng': -79.37834318376771}],
            'distance': 207,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['74 Dundas St E', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d153941735',
             'name': 'Burrito Place',
             'pluralName': 'Burrito Places',
             'shortName': 'Burritos',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/burrito_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-55a9bbf9498e00ffd7f4c71f-8'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5479da4f498e8569fb44985c',
           'name': 'MUJI',
           'location': {'address': '595 Bay St E',
            'crossStreet': 'at Dundas St W',
            'lat': 43.656024,
            'lng': -79.383284,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656024,
              'lng': -79.383284}],
            'distance': 193,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['595 Bay St E (at Dundas St W)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1ff941735',
             'name': 'Miscellaneous Shop',
             'pluralName': 'Miscellaneous Shops',
             'shortName': 'Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5479da4f498e8569fb44985c-9'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4e5d8181a8092f63968617ee',
           'name': 'Crepe Delicious',
           'location': {'address': '220 Yonge St.',
            'lat': 43.654536488277245,
            'lng': -79.38088885547485,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654536488277245,
              'lng': -79.38088885547485}],
            'distance': 198,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St.', 'Toronto ON M5B 2H1', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16e941735',
             'name': 'Fast Food Restaurant',
             'pluralName': 'Fast Food Restaurants',
             'shortName': 'Fast Food',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/fastfood_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '48449575'}},
          'referralId': 'e-0-4e5d8181a8092f63968617ee-10'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5c2151463362730039c4ef0b',
           'name': 'Danish Pastry House',
           'location': {'lat': 43.654574,
            'lng': -79.38074,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654574,
              'lng': -79.38074}],
            'distance': 195,
            'postalCode': 'M5G',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Toronto ON M5G', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16a941735',
             'name': 'Bakery',
             'pluralName': 'Bakeries',
             'shortName': 'Bakery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/bakery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5c2151463362730039c4ef0b-11'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c061f964a52095f720e3',
           'name': 'Salad King',
           'location': {'address': '340 Yonge St',
            'crossStreet': 'at Gould St',
            'lat': 43.65760101432665,
            'lng': -79.38161963017174,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65760101432665,
              'lng': -79.38161963017174}],
            'distance': 153,
            'postalCode': 'M5B 1R7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['340 Yonge St (at Gould St)',
             'Toronto ON M5B 1R7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d149941735',
             'name': 'Thai Restaurant',
             'pluralName': 'Thai Restaurants',
             'shortName': 'Thai',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/thai_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c061f964a52095f720e3-12'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5a43c3ddacb00b66ac2ee7d8',
           'name': 'Samsung Experience Store (Eaton Centre)',
           'location': {'address': 'Toronto Eaton Centre',
            'lat': 43.65564777812532,
            'lng': -79.3810108074576,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65564777812532,
              'lng': -79.3810108074576}],
            'distance': 75,
            'postalCode': 'M5B',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Toronto Eaton Centre',
             'Toronto ON M5B',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d122951735',
             'name': 'Electronics Store',
             'pluralName': 'Electronics Stores',
             'shortName': 'Electronics',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/technology_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5a43c3ddacb00b66ac2ee7d8-13'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '514cc159e4b0e4f73af4eced',
           'name': 'Jazz Bistro',
           'location': {'address': '251 Victoria St',
            'crossStreet': 'btwn Dundas St. E & Shuter St.',
            'lat': 43.65567828473835,
            'lng': -79.37927565514764,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65567828473835,
              'lng': -79.37927565514764}],
            'distance': 150,
            'postalCode': 'M5B 1T8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['251 Victoria St (btwn Dundas St. E & Shuter St.)',
             'Toronto ON M5B 1T8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e5931735',
             'name': 'Music Venue',
             'pluralName': 'Music Venues',
             'shortName': 'Music Venue',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/musicvenue_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '424851737'}},
          'referralId': 'e-0-514cc159e4b0e4f73af4eced-14'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad7929cf964a520500c21e3',
           'name': 'The Senator Restaurant',
           'location': {'address': '249 Victoria Street',
            'crossStreet': 'btwn Dundas St E and Shuter St',
            'lat': 43.65564091455335,
            'lng': -79.37919882575557,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65564091455335,
              'lng': -79.37919882575557}],
            'distance': 157,
            'postalCode': 'M5B 1T8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['249 Victoria Street (btwn Dundas St E and Shuter St)',
             'Toronto ON M5B 1T8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d147941735',
             'name': 'Diner',
             'pluralName': 'Diners',
             'shortName': 'Diner',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/diner_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '55585058'}},
          'referralId': 'e-0-4ad7929cf964a520500c21e3-15'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '539c6f13498e06f4cc765165',
           'name': 'The Elm Tree Restaurant',
           'location': {'address': '43 Elm St',
            'crossStreet': 'Bay',
            'lat': 43.65739749535259,
            'lng': -79.38376054171513,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65739749535259,
              'lng': -79.38376054171513}],
            'distance': 258,
            'postalCode': 'M5G 1H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['43 Elm St (Bay)',
             'Toronto ON M5G 1H1',
             'Canada']},
           'categories': [{'id': '52e81612bcbc57f1066b79f9',
             'name': 'Modern European Restaurant',
             'pluralName': 'Modern European Restaurants',
             'shortName': 'Modern European',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '88534436'}},
          'referralId': 'e-0-539c6f13498e06f4cc765165-16'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '502ad662e4b0cc2719059805',
           'name': 'Ryerson Image Centre',
           'location': {'address': '33 Gould St.',
            'crossStreet': 'at Bond St.',
            'lat': 43.65752337161094,
            'lng': -79.37945958709186,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65752337161094,
              'lng': -79.37945958709186}],
            'distance': 177,
            'postalCode': 'M5B 1X8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['33 Gould St. (at Bond St.)',
             'Toronto ON M5B 1X8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e2931735',
             'name': 'Art Gallery',
             'pluralName': 'Art Galleries',
             'shortName': 'Art Gallery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/artgallery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-502ad662e4b0cc2719059805-17'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ba5208ff964a520f5e038e3',
           'name': 'Solei Tanning Salon',
           'location': {'address': '239 Yonge St.',
            'crossStreet': 'Shuter',
            'lat': 43.654734082347616,
            'lng': -79.38024826065258,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654734082347616,
              'lng': -79.38024826065258}],
            'distance': 184,
            'postalCode': 'M5B 1N8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['239 Yonge St. (Shuter)',
             'Toronto ON M5B 1N8',
             'Canada']},
           'categories': [{'id': '4d1cf8421a97d635ce361c31',
             'name': 'Tanning Salon',
             'pluralName': 'Tanning Salons',
             'shortName': 'Tanning Salon',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/tanning_salon_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '98199950'}},
          'referralId': 'e-0-4ba5208ff964a520f5e038e3-18'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bd0b30d41b9ef3b8fa0fae5',
           'name': 'LUSH',
           'location': {'address': '220 Yonge St, Unit B215-A',
            'crossStreet': 'in Toronto Eaton Centre',
            'lat': 43.653557,
            'lng': -79.3804,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653557,
              'lng': -79.3804}],
            'distance': 310,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St, Unit B215-A (in Toronto Eaton Centre)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10c951735',
             'name': 'Cosmetics Shop',
             'pluralName': 'Cosmetics Shops',
             'shortName': 'Cosmetics',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/beauty_cosmetic_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bd0b30d41b9ef3b8fa0fae5-19'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b2a6eb8f964a52012a924e3',
           'name': 'Indigo',
           'location': {'address': '220 Yonge St',
            'lat': 43.65351471121164,
            'lng': -79.38069591056922,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65351471121164,
              'lng': -79.38069591056922}],
            'distance': 313,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'neighborhood': 'Downtown Yonge',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St', 'Toronto ON M5B 2H1', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d114951735',
             'name': 'Bookstore',
             'pluralName': 'Bookstores',
             'shortName': 'Bookstore',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/bookstore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b2a6eb8f964a52012a924e3-20'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '51755dc7498ece19b7261641',
           'name': 'Banh Mi Boys',
           'location': {'address': '399 Yonge St.',
            'crossStreet': 'Gerrard St.',
            'lat': 43.659292,
            'lng': -79.381949,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.659292,
              'lng': -79.381949}],
            'distance': 340,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['399 Yonge St. (Gerrard St.)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c5941735',
             'name': 'Sandwich Place',
             'pluralName': 'Sandwich Places',
             'shortName': 'Sandwiches',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/deli_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-51755dc7498ece19b7261641-21'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5a81ae339deb7d369fa7f146',
           'name': 'Hailed Coffee',
           'location': {'address': '44 Gerrard St W',
            'crossStreet': 'Yonge St',
            'lat': 43.65883296982352,
            'lng': -79.38368351986598,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65883296982352,
              'lng': -79.38368351986598}],
            'distance': 357,
            'postalCode': 'M5G',
            'cc': 'CA',
            'neighborhood': 'College Park',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['44 Gerrard St W (Yonge St)',
             'Toronto ON M5G',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5a81ae339deb7d369fa7f146-22'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '509e9ef6e4b0ab175389a6c5',
           'name': 'Hokkaido Ramen Santouka らーめん山頭火',
           'location': {'address': '91 Dundas St E',
            'crossStreet': 'at Church St',
            'lat': 43.65643520293576,
            'lng': -79.37758637997793,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65643520293576,
              'lng': -79.37758637997793}],
            'distance': 268,
            'postalCode': 'M5B 1E1',
            'cc': 'CA',
            'neighborhood': 'Downtown Toronto',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['91 Dundas St E (at Church St)',
             'Toronto ON M5B 1E1',
             'Canada']},
           'categories': [{'id': '55a59bace4b013909087cb24',
             'name': 'Ramen Restaurant',
             'pluralName': 'Ramen Restaurants',
             'shortName': 'Ramen',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/ramen_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-509e9ef6e4b0ab175389a6c5-23'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c062f964a520b5f720e3',
           'name': 'Elgin And Winter Garden Theatres',
           'location': {'address': '189 Yonge St',
            'crossStreet': 'btwn Queen St E & Shuter St',
            'lat': 43.653393796019586,
            'lng': -79.3785073962175,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653393796019586,
              'lng': -79.3785073962175}],
            'distance': 379,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['189 Yonge St (btwn Queen St E & Shuter St)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d137941735',
             'name': 'Theater',
             'pluralName': 'Theaters',
             'shortName': 'Theater',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/performingarts_theater_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c062f964a520b5f720e3-24'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4df909dfe4cd2129701c0690',
           'name': 'JOEY Eaton Centre',
           'location': {'address': '1 Dundas St W',
            'lat': 43.6560936540828,
            'lng': -79.38187792357716,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6560936540828,
              'lng': -79.38187792357716}],
            'distance': 81,
            'postalCode': 'M5G 1Z3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['1 Dundas St W', 'Toronto ON M5G 1Z3', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d157941735',
             'name': 'New American Restaurant',
             'pluralName': 'New American Restaurants',
             'shortName': 'New American',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/newamerican_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4df909dfe4cd2129701c0690-25'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b749e98f964a5202ce82de3',
           'name': 'SEPHORA',
           'location': {'address': '220 Yonge Street, Space #3-131',
            'crossStreet': 'in Toronto Eaton Centre',
            'lat': 43.6535272,
            'lng': -79.3801543,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6535272,
              'lng': -79.3801543}],
            'distance': 317,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge Street, Space #3-131 (in Toronto Eaton Centre)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10c951735',
             'name': 'Cosmetics Shop',
             'pluralName': 'Cosmetics Shops',
             'shortName': 'Cosmetics',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/beauty_cosmetic_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b749e98f964a5202ce82de3-26'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4e2284b11fc7c0ef9857d143',
           'name': 'Chatime 日出茶太',
           'location': {'address': '132 Dundas St W',
            'crossStreet': 'btwn Bay & University',
            'lat': 43.65554164147378,
            'lng': -79.38468427043244,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65554164147378,
              'lng': -79.38468427043244}],
            'distance': 315,
            'postalCode': 'M5G 1C3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['132 Dundas St W (btwn Bay & University)',
             'Toronto ON M5G 1C3',
             'Canada']},
           'categories': [{'id': '52e81612bcbc57f1066b7a0c',
             'name': 'Bubble Tea Shop',
             'pluralName': 'Bubble Tea Shops',
             'shortName': 'Bubble Tea',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/bubble_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4e2284b11fc7c0ef9857d143-27'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '55074b57498ef35eeeaafe29',
           'name': "Uncle Tetsu's Cheesecake (Uncle Tetsu's Japanese Cheesecake)",
           'location': {'address': '598 Bay St',
            'crossStreet': 'at Dundas St',
            'lat': 43.65606287245643,
            'lng': -79.38369472446465,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65606287245643,
              'lng': -79.38369472446465}],
            'distance': 225,
            'postalCode': 'M5G 1M5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['598 Bay St (at Dundas St)',
             'Toronto ON M5G 1M5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1d0941735',
             'name': 'Dessert Shop',
             'pluralName': 'Dessert Shops',
             'shortName': 'Desserts',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/dessert_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-55074b57498ef35eeeaafe29-28'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4adc9148f964a520512d21e3',
           'name': 'Chipotle Mexican Grill',
           'location': {'address': '323 Yonge St, Unit 114',
            'crossStreet': 'Yonge & Dundas',
            'lat': 43.65686,
            'lng': -79.38091,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65686,
              'lng': -79.38091}],
            'distance': 59,
            'postalCode': 'M5B 1R7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['323 Yonge St, Unit 114 (Yonge & Dundas)',
             'Toronto ON M5B 1R7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c1941735',
             'name': 'Mexican Restaurant',
             'pluralName': 'Mexican Restaurants',
             'shortName': 'Mexican',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/mexican_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4adc9148f964a520512d21e3-29'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4afec2eef964a520373022e3',
           'name': 'BMV Books',
           'location': {'address': '10 Edward St',
            'crossStreet': 'at Yonge St',
            'lat': 43.657047061091596,
            'lng': -79.38166061431659,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.657047061091596,
              'lng': -79.38166061431659}],
            'distance': 100,
            'postalCode': 'M5G 1C9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['10 Edward St (at Yonge St)',
             'Toronto ON M5G 1C9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d114951735',
             'name': 'Bookstore',
             'pluralName': 'Bookstores',
             'shortName': 'Bookstore',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/bookstore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4afec2eef964a520373022e3-30'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '537773d1498e74a75bb75c1e',
           'name': 'Eggspectation Bell Trinity Square',
           'location': {'address': '483 Bay Street',
            'crossStreet': 'Albert Street',
            'lat': 43.65314383888587,
            'lng': -79.38198016678167,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65314383888587,
              'lng': -79.38198016678167}],
            'distance': 364,
            'postalCode': 'M5G 2C9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['483 Bay Street (Albert Street)',
             'Toronto ON M5G 2C9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d143941735',
             'name': 'Breakfast Spot',
             'pluralName': 'Breakfast Spots',
             'shortName': 'Breakfast',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/breakfast_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '97507838'}},
          'referralId': 'e-0-537773d1498e74a75bb75c1e-31'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4adbbae6f964a520402a21e3',
           'name': 'Cineplex Cinemas',
           'location': {'address': '10 Dundas St E',
            'crossStreet': 'at Yonge St',
            'lat': 43.65612555948613,
            'lng': -79.38039005666784,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65612555948613,
              'lng': -79.38039005666784}],
            'distance': 47,
            'postalCode': 'M5B 2G9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['10 Dundas St E (at Yonge St)',
             'Toronto ON M5B 2G9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d17f941735',
             'name': 'Movie Theater',
             'pluralName': 'Movie Theaters',
             'shortName': 'Movie Theater',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/movietheater_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4adbbae6f964a520402a21e3-32'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4fe4a773e4b0ef61f5212ecd',
           'name': 'Spring Sushi',
           'location': {'address': '10 Dundas St. E',
            'crossStreet': 'at Yonge St.',
            'lat': 43.656252529816086,
            'lng': -79.38065954471074,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656252529816086,
              'lng': -79.38065954471074}],
            'distance': 22,
            'postalCode': 'M5B 2G9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['10 Dundas St. E (at Yonge St.)',
             'Toronto ON M5B 2G9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1d2941735',
             'name': 'Sushi Restaurant',
             'pluralName': 'Sushi Restaurants',
             'shortName': 'Sushi',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/sushi_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4fe4a773e4b0ef61f5212ecd-33'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '56d4d1b3cd1035fe77e1492c',
           'name': 'Page One Cafe',
           'location': {'address': '106 Mutual St',
            'crossStreet': 'btwn Dundas & Gould St',
            'lat': 43.65777161112601,
            'lng': -79.3760725691681,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65777161112601,
              'lng': -79.3760725691681}],
            'distance': 422,
            'postalCode': 'M5B 2R7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['106 Mutual St (btwn Dundas & Gould St)',
             'Toronto ON M5B 2R7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16d941735',
             'name': 'Café',
             'pluralName': 'Cafés',
             'shortName': 'Café',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-56d4d1b3cd1035fe77e1492c-34'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bc1db6e2a89ef3beec8f288',
           'name': 'Ryerson Athletics Centre',
           'location': {'address': '40 Gould St.',
            'crossStreet': 'btwn Victoria & Church',
            'lat': 43.65843443942876,
            'lng': -79.37929610991361,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65843443942876,
              'lng': -79.37929610991361}],
            'distance': 268,
            'postalCode': 'M5B 2K3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['40 Gould St. (btwn Victoria & Church)',
             'Toronto ON M5B 2K3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1a9941735',
             'name': 'College Rec Center',
             'pluralName': 'College Rec Centers',
             'shortName': 'Rec Center',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/education/reccenter_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bc1db6e2a89ef3beec8f288-35'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ae7b27df964a52068ad21e3',
           'name': 'Japango',
           'location': {'address': '122 Elizabeth St.',
            'crossStreet': 'at Dundas St. W',
            'lat': 43.65526771691681,
            'lng': -79.38516506734886,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65526771691681,
              'lng': -79.38516506734886}],
            'distance': 361,
            'postalCode': 'M5G 1P5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['122 Elizabeth St. (at Dundas St. W)',
             'Toronto ON M5G 1P5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1d2941735',
             'name': 'Sushi Restaurant',
             'pluralName': 'Sushi Restaurants',
             'shortName': 'Sushi',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/sushi_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ae7b27df964a52068ad21e3-36'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4fb13c20e4b011e6f93513c0',
           'name': "Balzac's Coffee",
           'location': {'address': '122 Bond Street',
            'crossStreet': 'at Gould St.',
            'lat': 43.65785440672277,
            'lng': -79.37919981155157,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65785440672277,
              'lng': -79.37919981155157}],
            'distance': 219,
            'postalCode': 'M5B 1X8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['122 Bond Street (at Gould St.)',
             'Toronto ON M5B 1X8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4fb13c20e4b011e6f93513c0-37'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '52f6816f11d24a43115dc834',
           'name': 'Scaddabush Italian Kitchen & Bar',
           'location': {'address': '382 Yonge Street, Unit #7',
            'crossStreet': 'Gerrard',
            'lat': 43.658920292028725,
            'lng': -79.38289105381784,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.658920292028725,
              'lng': -79.38289105381784}],
            'distance': 330,
            'postalCode': 'M5B 1S8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['382 Yonge Street, Unit #7 (Gerrard)',
             'Toronto ON M5B 1S8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d110941735',
             'name': 'Italian Restaurant',
             'pluralName': 'Italian Restaurants',
             'shortName': 'Italian',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/italian_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-52f6816f11d24a43115dc834-38'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b9e7418f964a52032e536e3',
           'name': 'Second Cup',
           'location': {'address': '220 Yonge St., Unit D102',
            'crossStreet': 'in Toronto Eaton Centre',
            'lat': 43.65602745564191,
            'lng': -79.38057491935326,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65602745564191,
              'lng': -79.38057491935326}],
            'distance': 42,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'neighborhood': 'Downtown Toronto',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St., Unit D102 (in Toronto Eaton Centre)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b9e7418f964a52032e536e3-39'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b0563c0f964a5200e5822e3',
           'name': 'Marriott Downtown at CF Toronto Eaton Centre',
           'location': {'address': '525 Bay Street',
            'lat': 43.654728444284025,
            'lng': -79.3824216350913,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654728444284025,
              'lng': -79.3824216350913}],
            'distance': 214,
            'postalCode': 'M5G 2L2',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['525 Bay Street',
             'Toronto ON M5G 2L2',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1fa931735',
             'name': 'Hotel',
             'pluralName': 'Hotels',
             'shortName': 'Hotel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/travel/hotel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '129932611'}},
          'referralId': 'e-0-4b0563c0f964a5200e5822e3-40'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '537d4d6d498ec171ba22e7fe',
           'name': "Jimmy's Coffee",
           'location': {'address': '82 Gerrard Street W',
            'crossStreet': 'Gerrard & LaPlante',
            'lat': 43.65842123574496,
            'lng': -79.38561319551111,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65842123574496,
              'lng': -79.38561319551111}],
            'distance': 444,
            'postalCode': 'M5G 1Z4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['82 Gerrard Street W (Gerrard & LaPlante)',
             'Toronto ON M5G 1Z4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-537d4d6d498ec171ba22e7fe-41'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4af72034f964a5202b0622e3',
           'name': 'HomeSense',
           'location': {'address': '195 Yonge Street',
            'crossStreet': 'at Queen St E',
            'lat': 43.6530533,
            'lng': -79.3794496,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6530533,
              'lng': -79.3794496}],
            'distance': 382,
            'postalCode': 'M5B 1M4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['195 Yonge Street (at Queen St E)',
             'Toronto ON M5B 1M4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1f8941735',
             'name': 'Furniture / Home Store',
             'pluralName': 'Furniture / Home Stores',
             'shortName': 'Furniture / Home',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/furniture_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4af72034f964a5202b0622e3-42'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad9ffbbf964a520091d21e3',
           'name': "Jack Astor's Bar & Grill",
           'location': {'address': '10 Dundas St. E',
            'crossStreet': 'at Yonge St.',
            'lat': 43.65601939992059,
            'lng': -79.38032551379719,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65601939992059,
              'lng': -79.38032551379719}],
            'distance': 58,
            'postalCode': 'M5B 0A1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['10 Dundas St. E (at Yonge St.)',
             'Toronto ON M5B 0A1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c4941735',
             'name': 'Restaurant',
             'pluralName': 'Restaurants',
             'shortName': 'Restaurant',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad9ffbbf964a520091d21e3-43'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '50885719498ea7b5aab3a74c',
           'name': 'GoodLife Fitness Toronto Bell Trinity Centre',
           'location': {'address': '483 Bay St',
            'lat': 43.653436,
            'lng': -79.382314,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653436,
              'lng': -79.382314}],
            'distance': 340,
            'postalCode': 'M5G 2C9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['483 Bay St', 'Toronto ON M5G 2C9', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d176941735',
             'name': 'Gym',
             'pluralName': 'Gyms',
             'shortName': 'Gym',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/building/gym_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-50885719498ea7b5aab3a74c-44'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '50a2c200e4b00b5527eb54a0',
           'name': 'Vans',
           'location': {'address': '245 Yonge St.',
            'lat': 43.654825509733335,
            'lng': -79.38024076741966,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654825509733335,
              'lng': -79.38024076741966}],
            'distance': 175,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['245 Yonge St.', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d107951735',
             'name': 'Shoe Store',
             'pluralName': 'Shoe Stores',
             'shortName': 'Shoes',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_shoestore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-50a2c200e4b00b5527eb54a0-45'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c8facf91664b1f79c90aa2f',
           'name': 'College Park Area',
           'location': {'address': 'College St.',
            'lat': 43.659453,
            'lng': -79.383785,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.659453,
              'lng': -79.383785}],
            'distance': 418,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['College St.', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d163941735',
             'name': 'Park',
             'pluralName': 'Parks',
             'shortName': 'Park',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/park_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c8facf91664b1f79c90aa2f-46'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5227bb01498e17bf485e6202',
           'name': 'Downtown Toronto',
           'location': {'lat': 43.65323167517444,
            'lng': -79.38529600606677,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65323167517444,
              'lng': -79.38529600606677}],
            'distance': 492,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Toronto ON', 'Canada']},
           'categories': [{'id': '4f2a25ac4b909258e854f55f',
             'name': 'Neighborhood',
             'pluralName': 'Neighborhoods',
             'shortName': 'Neighborhood',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/neighborhood_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5227bb01498e17bf485e6202-47'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4dc16c718877c00d6ad7f2a6',
           'name': 'Tangerine Café',
           'location': {'address': '221 Yonge Street',
            'crossStreet': 'at Shuter St',
            'lat': 43.653936903308185,
            'lng': -79.37972202364307,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653936903308185,
              'lng': -79.37972202364307}],
            'distance': 282,
            'postalCode': 'M5B 1M4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['221 Yonge Street (at Shuter St)',
             'Toronto ON M5B 1M4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10a951735',
             'name': 'Bank',
             'pluralName': 'Banks',
             'shortName': 'Bank',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/financial_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '34282013'}},
          'referralId': 'e-0-4dc16c718877c00d6ad7f2a6-48'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4cd2e74d3e0c8cfa87971a12',
           'name': 'Oakham Café',
           'location': {'address': '55 Gould St.',
            'crossStreet': 'Chruch',
            'lat': 43.658077944767534,
            'lng': -79.37831497990125,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.658077944767534,
              'lng': -79.37831497990125}],
            'distance': 286,
            'postalCode': 'M5B 2K3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['55 Gould St. (Chruch)',
             'Toronto ON M5B 2K3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16d941735',
             'name': 'Café',
             'pluralName': 'Cafés',
             'shortName': 'Café',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '81783617'}},
          'referralId': 'e-0-4cd2e74d3e0c8cfa87971a12-49'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '557231e3498e540f05f3083c',
           'name': 'Tim Hortons',
           'location': {'address': '70 Gerrard St West',
            'crossStreet': 'Bay St',
            'lat': 43.658569999999976,
            'lng': -79.38512341104502,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.658569999999976,
              'lng': -79.38512341104502}],
            'distance': 421,
            'postalCode': 'M5G 1J5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['70 Gerrard St West (Bay St)',
             'Toronto ON M5G 1J5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-557231e3498e540f05f3083c-50'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '57bcd3b7498e652a678d0378',
           'name': 'Poke Guys',
           'location': {'address': '112 Elizabeth St',
            'crossStreet': 'at Dundas St W',
            'lat': 43.65489527525682,
            'lng': -79.38505238381624,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65489527525682,
              'lng': -79.38505238381624}],
            'distance': 369,
            'postalCode': 'M5G 1P5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['112 Elizabeth St (at Dundas St W)',
             'Toronto ON M5G 1P5',
             'Canada']},
           'categories': [{'id': '5bae9231bedf3950379f89d4',
             'name': 'Poke Place',
             'pluralName': 'Poke Places',
             'shortName': 'Poke Place',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-57bcd3b7498e652a678d0378-51'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4afc79c8f964a520182322e3',
           'name': 'Magic Tailor',
           'location': {'address': '2G-211 Yonge Street',
            'crossStreet': 'Shuter',
            'lat': 43.65374158228865,
            'lng': -79.37974523881418,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65374158228865,
              'lng': -79.37974523881418}],
            'distance': 302,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['2G-211 Yonge Street (Shuter)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d103951735',
             'name': 'Clothing Store',
             'pluralName': 'Clothing Stores',
             'shortName': 'Apparel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4afc79c8f964a520182322e3-52'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ae5df5af964a520c4a221e3',
           'name': 'Bell Trinity Square',
           'location': {'address': '483 Bay St.',
            'crossStreet': 'at Albert St.',
            'lat': 43.65347479872822,
            'lng': -79.38246987630343,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65347479872822,
              'lng': -79.38246987630343}],
            'distance': 340,
            'postalCode': 'M5G 2C9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['483 Bay St. (at Albert St.)',
             'Toronto ON M5G 2C9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d124941735',
             'name': 'Office',
             'pluralName': 'Offices',
             'shortName': 'Office',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/building/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ae5df5af964a520c4a221e3-53'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5a62412b1108ba191da5ba06',
           'name': 'KAKA',
           'location': {'address': '655 Bay Street',
            'crossStreet': '( Bay st & Elm st)',
            'lat': 43.65745745164475,
            'lng': -79.38419169987876,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65745745164475,
              'lng': -79.38419169987876}],
            'distance': 292,
            'postalCode': 'M5G 1Z4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['655 Bay Street (( Bay st & Elm st))',
             'Toronto ON M5G 1Z4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d111941735',
             'name': 'Japanese Restaurant',
             'pluralName': 'Japanese Restaurants',
             'shortName': 'Japanese',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/japanese_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5a62412b1108ba191da5ba06-54'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c05ef964a520a6f620e3',
           'name': 'Nathan Phillips Square',
           'location': {'address': '100 Queen St W',
            'crossStreet': 'at Bay St',
            'lat': 43.65227047322295,
            'lng': -79.38351631164551,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65227047322295,
              'lng': -79.38351631164551}],
            'distance': 497,
            'postalCode': 'M5H 2N1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['100 Queen St W (at Bay St)',
             'Toronto ON M5H 2N1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d164941735',
             'name': 'Plaza',
             'pluralName': 'Plazas',
             'shortName': 'Plaza',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/plaza_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c05ef964a520a6f620e3-55'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '56ccd5cfcd1069ca160a797e',
           'name': 'Tsujiri',
           'location': {'address': '147 Dundas St W',
            'crossStreet': 'at Elizabeth St',
            'lat': 43.65537430780922,
            'lng': -79.38535434742991,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65537430780922,
              'lng': -79.38535434742991}],
            'distance': 372,
            'postalCode': 'M5G 1P5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['147 Dundas St W (at Elizabeth St)',
             'Toronto ON M5G 1P5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1dc931735',
             'name': 'Tea Room',
             'pluralName': 'Tea Rooms',
             'shortName': 'Tea Room',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/tearoom_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-56ccd5cfcd1069ca160a797e-56'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b40f62ef964a52050be25e3',
           'name': 'Roots',
           'location': {'address': '220 Yonge Street, Unit C 32',
            'lat': 43.65361341708363,
            'lng': -79.38024401664734,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65361341708363,
              'lng': -79.38024401664734}],
            'distance': 306,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'neighborhood': 'Downtown Yonge',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge Street, Unit C 32',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d103951735',
             'name': 'Clothing Store',
             'pluralName': 'Clothing Stores',
             'shortName': 'Apparel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b40f62ef964a52050be25e3-57'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b744336f964a520d8d02de3',
           'name': "Somethin' 2 Talk About",
           'location': {'address': '78 Gerrard St W',
            'lat': 43.65839479027968,
            'lng': -79.38533765920816,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65839479027968,
              'lng': -79.38533765920816}],
            'distance': 424,
            'postalCode': 'M5G 1J5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['78 Gerrard St W',
             'Toronto ON M5G 1J5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d115941735',
             'name': 'Middle Eastern Restaurant',
             'pluralName': 'Middle Eastern Restaurants',
             'shortName': 'Middle Eastern',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/middleeastern_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b744336f964a520d8d02de3-58'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad788c8f964a520e40b21e3',
           'name': 'Apple Eaton Centre',
           'location': {'address': '220 Yonge St',
            'crossStreet': 'Queen St W',
            'lat': 43.6528177,
            'lng': -79.3806173,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6528177,
              'lng': -79.3806173}],
            'distance': 390,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St (Queen St W)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d122951735',
             'name': 'Electronics Store',
             'pluralName': 'Electronics Stores',
             'shortName': 'Electronics',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/technology_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad788c8f964a520e40b21e3-59'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '522a616311d2b982aa3b8111',
           'name': 'Disney Store',
           'location': {'address': '220 Yonge Street',
            'crossStreet': 'Eaton Centre',
            'lat': 43.654248,
            'lng': -79.381232,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654248,
              'lng': -79.381232}],
            'distance': 232,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge Street (Eaton Centre)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1f3941735',
             'name': 'Toy / Game Store',
             'pluralName': 'Toy / Game Stores',
             'shortName': 'Toys & Games',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/toys_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-522a616311d2b982aa3b8111-60'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '59bd8286b0405605d8d13ea6',
           'name': 'Coco Fresh Tea & Juice',
           'location': {'address': '372 Yonge Street #2',
            'crossStreet': 'Yonge & Gerrard',
            'lat': 43.65865989038909,
            'lng': -79.38204032226336,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65865989038909,
              'lng': -79.38204032226336}],
            'distance': 275,
            'postalCode': 'M5G',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['372 Yonge Street #2 (Yonge & Gerrard)',
             'Toronto ON M5G',
             'Canada']},
           'categories': [{'id': '52e81612bcbc57f1066b7a0c',
             'name': 'Bubble Tea Shop',
             'pluralName': 'Bubble Tea Shops',
             'shortName': 'Bubble Tea',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/bubble_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-59bd8286b0405605d8d13ea6-61'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '504bbf2ce4b0168121235cbe',
           'name': 'Sansotei Ramen 三草亭',
           'location': {'address': '179 Dundas St. W',
            'crossStreet': 'btwn Centre Ave. & Chestnut St.',
            'lat': 43.655157467561246,
            'lng': -79.38650067479335,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.655157467561246,
              'lng': -79.38650067479335}],
            'distance': 468,
            'postalCode': 'M5G 1Z8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['179 Dundas St. W (btwn Centre Ave. & Chestnut St.)',
             'Toronto ON M5G 1Z8',
             'Canada']},
           'categories': [{'id': '55a59bace4b013909087cb24',
             'name': 'Ramen Restaurant',
             'pluralName': 'Ramen Restaurants',
             'shortName': 'Ramen',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/ramen_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-504bbf2ce4b0168121235cbe-62'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4e9de9fbb803b7506dcf4c41',
           'name': 'Starbucks',
           'location': {'address': '209 Victoria St.',
            'crossStreet': 'in Keenan Research Centre lobby',
            'lat': 43.65446528594945,
            'lng': -79.37891894388453,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65446528594945,
              'lng': -79.37891894388453}],
            'distance': 261,
            'cc': 'CA',
            'neighborhood': 'Downtown Toronto',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['209 Victoria St. (in Keenan Research Centre lobby)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4e9de9fbb803b7506dcf4c41-63'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c05ef964a5208ef620e3',
           'name': 'Old City Hall',
           'location': {'address': '60 Queen Street West',
            'lat': 43.652008800876125,
            'lng': -79.3817442232328,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.652008800876125,
              'lng': -79.3817442232328}],
            'distance': 484,
            'postalCode': 'M5H 1A1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['60 Queen Street West',
             'Toronto ON M5H 1A1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d12d941735',
             'name': 'Monument / Landmark',
             'pluralName': 'Monuments / Landmarks',
             'shortName': 'Landmark',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/building/government_monument_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c05ef964a5208ef620e3-64'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4adf8077f964a5202d7b21e3',
           'name': 'Kabul Express',
           'location': {'address': '126 Dundas Street East',
            'lat': 43.65669146835466,
            'lng': -79.37664313658586,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65669146835466,
              'lng': -79.37664313658586}],
            'distance': 346,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['126 Dundas Street East',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d115941735',
             'name': 'Middle Eastern Restaurant',
             'pluralName': 'Middle Eastern Restaurants',
             'shortName': 'Middle Eastern',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/middleeastern_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4adf8077f964a5202d7b21e3-65'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4adf85e1f964a5206e7b21e3',
           'name': "Hudson's Bay",
           'location': {'address': '176 Yonge St',
            'crossStreet': 'Queen St W',
            'lat': 43.652039815876805,
            'lng': -79.38039146122816,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.652039815876805,
              'lng': -79.38039146122816}],
            'distance': 478,
            'postalCode': 'M5C 2L7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['176 Yonge St (Queen St W)',
             'Toronto ON M5C 2L7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1f6941735',
             'name': 'Department Store',
             'pluralName': 'Department Stores',
             'shortName': 'Department Store',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/departmentstore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4adf85e1f964a5206e7b21e3-66'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '512ce732e4b0aa0ac5f50f05',
           'name': 'Bubble Bath & Spa',
           'location': {'address': '736 Bay Street',
            'crossStreet': 'Bay & Hayter',
            'lat': 43.65904951746615,
            'lng': -79.38534357912432,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65904951746615,
              'lng': -79.38534357912432}],
            'distance': 468,
            'postalCode': 'M5G 2M4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['736 Bay Street (Bay & Hayter)',
             'Toronto ON M5G 2M4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1ed941735',
             'name': 'Spa',
             'pluralName': 'Spas',
             'shortName': 'Spa',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/spa_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-512ce732e4b0aa0ac5f50f05-67'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c05ef964a520e2f620e3',
           'name': 'Textile Museum of Canada',
           'location': {'address': '55 Centre Avenue',
            'crossStreet': 'University Ave. and Dundas St W.',
            'lat': 43.65439630500274,
            'lng': -79.38650010906946,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65439630500274,
              'lng': -79.38650010906946}],
            'distance': 498,
            'postalCode': 'M5G 2H5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['55 Centre Avenue (University Ave. and Dundas St W.)',
             'Toronto ON M5G 2H5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d18f941735',
             'name': 'Art Museum',
             'pluralName': 'Art Museums',
             'shortName': 'Art Museum',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/museum_art_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '56305293'}},
          'referralId': 'e-0-4ad4c05ef964a520e2f620e3-68'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d9c674c48b6224baa310c9f',
           'name': 'EB Games',
           'location': {'address': '267 Yonge St.',
            'crossStreet': 'btwn Dundas St E and Shuter St',
            'lat': 43.65529344166874,
            'lng': -79.38032812642462,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65529344166874,
              'lng': -79.38032812642462}],
            'distance': 123,
            'postalCode': 'M5B 2H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['267 Yonge St. (btwn Dundas St E and Shuter St)',
             'Toronto ON M5B 2H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10b951735',
             'name': 'Video Game Store',
             'pluralName': 'Video Game Stores',
             'shortName': 'Video Games',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/videogames_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d9c674c48b6224baa310c9f-69'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '511b918be4b00262d6b926b6',
           'name': 'Marshalls',
           'location': {'address': '382 Yonge Street',
            'crossStreet': 'Gerrard St',
            'lat': 43.659308,
            'lng': -79.3824621,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.659308,
              'lng': -79.3824621}],
            'distance': 354,
            'postalCode': 'M5B 1S8',
            'cc': 'CA',
            'neighborhood': 'downtown',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['382 Yonge Street (Gerrard St)',
             'Toronto ON M5B 1S8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1f6941735',
             'name': 'Department Store',
             'pluralName': 'Department Stores',
             'shortName': 'Department Store',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/departmentstore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-511b918be4b00262d6b926b6-70'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c9cd4d37c096dcb9034c5d1',
           'name': 'Paramount Fine Foods',
           'location': {'address': '253 Yonge St',
            'crossStreet': 'btwn Dundas St E & Shuter St',
            'lat': 43.65502870847598,
            'lng': -79.38024513018975,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65502870847598,
              'lng': -79.38024513018975}],
            'distance': 153,
            'postalCode': 'M5B 1N8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['253 Yonge St (btwn Dundas St E & Shuter St)',
             'Toronto ON M5B 1N8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d115941735',
             'name': 'Middle Eastern Restaurant',
             'pluralName': 'Middle Eastern Restaurants',
             'shortName': 'Middle Eastern',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/middleeastern_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c9cd4d37c096dcb9034c5d1-71'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4e91f9b546908c033fb57c50',
           'name': 'Topshop',
           'location': {'address': '176 Yonge St',
            'crossStreet': 'in the Bay',
            'lat': 43.65225092432807,
            'lng': -79.3802210071282,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65225092432807,
              'lng': -79.3802210071282}],
            'distance': 456,
            'postalCode': 'M5C 2L7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['176 Yonge St (in the Bay)',
             'Toronto ON M5C 2L7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d108951735',
             'name': "Women's Store",
             'pluralName': "Women's Stores",
             'shortName': "Women's Store",
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_women_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4e91f9b546908c033fb57c50-72'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c063f964a5204af820e3',
           'name': 'Urban Outfitters',
           'location': {'address': '235 Yonge St',
            'crossStreet': 'at Shuter St',
            'lat': 43.65441107845698,
            'lng': -79.38005491917738,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65441107845698,
              'lng': -79.38005491917738}],
            'distance': 223,
            'postalCode': 'M5B 1N8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['235 Yonge St (at Shuter St)',
             'Toronto ON M5B 1N8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d103951735',
             'name': 'Clothing Store',
             'pluralName': 'Clothing Stores',
             'shortName': 'Apparel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c063f964a5204af820e3-73'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '522e017f11d203fd9df25f43',
           'name': 'Hard Candy Fitness',
           'location': {'address': '382 Yonge St',
            'crossStreet': 'Yonge & Gerrard',
            'lat': 43.659556180497816,
            'lng': -79.38244024080487,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.659556180497816,
              'lng': -79.38244024080487}],
            'distance': 380,
            'postalCode': 'M5B 1S8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['382 Yonge St (Yonge & Gerrard)',
             'Toronto ON M5B 1S8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d175941735',
             'name': 'Gym / Fitness Center',
             'pluralName': 'Gyms or Fitness Centers',
             'shortName': 'Gym / Fitness',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/building/gym_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-522e017f11d203fd9df25f43-74'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d1cdda5d2a3a090cbd26772',
           'name': 'Lake Devo',
           'location': {'address': 'Victoria St.',
            'crossStreet': 'at Gould St.',
            'lat': 43.6569936860937,
            'lng': -79.37689788366988,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6569936860937,
              'lng': -79.37689788366988}],
            'distance': 332,
            'postalCode': 'M5B 1W1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Victoria St. (at Gould St.)',
             'Toronto ON M5B 1W1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d161941735',
             'name': 'Lake',
             'pluralName': 'Lakes',
             'shortName': 'Lake',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/lake_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d1cdda5d2a3a090cbd26772-75'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c057c32d3842d7fd03fbe41',
           'name': 'Starbucks',
           'location': {'address': '65 Dundas St. W, Unit 303',
            'crossStreet': 'at Bay St.',
            'lat': 43.65596912045543,
            'lng': -79.38268426666406,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65596912045543,
              'lng': -79.38268426666406}],
            'distance': 147,
            'postalCode': 'M5G 2C5',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['65 Dundas St. W, Unit 303 (at Bay St.)',
             'Toronto ON M5G 2C5',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c057c32d3842d7fd03fbe41-76'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b8eaea1f964a520b03033e3',
           'name': 'Booster Juice',
           'location': {'address': '2 Queen Street East, Suite #110',
            'crossStreet': 'Suite #110',
            'lat': 43.65265752,
            'lng': -79.37845927,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65265752,
              'lng': -79.37845927}],
            'distance': 453,
            'postalCode': 'M5C 3G7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['2 Queen Street East, Suite #110 (Suite #110)',
             'Toronto ON M5C 3G7',
             'Canada']},
           'categories': [{'id': '52f2ab2ebcbc57f1066b8b41',
             'name': 'Smoothie Shop',
             'pluralName': 'Smoothie Shops',
             'shortName': 'Smoothie Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/juicebar_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b8eaea1f964a520b03033e3-77'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c4e2474f53d0f47f18b13a6',
           'name': 'Ethiopiques',
           'location': {'address': '227 Church St.',
            'crossStreet': 'at Dundas St. E',
            'lat': 43.65651274304155,
            'lng': -79.377077748846,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65651274304155,
              'lng': -79.377077748846}],
            'distance': 309,
            'postalCode': 'M5B 1Y7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['227 Church St. (at Dundas St. E)',
             'Toronto ON M5B 1Y7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10a941735',
             'name': 'Ethiopian Restaurant',
             'pluralName': 'Ethiopian Restaurants',
             'shortName': 'Ethiopian',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/ethiopian_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '44256543'}},
          'referralId': 'e-0-4c4e2474f53d0f47f18b13a6-78'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '54516134498e019be8e568fc',
           'name': 'Ted Baker',
           'location': {'address': '220 Yonge St',
            'lat': 43.65284284567131,
            'lng': -79.38032491863406,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65284284567131,
              'lng': -79.38032491863406}],
            'distance': 390,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['220 Yonge St', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d103951735',
             'name': 'Clothing Store',
             'pluralName': 'Clothing Stores',
             'shortName': 'Apparel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-54516134498e019be8e568fc-79'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '52ce14b0498e50457ce11780',
           'name': 'DoubleTree by Hilton',
           'location': {'address': '108 Chestnut Street',
            'crossStreet': 'Dundas St W',
            'lat': 43.6546083,
            'lng': -79.3859415,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6546083,
              'lng': -79.3859415}],
            'distance': 447,
            'postalCode': 'M5G 1R3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['108 Chestnut Street (Dundas St W)',
             'Toronto ON M5G 1R3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1fa931735',
             'name': 'Hotel',
             'pluralName': 'Hotels',
             'shortName': 'Hotel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/travel/hotel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-52ce14b0498e50457ce11780-80'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '523c961b11d2076f0e5dc8f4',
           'name': 'Starbucks',
           'location': {'address': '407 Yonge St',
            'crossStreet': 'at Gerrard St.',
            'lat': 43.659509,
            'lng': -79.382132,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.659509,
              'lng': -79.382132}],
            'distance': 368,
            'postalCode': 'M5B 1S9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['407 Yonge St (at Gerrard St.)',
             'Toronto ON M5B 1S9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-523c961b11d2076f0e5dc8f4-81'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4aea2b14f964a520ebb921e3',
           'name': 'Fine Asian Bowl',
           'location': {'address': '271 Yonge St.',
            'crossStreet': 'btwn Dundas St. E & Shuter St.',
            'lat': 43.6553866,
            'lng': -79.3803264,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6553866,
              'lng': -79.3803264}],
            'distance': 114,
            'postalCode': 'M5B 1N8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['271 Yonge St. (btwn Dundas St. E & Shuter St.)',
             'Toronto ON M5B 1N8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d14a941735',
             'name': 'Vietnamese Restaurant',
             'pluralName': 'Vietnamese Restaurants',
             'shortName': 'Vietnamese',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/vietnamese_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4aea2b14f964a520ebb921e3-82'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '525ebec6498e33c5702cd7d3',
           'name': 'Reds Midtown Tavern',
           'location': {'address': '382 Yonge Street, Unit #6',
            'crossStreet': 'Gerrard St W',
            'lat': 43.659128249470015,
            'lng': -79.38226565563255,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.659128249470015,
              'lng': -79.38226565563255}],
            'distance': 330,
            'postalCode': 'M5B 1S8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['382 Yonge Street, Unit #6 (Gerrard St W)',
             'Toronto ON M5B 1S8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d123941735',
             'name': 'Wine Bar',
             'pluralName': 'Wine Bars',
             'shortName': 'Wine Bar',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/winery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-525ebec6498e33c5702cd7d3-83'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4aefac72f964a520ded921e3',
           'name': 'Imperial Pub',
           'location': {'address': '54 Dundas St. E.',
            'crossStreet': 'at Victoria Ln.',
            'lat': 43.656254269765554,
            'lng': -79.37895528846455,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656254269765554,
              'lng': -79.37895528846455}],
            'distance': 158,
            'postalCode': 'M5B 1C7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['54 Dundas St. E. (at Victoria Ln.)',
             'Toronto ON M5B 1C7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d11b941735',
             'name': 'Pub',
             'pluralName': 'Pubs',
             'shortName': 'Pub',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/pub_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4aefac72f964a520ded921e3-84'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4e88e0d1d3e39d6f4e81a1dd',
           'name': 'Topman',
           'location': {'address': '176 Yonge St',
            'crossStreet': 'at Queen St',
            'lat': 43.652281488745565,
            'lng': -79.38001554865625,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.652281488745565,
              'lng': -79.38001554865625}],
            'distance': 455,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['176 Yonge St (at Queen St)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d103951735',
             'name': 'Clothing Store',
             'pluralName': 'Clothing Stores',
             'shortName': 'Apparel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4e88e0d1d3e39d6f4e81a1dd-85'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '599cb69eb5461876d54ab713',
           'name': 'Katsuya',
           'location': {'address': '66 Gerrard St E',
            'lat': 43.65986035759321,
            'lng': -79.37878806223506,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65986035759321,
              'lng': -79.37878806223506}],
            'distance': 429,
            'postalCode': 'M5B 1G3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['66 Gerrard St E',
             'Toronto ON M5B 1G3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d111941735',
             'name': 'Japanese Restaurant',
             'pluralName': 'Japanese Restaurants',
             'shortName': 'Japanese',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/japanese_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-599cb69eb5461876d54ab713-86'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ada04eff964a520571d21e3',
           'name': "Fran's",
           'location': {'address': '200 Victoria St',
            'crossStreet': 'at Shuter St',
            'lat': 43.65426474929814,
            'lng': -79.37912005538317,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65426474929814,
              'lng': -79.37912005538317}],
            'distance': 270,
            'postalCode': 'M5B 2R3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['200 Victoria St (at Shuter St)',
             'Toronto ON M5B 2R3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d147941735',
             'name': 'Diner',
             'pluralName': 'Diners',
             'shortName': 'Diner',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/diner_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ada04eff964a520571d21e3-87'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d21f9a05c4ca1cdd1f9ad3d',
           'name': 'Ryerson Square',
           'location': {'address': 'Gould St.',
            'crossStreet': 'at Victoria St.',
            'lat': 43.656988428886045,
            'lng': -79.37689617296417,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656988428886045,
              'lng': -79.37689617296417}],
            'distance': 332,
            'postalCode': 'M5B 2K3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Gould St. (at Victoria St.)',
             'Toronto ON M5B 2K3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d162941735',
             'name': 'Other Great Outdoors',
             'pluralName': 'Other Great Outdoors',
             'shortName': 'Other Outdoors',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/outdoors_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d21f9a05c4ca1cdd1f9ad3d-88'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '52463e4711d289b4527b43bf',
           'name': 'Ali Basha Cafè',
           'location': {'address': '147 Dundas St E',
            'lat': 43.6566897,
            'lng': -79.3754589,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6566897,
              'lng': -79.3754589}],
            'distance': 441,
            'postalCode': 'M5B 1E4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['147 Dundas St E',
             'Toronto ON M5B 1E4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d119941735',
             'name': 'Hookah Bar',
             'pluralName': 'Hookah Bars',
             'shortName': 'Hookah Bar',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/hookahbar_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-52463e4711d289b4527b43bf-89'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d077daee350b60c3bbb9242',
           'name': 'Starbucks',
           'location': {'address': '350 Victoria St.',
            'crossStreet': 'in Ryerson Podium Building',
            'lat': 43.65907956058055,
            'lng': -79.38056191501433,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65907956058055,
              'lng': -79.38056191501433}],
            'distance': 308,
            'postalCode': 'M5B 2K3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['350 Victoria St. (in Ryerson Podium Building)',
             'Toronto ON M5B 2K3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d077daee350b60c3bbb9242-90'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '50a3dc33e4b07d69fbfe80ee',
           'name': 'Bed Bath & Beyond',
           'location': {'address': '382 Yonge St',
            'crossStreet': 'Gerrard St. E',
            'lat': 43.6596432,
            'lng': -79.3827044,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.6596432,
              'lng': -79.3827044}],
            'distance': 396,
            'postalCode': 'M5B 1S8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['382 Yonge St (Gerrard St. E)',
             'Toronto ON M5B 1S8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1f8941735',
             'name': 'Furniture / Home Store',
             'pluralName': 'Furniture / Home Stores',
             'shortName': 'Furniture / Home',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/furniture_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-50a3dc33e4b07d69fbfe80ee-91'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b7aa994f964a5200c362fe3',
           'name': 'Tim Hortons',
           'location': {'address': '261 Yonge St',
            'lat': 43.65521249025681,
            'lng': -79.38006296753883,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65521249025681,
              'lng': -79.38006296753883}],
            'distance': 141,
            'postalCode': 'M5B 1N8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['261 Yonge St', 'Toronto ON M5B 1N8', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b7aa994f964a5200c362fe3-92'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4e776dd788775d593ed36897',
           'name': 'Tim Hortons',
           'location': {'address': '245 Church Street',
            'crossStreet': 'Gould Street',
            'lat': 43.65690736298195,
            'lng': -79.3773287381231,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65690736298195,
              'lng': -79.3773287381231}],
            'distance': 296,
            'postalCode': 'M5B 2K3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['245 Church Street (Gould Street)',
             'Toronto ON M5B 2K3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4e776dd788775d593ed36897-93'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ae61cf6f964a520caa421e3',
           'name': 'Pantages Hotel & Spa',
           'location': {'address': '200 Victoria St',
            'crossStreet': 'at Shuter St',
            'lat': 43.65449797039222,
            'lng': -79.37903488923283,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65449797039222,
              'lng': -79.37903488923283}],
            'distance': 253,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['200 Victoria St (at Shuter St)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1fa931735',
             'name': 'Hotel',
             'pluralName': 'Hotels',
             'shortName': 'Hotel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/travel/hotel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ae61cf6f964a520caa421e3-94'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b6a0069f964a520f3c02be3',
           'name': 'Pantages Lounge & Bar',
           'location': {'address': '200 Victoria St.',
            'crossStreet': 'at Shuter St.',
            'lat': 43.654493431269564,
            'lng': -79.3789996004105,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.654493431269564,
              'lng': -79.3789996004105}],
            'distance': 255,
            'postalCode': 'M5B 1W8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['200 Victoria St. (at Shuter St.)',
             'Toronto ON M5B 1W8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d11e941735',
             'name': 'Cocktail Bar',
             'pluralName': 'Cocktail Bars',
             'shortName': 'Cocktail',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/cocktails_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b6a0069f964a520f3c02be3-95'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '53a85b39498e80b97c5bffed',
           'name': "It's All Grk",
           'location': {'address': '101 Dundas St E',
            'crossStreet': 'Church',
            'lat': 43.65678120012301,
            'lng': -79.37682835542365,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65678120012301,
              'lng': -79.37682835542365}],
            'distance': 333,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['101 Dundas St E (Church)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10e941735',
             'name': 'Greek Restaurant',
             'pluralName': 'Greek Restaurants',
             'shortName': 'Greek',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/greek_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-53a85b39498e80b97c5bffed-96'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bddd88f0ee3a593d7e12eb0',
           'name': 'Ryerson Theatre',
           'location': {'address': '43 Gerrard St. E, KHN-162',
            'crossStreet': 'btwn Yonge & Church',
            'lat': 43.65941278506566,
            'lng': -79.37999674976807,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65941278506566,
              'lng': -79.37999674976807}],
            'distance': 351,
            'postalCode': 'M5B 2K8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['43 Gerrard St. E, KHN-162 (btwn Yonge & Church)',
             'Toronto ON M5B 2K8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d17f941735',
             'name': 'Movie Theater',
             'pluralName': 'Movie Theaters',
             'shortName': 'Movie Theater',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/movietheater_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bddd88f0ee3a593d7e12eb0-97'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d964b2397d06ea8849fff0a',
           'name': 'Subway',
           'location': {'address': '173 Church Street, Unit #4',
            'crossStreet': 'at Shuter St.',
            'lat': 43.65461271472636,
            'lng': -79.37618851036927,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65461271472636,
              'lng': -79.37618851036927}],
            'distance': 425,
            'postalCode': 'M5B 1Y4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['173 Church Street, Unit #4 (at Shuter St.)',
             'Toronto ON M5B 1Y4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c5941735',
             'name': 'Sandwich Place',
             'pluralName': 'Sandwich Places',
             'shortName': 'Sandwiches',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/deli_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d964b2397d06ea8849fff0a-98'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4adfbe44f964a5202d7d21e3',
           'name': 'The Ram in the Rye',
           'location': {'address': '63 Gould St.',
            'crossStreet': 'Church',
            'lat': 43.65794559312108,
            'lng': -79.37783573199916,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65794559312108,
              'lng': -79.37783573199916}],
            'distance': 306,
            'postalCode': 'M5B 1E9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['63 Gould St. (Church)',
             'Toronto ON M5B 1E9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d116941735',
             'name': 'Bar',
             'pluralName': 'Bars',
             'shortName': 'Bar',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/pub_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '81783148'}},
          'referralId': 'e-0-4adfbe44f964a5202d7d21e3-99'}]}]}}



## Step 11 - Functions and Pre-adjust Util


```python
## REASONING COPIED FROM PREVIOUS EXERCISE
## Support Functions

def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighbourhood', 
                  'Neighbourhood Latitude', 
                  'Neighbourhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)

def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending = False)
    
    return row_categories_sorted.index.values[0:num_top_venues]

```


```python
## REASONING COPIED FROM PREVIOUS EXERCISE

venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues = nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis = 1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>UNIQLO ユニクロ</td>
      <td>Clothing Store</td>
      <td>43.655910</td>
      <td>-79.380641</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Silver Snail Comics</td>
      <td>Comic Shop</td>
      <td>43.657031</td>
      <td>-79.381403</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ed Mirvish Theatre</td>
      <td>Theater</td>
      <td>43.655102</td>
      <td>-79.379768</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Blaze Pizza</td>
      <td>Pizza Place</td>
      <td>43.656518</td>
      <td>-79.380015</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Yonge-Dundas Square</td>
      <td>Plaza</td>
      <td>43.656054</td>
      <td>-79.380495</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CF Toronto Eaton Centre</td>
      <td>Shopping Mall</td>
      <td>43.654646</td>
      <td>-79.381139</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Red Lobster</td>
      <td>Seafood Restaurant</td>
      <td>43.656328</td>
      <td>-79.383621</td>
    </tr>
    <tr>
      <th>7</th>
      <td>The Queen and Beaver Public House</td>
      <td>Gastropub</td>
      <td>43.657472</td>
      <td>-79.383524</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Burrito Boyz</td>
      <td>Burrito Place</td>
      <td>43.656265</td>
      <td>-79.378343</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MUJI</td>
      <td>Miscellaneous Shop</td>
      <td>43.656024</td>
      <td>-79.383284</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Crepe Delicious</td>
      <td>Fast Food Restaurant</td>
      <td>43.654536</td>
      <td>-79.380889</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Danish Pastry House</td>
      <td>Bakery</td>
      <td>43.654574</td>
      <td>-79.380740</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Salad King</td>
      <td>Thai Restaurant</td>
      <td>43.657601</td>
      <td>-79.381620</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Samsung Experience Store (Eaton Centre)</td>
      <td>Electronics Store</td>
      <td>43.655648</td>
      <td>-79.381011</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Jazz Bistro</td>
      <td>Music Venue</td>
      <td>43.655678</td>
      <td>-79.379276</td>
    </tr>
    <tr>
      <th>15</th>
      <td>The Senator Restaurant</td>
      <td>Diner</td>
      <td>43.655641</td>
      <td>-79.379199</td>
    </tr>
    <tr>
      <th>16</th>
      <td>The Elm Tree Restaurant</td>
      <td>Modern European Restaurant</td>
      <td>43.657397</td>
      <td>-79.383761</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ryerson Image Centre</td>
      <td>Art Gallery</td>
      <td>43.657523</td>
      <td>-79.379460</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Solei Tanning Salon</td>
      <td>Tanning Salon</td>
      <td>43.654734</td>
      <td>-79.380248</td>
    </tr>
    <tr>
      <th>19</th>
      <td>LUSH</td>
      <td>Cosmetics Shop</td>
      <td>43.653557</td>
      <td>-79.380400</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Indigo</td>
      <td>Bookstore</td>
      <td>43.653515</td>
      <td>-79.380696</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Banh Mi Boys</td>
      <td>Sandwich Place</td>
      <td>43.659292</td>
      <td>-79.381949</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Hailed Coffee</td>
      <td>Coffee Shop</td>
      <td>43.658833</td>
      <td>-79.383684</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Hokkaido Ramen Santouka らーめん山頭火</td>
      <td>Ramen Restaurant</td>
      <td>43.656435</td>
      <td>-79.377586</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Elgin And Winter Garden Theatres</td>
      <td>Theater</td>
      <td>43.653394</td>
      <td>-79.378507</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
```

    100 venues were returned by Foursquare.



```python
downtown_venues = getNearbyVenues(names = ds_downtown['Neighbourhood'],
                                  latitudes = ds_downtown['Latitude'],
                                  longitudes = ds_downtown['Longitude']
                                 )
```

    Regent Park, Harbourfront
    Queen's Park, Ontario Provincial Government
    Garden District, Ryerson
    St. James Town
    Berczy Park
    Central Bay Street
    Christie
    Richmond, Adelaide, King
    Harbourfront East, Union Station, Toronto Islands
    Toronto Dominion Centre, Design Exchange
    Commerce Court, Victoria Hotel
    University of Toronto, Harbord
    Kensington Market, Chinatown, Grange Park
    CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport
    Rosedale
    Stn A PO Boxes
    St. James Town, Cabbagetown
    First Canadian Place, Underground city
    Church and Wellesley


## Step 12 - Checking Result in Data Frame


```python
print(downtown_venues.shape)
downtown_venues.head(25)
```

    (1248, 7)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Neighbourhood Latitude</th>
      <th>Neighbourhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Roselle Desserts</td>
      <td>43.653447</td>
      <td>-79.362017</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Tandem Coffee</td>
      <td>43.653559</td>
      <td>-79.361809</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Cooper Koo Family YMCA</td>
      <td>43.653249</td>
      <td>-79.358008</td>
      <td>Distribution Center</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Body Blitz Spa East</td>
      <td>43.654735</td>
      <td>-79.359874</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Impact Kitchen</td>
      <td>43.656369</td>
      <td>-79.356980</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Corktown Common</td>
      <td>43.655618</td>
      <td>-79.356211</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Dominion Pub and Kitchen</td>
      <td>43.656919</td>
      <td>-79.358967</td>
      <td>Pub</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>The Distillery Historic District</td>
      <td>43.650244</td>
      <td>-79.359323</td>
      <td>Historic Site</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Morning Glory Cafe</td>
      <td>43.653947</td>
      <td>-79.361149</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>The Extension Room</td>
      <td>43.653313</td>
      <td>-79.359725</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Distillery Sunday Market</td>
      <td>43.650075</td>
      <td>-79.361832</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>SOMA chocolatemaker</td>
      <td>43.650622</td>
      <td>-79.358127</td>
      <td>Chocolate Shop</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Figs Breakfast &amp; Lunch</td>
      <td>43.655675</td>
      <td>-79.364503</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Arvo</td>
      <td>43.649963</td>
      <td>-79.361442</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Rooster Coffee</td>
      <td>43.651900</td>
      <td>-79.365609</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Sumach Espresso</td>
      <td>43.658135</td>
      <td>-79.359515</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Starbucks</td>
      <td>43.651613</td>
      <td>-79.364917</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Cacao 70</td>
      <td>43.650067</td>
      <td>-79.360723</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Underpass Park</td>
      <td>43.655764</td>
      <td>-79.354806</td>
      <td>Park</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Young Centre for the Performing Arts</td>
      <td>43.650825</td>
      <td>-79.357593</td>
      <td>Performing Arts Venue</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Alumnae Theatre</td>
      <td>43.652756</td>
      <td>-79.364753</td>
      <td>Theater</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>El Catrin</td>
      <td>43.650601</td>
      <td>-79.358920</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Cluny Bistro &amp; Boulangerie</td>
      <td>43.650565</td>
      <td>-79.357843</td>
      <td>French Restaurant</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Brick Street Bakery</td>
      <td>43.650574</td>
      <td>-79.359539</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>The Yoga Lounge</td>
      <td>43.655515</td>
      <td>-79.364955</td>
      <td>Yoga Studio</td>
    </tr>
  </tbody>
</table>
</div>




```python
downtown_venues.groupby('Neighbourhood').count()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood Latitude</th>
      <th>Neighbourhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighbourhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Berczy Park</th>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
      <td>55</td>
    </tr>
    <tr>
      <th>CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
      <td>68</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Commerce Court, Victoria Hotel</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>First Canadian Place, Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Garden District, Ryerson</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Harbourfront East, Union Station, Toronto Islands</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Kensington Market, Chinatown, Grange Park</th>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
    </tr>
    <tr>
      <th>Queen's Park, Ontario Provincial Government</th>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
      <td>33</td>
    </tr>
    <tr>
      <th>Regent Park, Harbourfront</th>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
      <td>44</td>
    </tr>
    <tr>
      <th>Richmond, Adelaide, King</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
      <td>85</td>
    </tr>
    <tr>
      <th>St. James Town, Cabbagetown</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes</th>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
      <td>96</td>
    </tr>
    <tr>
      <th>Toronto Dominion Centre, Design Exchange</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>University of Toronto, Harbord</th>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
      <td>34</td>
    </tr>
  </tbody>
</table>
</div>



## Step 13 - Check Unique Categories


```python
print('There are {} uniques categories.'.format(len(downtown_venues['Venue Category'].unique())))
```

    There are 213 uniques categories.


## Step 14 - Analyzing Each Neighborhood


```python
# one hot encoding
downtown_onehot = pd.get_dummies(downtown_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
downtown_onehot['Neighbourhood'] = downtown_venues['Neighbourhood'] 

# move neighborhood column to the first column
fixed_columns = [downtown_onehot.columns[-1]] + list(downtown_onehot.columns[:-1])
downtown_onehot = downtown_onehot[fixed_columns]

downtown_onehot.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>...</th>
      <th>Theme Restaurant</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 214 columns</p>
</div>




```python
downtown_onehot.shape
```




    (1248, 214)




```python
downtown_grouped = downtown_onehot.groupby('Neighbourhood').mean().reset_index()
downtown_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>Antique Shop</th>
      <th>...</th>
      <th>Theme Restaurant</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Women's Store</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.018182</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>0.000000</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.0625</td>
      <td>0.125</td>
      <td>0.125</td>
      <td>0.0625</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Central Bay Street</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.014706</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014706</td>
      <td>0.00</td>
      <td>0.014706</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Church and Wellesley</td>
      <td>0.013333</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.013333</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.013333</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.013333</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.026667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>First Canadian Place, Underground city</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.040000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Garden District, Ryerson</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Kensington Market, Chinatown, Grange Park</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.054054</td>
      <td>0.000000</td>
      <td>0.040541</td>
      <td>0.013514</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.030303</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Regent Park, Harbourfront</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.022727</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.022727</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Richmond, Adelaide, King</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.01</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>St. James Town</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.035294</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.011765</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011765</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>St. James Town, Cabbagetown</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Stn A PO Boxes</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.010417</td>
      <td>0.010417</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010417</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.010417</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.00</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>University of Toronto, Harbord</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.029412</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.029412</td>
    </tr>
  </tbody>
</table>
<p>19 rows × 214 columns</p>
</div>




```python
downtown_grouped.shape
```




    (19, 214)




```python
num_top_venues = 5

for hood in downtown_grouped['Neighbourhood']:
    print("----"+hood+"----")
    temp = downtown_grouped[downtown_grouped['Neighbourhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending = False).reset_index(drop = True).head(num_top_venues))
    print('\n')
```

    ----Berczy Park----
                    venue  freq
    0         Coffee Shop  0.09
    1  Seafood Restaurant  0.04
    2         Cheese Shop  0.04
    3        Cocktail Bar  0.04
    4            Beer Bar  0.04
    
    
    ----CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport----
                     venue  freq
    0       Airport Lounge  0.12
    1      Airport Service  0.12
    2      Harbor / Marina  0.06
    3              Airport  0.06
    4  Rental Car Location  0.06
    
    
    ----Central Bay Street----
                    venue  freq
    0         Coffee Shop  0.18
    1                Café  0.06
    2  Italian Restaurant  0.04
    3      Sandwich Place  0.04
    4     Bubble Tea Shop  0.03
    
    
    ----Christie----
               venue  freq
    0  Grocery Store  0.25
    1           Café  0.19
    2           Park  0.12
    3     Restaurant  0.06
    4     Baby Store  0.06
    
    
    ----Church and Wellesley----
                     venue  freq
    0          Coffee Shop  0.09
    1  Japanese Restaurant  0.05
    2     Sushi Restaurant  0.05
    3              Gay Bar  0.05
    4           Restaurant  0.04
    
    
    ----Commerce Court, Victoria Hotel----
                     venue  freq
    0          Coffee Shop  0.13
    1           Restaurant  0.07
    2                 Café  0.06
    3                Hotel  0.06
    4  American Restaurant  0.04
    
    
    ----First Canadian Place, Underground city----
                     venue  freq
    0          Coffee Shop  0.10
    1                 Café  0.07
    2                Hotel  0.06
    3                  Gym  0.04
    4  Japanese Restaurant  0.04
    
    
    ----Garden District, Ryerson----
                     venue  freq
    0       Clothing Store  0.10
    1          Coffee Shop  0.09
    2                 Café  0.04
    3       Cosmetics Shop  0.03
    4  Japanese Restaurant  0.03
    
    
    ----Harbourfront East, Union Station, Toronto Islands----
                venue  freq
    0     Coffee Shop  0.13
    1        Aquarium  0.05
    2            Café  0.04
    3           Hotel  0.04
    4  Scenic Lookout  0.03
    
    
    ----Kensington Market, Chinatown, Grange Park----
                               venue  freq
    0                           Café  0.05
    1  Vegetarian / Vegan Restaurant  0.05
    2                    Coffee Shop  0.05
    3             Mexican Restaurant  0.05
    4                            Bar  0.05
    
    
    ----Queen's Park, Ontario Provincial Government----
                  venue  freq
    0       Coffee Shop  0.24
    1       Yoga Studio  0.03
    2       Music Venue  0.03
    3  Sushi Restaurant  0.03
    4          Beer Bar  0.03
    
    
    ----Regent Park, Harbourfront----
                venue  freq
    0     Coffee Shop  0.18
    1          Bakery  0.07
    2             Pub  0.07
    3            Park  0.07
    4  Breakfast Spot  0.05
    
    
    ----Richmond, Adelaide, King----
             venue  freq
    0  Coffee Shop  0.08
    1         Café  0.05
    2          Bar  0.04
    3        Hotel  0.04
    4   Restaurant  0.04
    
    
    ----Rosedale----
                   venue  freq
    0               Park  0.50
    1         Playground  0.25
    2              Trail  0.25
    3  Afghan Restaurant  0.00
    4             Museum  0.00
    
    
    ----St. James Town----
              venue  freq
    0   Coffee Shop  0.07
    1          Café  0.06
    2    Restaurant  0.05
    3  Cocktail Bar  0.05
    4      Beer Bar  0.04
    
    
    ----St. James Town, Cabbagetown----
                    venue  freq
    0         Coffee Shop  0.08
    1          Restaurant  0.06
    2         Pizza Place  0.06
    3                Café  0.06
    4  Italian Restaurant  0.04
    
    
    ----Stn A PO Boxes----
                     venue  freq
    0          Coffee Shop  0.10
    1   Italian Restaurant  0.04
    2  Japanese Restaurant  0.03
    3                 Café  0.03
    4                Hotel  0.03
    
    
    ----Toronto Dominion Centre, Design Exchange----
                     venue  freq
    0          Coffee Shop  0.14
    1                Hotel  0.08
    2           Restaurant  0.05
    3                 Café  0.05
    4  Japanese Restaurant  0.03
    
    
    ----University of Toronto, Harbord----
                     venue  freq
    0                 Café  0.15
    1            Bookstore  0.09
    2               Bakery  0.06
    3  Japanese Restaurant  0.06
    4       Sandwich Place  0.06
    
    



```python
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighbourhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind + 1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind + 1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighbourhood'] = downtown_grouped['Neighbourhood']

for ind in np.arange(downtown_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(downtown_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Cheese Shop</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Seafood Restaurant</td>
      <td>Beer Bar</td>
      <td>Farmers Market</td>
      <td>Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Shopping Mall</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>Airport Lounge</td>
      <td>Airport Service</td>
      <td>Harbor / Marina</td>
      <td>Bar</td>
      <td>Plane</td>
      <td>Rental Car Location</td>
      <td>Sculpture Garden</td>
      <td>Boutique</td>
      <td>Boat or Ferry</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Central Bay Street</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Salad Place</td>
      <td>Department Store</td>
      <td>Japanese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Christie</td>
      <td>Grocery Store</td>
      <td>Café</td>
      <td>Park</td>
      <td>Athletics &amp; Sports</td>
      <td>Italian Restaurant</td>
      <td>Candy Store</td>
      <td>Restaurant</td>
      <td>Baby Store</td>
      <td>Nightclub</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Church and Wellesley</td>
      <td>Coffee Shop</td>
      <td>Japanese Restaurant</td>
      <td>Gay Bar</td>
      <td>Sushi Restaurant</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Men's Store</td>
      <td>Mediterranean Restaurant</td>
      <td>Hotel</td>
      <td>Yoga Studio</td>
    </tr>
  </tbody>
</table>
</div>



## Step 15 - Cluster Neighborhoods


```python
# set number of clusters
kclusters = 5

downtown_grouped_clustering = downtown_grouped.drop('Neighbourhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters = kclusters, random_state = 0).fit(downtown_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([4, 3, 0, 2, 4, 4, 4, 4, 0, 4], dtype=int32)




```python
# add clustering labels
downtown_merged = ds_downtown
downtown_merged.head(25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M5G</td>
      <td>Downtown Toronto</td>
      <td>Central Bay Street</td>
      <td>43.657952</td>
      <td>-79.387383</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M6G</td>
      <td>Downtown Toronto</td>
      <td>Christie</td>
      <td>43.669542</td>
      <td>-79.422564</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M5H</td>
      <td>Downtown Toronto</td>
      <td>Richmond, Adelaide, King</td>
      <td>43.650571</td>
      <td>-79.384568</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M5J</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>43.640816</td>
      <td>-79.381752</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M5K</td>
      <td>Downtown Toronto</td>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>43.647177</td>
      <td>-79.381576</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M5L</td>
      <td>Downtown Toronto</td>
      <td>Commerce Court, Victoria Hotel</td>
      <td>43.648198</td>
      <td>-79.379817</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M5S</td>
      <td>Downtown Toronto</td>
      <td>University of Toronto, Harbord</td>
      <td>43.662696</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>12</th>
      <td>M5T</td>
      <td>Downtown Toronto</td>
      <td>Kensington Market, Chinatown, Grange Park</td>
      <td>43.653206</td>
      <td>-79.400049</td>
    </tr>
    <tr>
      <th>13</th>
      <td>M5V</td>
      <td>Downtown Toronto</td>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>43.628947</td>
      <td>-79.394420</td>
    </tr>
    <tr>
      <th>14</th>
      <td>M4W</td>
      <td>Downtown Toronto</td>
      <td>Rosedale</td>
      <td>43.679563</td>
      <td>-79.377529</td>
    </tr>
    <tr>
      <th>15</th>
      <td>M5W</td>
      <td>Downtown Toronto</td>
      <td>Stn A PO Boxes</td>
      <td>43.646435</td>
      <td>-79.374846</td>
    </tr>
    <tr>
      <th>16</th>
      <td>M4X</td>
      <td>Downtown Toronto</td>
      <td>St. James Town, Cabbagetown</td>
      <td>43.667967</td>
      <td>-79.367675</td>
    </tr>
    <tr>
      <th>17</th>
      <td>M5X</td>
      <td>Downtown Toronto</td>
      <td>First Canadian Place, Underground city</td>
      <td>43.648429</td>
      <td>-79.382280</td>
    </tr>
    <tr>
      <th>18</th>
      <td>M4Y</td>
      <td>Downtown Toronto</td>
      <td>Church and Wellesley</td>
      <td>43.665860</td>
      <td>-79.383160</td>
    </tr>
  </tbody>
</table>
</div>




```python
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
downtown_merged = ds_downtown
downtown_merged = downtown_merged.join(neighborhoods_venues_sorted.set_index('Neighbourhood'), on = 'Neighbourhood')
downtown_merged.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Borough</th>
      <th>Neighbourhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Beer Store</td>
      <td>Mexican Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.662301</td>
      <td>-79.389494</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Portuguese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Smoothie Shop</td>
      <td>Beer Bar</td>
      <td>Sandwich Place</td>
      <td>Distribution Center</td>
      <td>Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.657162</td>
      <td>-79.378937</td>
      <td>4</td>
      <td>Clothing Store</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Cosmetics Shop</td>
      <td>Bubble Tea Shop</td>
      <td>Japanese Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Hotel</td>
      <td>Fast Food Restaurant</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
      <td>4</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Cocktail Bar</td>
      <td>Restaurant</td>
      <td>Gastropub</td>
      <td>Beer Bar</td>
      <td>American Restaurant</td>
      <td>Gym</td>
      <td>Farmers Market</td>
      <td>Hotel</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M5E</td>
      <td>Downtown Toronto</td>
      <td>Berczy Park</td>
      <td>43.644771</td>
      <td>-79.373306</td>
      <td>4</td>
      <td>Coffee Shop</td>
      <td>Cheese Shop</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Seafood Restaurant</td>
      <td>Beer Bar</td>
      <td>Farmers Market</td>
      <td>Restaurant</td>
      <td>Breakfast Spot</td>
      <td>Shopping Mall</td>
    </tr>
  </tbody>
</table>
</div>



## Step 16 - Show Cluster on Map


```python
# create map
map_clusters = folium.Map(location = [latitude_downtown, longitude_downtown], zoom_start = 11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i * x) ** 2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(downtown_merged['Latitude'], downtown_merged['Longitude'], downtown_merged['Neighbourhood'], downtown_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html = True)
    folium.CircleMarker(
        [lat, lon],
        radius = 5,
        popup = label,
        color = rainbow[cluster - 1],
        fill = True,
        fill_color = rainbow[cluster - 1],
        fill_opacity = 0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDggewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjU2MzIyMSwtNzkuMzgwOTE2MV0sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB6b29tOiAxMSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIG1heEJvdW5kczogYm91bmRzLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgbGF5ZXJzOiBbXSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIHdvcmxkQ29weUp1bXA6IGZhbHNlLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY3JzOiBMLkNSUy5FUFNHMzg1NwogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB9KTsKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHRpbGVfbGF5ZXJfYWYzNDg1NDFiMDFmNGUzYmFhYWRjN2JmZWMyNDliZmUgPSBMLnRpbGVMYXllcigKICAgICAgICAgICAgICAgICdodHRwczovL3tzfS50aWxlLm9wZW5zdHJlZXRtYXAub3JnL3t6fS97eH0ve3l9LnBuZycsCiAgICAgICAgICAgICAgICB7CiAgImF0dHJpYnV0aW9uIjogbnVsbCwKICAiZGV0ZWN0UmV0aW5hIjogZmFsc2UsCiAgIm1heFpvb20iOiAxOCwKICAibWluWm9vbSI6IDEsCiAgIm5vV3JhcCI6IGZhbHNlLAogICJzdWJkb21haW5zIjogImFiYyIKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDgpOwogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzI5NDIzM2Y0ZWU5NjQ4ZDA5NGFmMzE2NmVlNTk5MTAzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOWZjYTRiNjNjMTg0ZWIyODVkMDUyYjQ5OTM3NGRiYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMmQ1Nzc3NzUzNDI0NmRjYmM4NWE5M2YxNzgxYjAxNCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDJkNTc3Nzc1MzQyNDZkY2JjODVhOTNmMTc4MWIwMTQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZ2VudCBQYXJrLCBIYXJib3VyZnJvbnQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yOWZjYTRiNjNjMTg0ZWIyODVkMDUyYjQ5OTM3NGRiYi5zZXRDb250ZW50KGh0bWxfZDJkNTc3Nzc1MzQyNDZkY2JjODVhOTNmMTc4MWIwMTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjk0MjMzZjRlZTk2NDhkMDk0YWYzMTY2ZWU1OTkxMDMuYmluZFBvcHVwKHBvcHVwXzI5ZmNhNGI2M2MxODRlYjI4NWQwNTJiNDk5Mzc0ZGJiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYwYjIxMDFiYjg4OTQwNTZiMjVkM2E1NjRkMzJiYzJlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyMzAxNSwtNzkuMzg5NDkzOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOWRjNTgwNmYxYjc0ZDRkYmNjMzcyZGI2MzlhNmY1NSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kMTU1MDQ5ZDk1YTQ0YmQ3YmVkYTc5ZmU4NGQ0YWZkZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZDE1NTA0OWQ5NWE0NGJkN2JlZGE3OWZlODRkNGFmZGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIE9udGFyaW8gUHJvdmluY2lhbCBHb3Zlcm5tZW50IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzlkYzU4MDZmMWI3NGQ0ZGJjYzM3MmRiNjM5YTZmNTUuc2V0Q29udGVudChodG1sX2QxNTUwNDlkOTVhNDRiZDdiZWRhNzlmZTg0ZDRhZmRmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYwYjIxMDFiYjg4OTQwNTZiMjVkM2E1NjRkMzJiYzJlLmJpbmRQb3B1cChwb3B1cF8zOWRjNTgwNmYxYjc0ZDRkYmNjMzcyZGI2MzlhNmY1NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNzZhZmRhNWRkYmU0NTgyYjFiNmIwYzExYzYwYzllYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZiNTFmYjIyMjIwODQwNzZhMjAxMGFhMjVjNjdkNGY2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNmMmRhMWM5Yzg5ZTQyOWY4M2E3MjVhMmI5NjJjY2FlID0gJCgnPGRpdiBpZD0iaHRtbF8zZjJkYTFjOWM4OWU0MjlmODNhNzI1YTJiOTYyY2NhZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R2FyZGVuIERpc3RyaWN0LCBSeWVyc29uIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmI1MWZiMjIyMjA4NDA3NmEyMDEwYWEyNWM2N2Q0ZjYuc2V0Q29udGVudChodG1sXzNmMmRhMWM5Yzg5ZTQyOWY4M2E3MjVhMmI5NjJjY2FlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM3NmFmZGE1ZGRiZTQ1ODJiMWI2YjBjMTFjNjBjOWViLmJpbmRQb3B1cChwb3B1cF82YjUxZmIyMjIyMDg0MDc2YTIwMTBhYTI1YzY3ZDRmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYjdjZDRhN2ZmYTE0Njg2ODQxM2MzY2Q5YjAxMWFkNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTY3NDk1YzE0YWVlNDdlZWEzOTAzZTBlZDY2YzdiNDYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDI4MWI0NWY4ZTM2NDg2ZjhlZDdmZDExNDQ1NjI4NWEgPSAkKCc8ZGl2IGlkPSJodG1sXzAyODFiNDVmOGUzNjQ4NmY4ZWQ3ZmQxMTQ0NTYyODVhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93biBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E2NzQ5NWMxNGFlZTQ3ZWVhMzkwM2UwZWQ2NmM3YjQ2LnNldENvbnRlbnQoaHRtbF8wMjgxYjQ1ZjhlMzY0ODZmOGVkN2ZkMTE0NDU2Mjg1YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYjdjZDRhN2ZmYTE0Njg2ODQxM2MzY2Q5YjAxMWFkNy5iaW5kUG9wdXAocG9wdXBfYTY3NDk1YzE0YWVlNDdlZWEzOTAzZTBlZDY2YzdiNDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWU5YmM1OGVlMGZlNDM4ODhjZWRjNjZhODJhN2Q0YmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzMzNDNhMjg3YTEzNGJkOTkyNTNjYjYyZjVjZjVhZTIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTI2NWRjM2I4NmY5NDQ0MWJjYjFiMDkyNWQzMmFlZDIgPSAkKCc8ZGl2IGlkPSJodG1sX2EyNjVkYzNiODZmOTQ0NDFiY2IxYjA5MjVkMzJhZWQyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5CZXJjenkgUGFyayBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MzMzQzYTI4N2ExMzRiZDk5MjUzY2I2MmY1Y2Y1YWUyLnNldENvbnRlbnQoaHRtbF9hMjY1ZGMzYjg2Zjk0NDQxYmNiMWIwOTI1ZDMyYWVkMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZTliYzU4ZWUwZmU0Mzg4OGNlZGM2NmE4MmE3ZDRiZi5iaW5kUG9wdXAocG9wdXBfYzMzNDNhMjg3YTEzNGJkOTkyNTNjYjYyZjVjZjVhZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjQyZDcyNmM1NTEyNDc4NGJiNDFiMzY3YmNkYTRmNTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2EzYjNhZDg1YjU4MTQwODM4OWVmODU5MjZmNWM5ZTg5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhmZTVmY2Y0NzU5ODRiYmU5MmZjMzBmNjU0MjgzYmZkID0gJCgnPGRpdiBpZD0iaHRtbF84ZmU1ZmNmNDc1OTg0YmJlOTJmYzMwZjY1NDI4M2JmZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VudHJhbCBCYXkgU3RyZWV0IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTNiM2FkODViNTgxNDA4Mzg5ZWY4NTkyNmY1YzllODkuc2V0Q29udGVudChodG1sXzhmZTVmY2Y0NzU5ODRiYmU5MmZjMzBmNjU0MjgzYmZkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y0MmQ3MjZjNTUxMjQ3ODRiYjQxYjM2N2JjZGE0ZjU2LmJpbmRQb3B1cChwb3B1cF9hM2IzYWQ4NWI1ODE0MDgzODllZjg1OTI2ZjVjOWU4OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNmNkMWJlNWM5NDQ0OGQ2YTI2MGQzNTNmODJmY2YxYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZGQ1ZDYzMDIzYjI0ZjE4OGRlMTZjNGY5MTIxNzA1MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lMmQ4YTM2ZTNiMGY0OGNjYjAyZWU2ODIwMjI4MGJlYiA9ICQoJzxkaXYgaWQ9Imh0bWxfZTJkOGEzNmUzYjBmNDhjY2IwMmVlNjgyMDIyODBiZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2RkNWQ2MzAyM2IyNGYxODhkZTE2YzRmOTEyMTcwNTMuc2V0Q29udGVudChodG1sX2UyZDhhMzZlM2IwZjQ4Y2NiMDJlZTY4MjAyMjgwYmViKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q2Y2QxYmU1Yzk0NDQ4ZDZhMjYwZDM1M2Y4MmZjZjFjLmJpbmRQb3B1cChwb3B1cF9jZGQ1ZDYzMDIzYjI0ZjE4OGRlMTZjNGY5MTIxNzA1Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80OTZjZGY5NDA0NDg0Zjc2YTFiOTk2M2U3YmVjODEyZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcyNTYwODJmNWE1NDRiZGNhMDdjMjU3NjMyYjUyMGM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFjMGQ2MmU1NTVkYTQ0YjVhNjUwMTRjZGE4YjEzMGNjID0gJCgnPGRpdiBpZD0iaHRtbF8xYzBkNjJlNTU1ZGE0NGI1YTY1MDE0Y2RhOGIxMzBjYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaG1vbmQsIEFkZWxhaWRlLCBLaW5nIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzI1NjA4MmY1YTU0NGJkY2EwN2MyNTc2MzJiNTIwYzcuc2V0Q29udGVudChodG1sXzFjMGQ2MmU1NTVkYTQ0YjVhNjUwMTRjZGE4YjEzMGNjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ5NmNkZjk0MDQ0ODRmNzZhMWI5OTYzZTdiZWM4MTJmLmJpbmRQb3B1cChwb3B1cF83MjU2MDgyZjVhNTQ0YmRjYTA3YzI1NzYzMmI1MjBjNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZjM2NDFkNGNlZTU0ZjdhOTU2MDMzNmQ3MzRkZTFlZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA2ODc4YzE2YTdlMjQxZTE4NWU5Zjg3OWE0Y2I4MGM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBiMDJkOTZiZjlkNzRlYWZhY2Q0ZTQ4NmI1OGViN2Y3ID0gJCgnPGRpdiBpZD0iaHRtbF8wYjAyZDk2YmY5ZDc0ZWFmYWNkNGU0ODZiNThlYjdmNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIFVuaW9uIFN0YXRpb24sIFRvcm9udG8gSXNsYW5kcyBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA2ODc4YzE2YTdlMjQxZTE4NWU5Zjg3OWE0Y2I4MGM2LnNldENvbnRlbnQoaHRtbF8wYjAyZDk2YmY5ZDc0ZWFmYWNkNGU0ODZiNThlYjdmNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZjM2NDFkNGNlZTU0ZjdhOTU2MDMzNmQ3MzRkZTFlZS5iaW5kUG9wdXAocG9wdXBfMDY4NzhjMTZhN2UyNDFlMTg1ZTlmODc5YTRjYjgwYzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmFmZjhlOTU3NDk4NGJjM2FlZmI4MjE1NDVkMjY4YWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNzBhMGNiNDA0NjQ0MmU4ODM0ZmFjM2YyNGNhZDJhZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81Yzc5NmY3NzY4NmI0NTQzYTk5MWE4MjZjNTNmYWY2NCA9ICQoJzxkaXYgaWQ9Imh0bWxfNWM3OTZmNzc2ODZiNDU0M2E5OTFhODI2YzUzZmFmNjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRvcm9udG8gRG9taW5pb24gQ2VudHJlLCBEZXNpZ24gRXhjaGFuZ2UgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNzBhMGNiNDA0NjQ0MmU4ODM0ZmFjM2YyNGNhZDJhZC5zZXRDb250ZW50KGh0bWxfNWM3OTZmNzc2ODZiNDU0M2E5OTFhODI2YzUzZmFmNjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmFmZjhlOTU3NDk4NGJjM2FlZmI4MjE1NDVkMjY4YWIuYmluZFBvcHVwKHBvcHVwX2E3MGEwY2I0MDQ2NDQyZTg4MzRmYWMzZjI0Y2FkMmFkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcwNDYyYWI0ZTdiNjQwYWE4YzE2ZTg3MmIwZDM2MjVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzNlMDk0MWNlM2I5NGVkN2JhZjdlZWI4YzE4ZDA5ZWUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGU3MTY2NzRjNWI2NGE0MDhkYjcwYmZiNzA0YTI4NzEgPSAkKCc8ZGl2IGlkPSJodG1sXzBlNzE2Njc0YzViNjRhNDA4ZGI3MGJmYjcwNGEyODcxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21tZXJjZSBDb3VydCwgVmljdG9yaWEgSG90ZWwgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jM2UwOTQxY2UzYjk0ZWQ3YmFmN2VlYjhjMThkMDllZS5zZXRDb250ZW50KGh0bWxfMGU3MTY2NzRjNWI2NGE0MDhkYjcwYmZiNzA0YTI4NzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzA0NjJhYjRlN2I2NDBhYThjMTZlODcyYjBkMzYyNWEuYmluZFBvcHVwKHBvcHVwX2MzZTA5NDFjZTNiOTRlZDdiYWY3ZWViOGMxOGQwOWVlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlkZGVhN2M3ZGEyYzRmMGY4MmJjMTBhMjRmMTA5ODJjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kN2IyOTA0Nzc2NDA0NzRjOWYyNTUwOTA5OTZiMmNlMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZjE5YzNmM2ViZjc0ZjE0OGZlOGE4OWM4NWZlODUyYyA9ICQoJzxkaXYgaWQ9Imh0bWxfOGYxOWMzZjNlYmY3NGYxNDhmZThhODljODVmZTg1MmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgVG9yb250bywgSGFyYm9yZCBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q3YjI5MDQ3NzY0MDQ3NGM5ZjI1NTA5MDk5NmIyY2UyLnNldENvbnRlbnQoaHRtbF84ZjE5YzNmM2ViZjc0ZjE0OGZlOGE4OWM4NWZlODUyYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZGRlYTdjN2RhMmM0ZjBmODJiYzEwYTI0ZjEwOTgyYy5iaW5kUG9wdXAocG9wdXBfZDdiMjkwNDc3NjQwNDc0YzlmMjU1MDkwOTk2YjJjZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGM1YWIwNTY5NjI4NDhkNzgyODFlMzBiNDA1M2YzNzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc2ZjRjMDU0ZjgyYTRhMTBiY2Q0Y2Q4NDJjZTIxMzEzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1NDM3NjIxYTUxOTRmYzc5YmNlMGM2YjVkYTAyYmNjID0gJCgnPGRpdiBpZD0iaHRtbF9hNTQzNzYyMWE1MTk0ZmM3OWJjZTBjNmI1ZGEwMmJjYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2Vuc2luZ3RvbiBNYXJrZXQsIENoaW5hdG93biwgR3JhbmdlIFBhcmsgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83NmY0YzA1NGY4MmE0YTEwYmNkNGNkODQyY2UyMTMxMy5zZXRDb250ZW50KGh0bWxfYTU0Mzc2MjFhNTE5NGZjNzliY2UwYzZiNWRhMDJiY2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGM1YWIwNTY5NjI4NDhkNzgyODFlMzBiNDA1M2YzNzIuYmluZFBvcHVwKHBvcHVwXzc2ZjRjMDU0ZjgyYTRhMTBiY2Q0Y2Q4NDJjZTIxMzEzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM4ZDcyN2JhNzg1ODRlNjdhYTIyZmFkYTcxNmQyN2U3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZjRmMTRkNzAwMTU0ODc3OWFiN2Q3OTI0ZWZmNmJkNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85N2IzNjI2ZDk3NzQ0ZDRhYWJlYjc0ZmY4MDMxYTlkNCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTdiMzYyNmQ5Nzc0NGQ0YWFiZWI3NGZmODAzMWE5ZDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNOIFRvd2VyLCBLaW5nIGFuZCBTcGFkaW5hLCBSYWlsd2F5IExhbmRzLCBIYXJib3VyZnJvbnQgV2VzdCwgQmF0aHVyc3QgUXVheSwgU291dGggTmlhZ2FyYSwgSXNsYW5kIGFpcnBvcnQgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kZjRmMTRkNzAwMTU0ODc3OWFiN2Q3OTI0ZWZmNmJkNi5zZXRDb250ZW50KGh0bWxfOTdiMzYyNmQ5Nzc0NGQ0YWFiZWI3NGZmODAzMWE5ZDQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzhkNzI3YmE3ODU4NGU2N2FhMjJmYWRhNzE2ZDI3ZTcuYmluZFBvcHVwKHBvcHVwX2RmNGYxNGQ3MDAxNTQ4Nzc5YWI3ZDc5MjRlZmY2YmQ2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EwYzgxNzM5YTQwNzQ4ZmNhNDU1YzZkYTEzYWI3OWVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzgwMDBmZiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiM4MDAwZmYiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODllMzViNDIzMDhjNGRjNDkzZjAwZjc4NWE3ZjJmZTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzhiNzEyOWI2ZmM4NDFiMWEyYTRhNWJkOWFjNTc3YTYgPSAkKCc8ZGl2IGlkPSJodG1sXzM4YjcxMjliNmZjODQxYjFhMmE0YTViZDlhYzU3N2E2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3NlZGFsZSBDbHVzdGVyIDE8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg5ZTM1YjQyMzA4YzRkYzQ5M2YwMGY3ODVhN2YyZmU4LnNldENvbnRlbnQoaHRtbF8zOGI3MTI5YjZmYzg0MWIxYTJhNGE1YmQ5YWM1NzdhNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMGM4MTczOWE0MDc0OGZjYTQ1NWM2ZGExM2FiNzllYS5iaW5kUG9wdXAocG9wdXBfODllMzViNDIzMDhjNGRjNDkzZjAwZjc4NWE3ZjJmZTgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTNiYjUxODhmMDQyNGMzMzk0NzAzMDU2OWY2OTkzMGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hMDg4NjcxNDIzNzA0ZjgzYjBiZjI4YmM3YzJkOTBjYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MzlhZTA2MDJmMzE0NTRlOGFmZjg5MTk4MzdhMDNiMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNDM5YWUwNjAyZjMxNDU0ZThhZmY4OTE5ODM3YTAzYjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIENsdXN0ZXIgNDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTA4ODY3MTQyMzcwNGY4M2IwYmYyOGJjN2MyZDkwY2Iuc2V0Q29udGVudChodG1sXzQzOWFlMDYwMmYzMTQ1NGU4YWZmODkxOTgzN2EwM2IxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEzYmI1MTg4ZjA0MjRjMzM5NDcwMzA1NjlmNjk5MzBhLmJpbmRQb3B1cChwb3B1cF9hMDg4NjcxNDIzNzA0ZjgzYjBiZjI4YmM3YzJkOTBjYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84YjA3OTgxNDgwMzk0N2MyODc0ZDIyNDhiN2UyZTc2ZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmZiMzYwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmYjM2MCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82MTIwZjc5MjBjYmQ0MGNhYTRlNDU1Mzg2Nzg0YTI0OCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NjYyYzg3ZDE3NWQ0ZGE0ODVmYjBlYTU0OGNjYzlmNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MzlkNWFhM2VlZWY0N2EzYTViOWEzYWZhMmQ3ZTdlNSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTM5ZDVhYTNlZWVmNDdhM2E1YjlhM2FmYTJkN2U3ZTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBDYWJiYWdldG93biBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ2NjJjODdkMTc1ZDRkYTQ4NWZiMGVhNTQ4Y2NjOWY2LnNldENvbnRlbnQoaHRtbF85MzlkNWFhM2VlZWY0N2EzYTViOWEzYWZhMmQ3ZTdlNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YjA3OTgxNDgwMzk0N2MyODc0ZDIyNDhiN2UyZTc2ZC5iaW5kUG9wdXAocG9wdXBfNDY2MmM4N2QxNzVkNGRhNDg1ZmIwZWE1NDhjY2M5ZjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTUwYzhhMWE0OGI0NGYwODgzYWY2MjA4ZmIwMzY4ZDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZmIzNjAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmZiMzYwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzYxMjBmNzkyMGNiZDQwY2FhNGU0NTUzODY3ODRhMjQ4KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhlNjNmY2NhYzMxNTRkYTU4M2Y2ZDVkNzk4Zjk1NWRiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1YzA3YmU1NGZlNDQxOTg4MGE5ODAyY2ZkNzY1Y2NlID0gJCgnPGRpdiBpZD0iaHRtbF9hNWMwN2JlNTRmZTQ0MTk4ODBhOTgwMmNmZDc2NWNjZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIFVuZGVyZ3JvdW5kIGNpdHkgQ2x1c3RlciA0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZTYzZmNjYWMzMTU0ZGE1ODNmNmQ1ZDc5OGY5NTVkYi5zZXRDb250ZW50KGh0bWxfYTVjMDdiZTU0ZmU0NDE5ODgwYTk4MDJjZmQ3NjVjY2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTUwYzhhMWE0OGI0NGYwODgzYWY2MjA4ZmIwMzY4ZDQuYmluZFBvcHVwKHBvcHVwXzhlNjNmY2NhYzMxNTRkYTU4M2Y2ZDVkNzk4Zjk1NWRiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlmYjM4NjNiZDhkNjQ1YTFhMzQ2ZTlkMzEwMzAxZDkwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjY1ODU5OSwtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjEyMGY3OTIwY2JkNDBjYWE0ZTQ1NTM4Njc4NGEyNDgpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzFiYzUwYWIwMWVkNGEyOGJkZTc4NjhhYmQwNTc0NjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzEyMmZkNDE4MDA4NGQ5Zjk3NTVmOTczN2Y3NjY3ZjQgPSAkKCc8ZGl2IGlkPSJodG1sX2MxMjJmZDQxODAwODRkOWY5NzU1Zjk3MzdmNzY2N2Y0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHVyY2ggYW5kIFdlbGxlc2xleSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMxYmM1MGFiMDFlZDRhMjhiZGU3ODY4YWJkMDU3NDY4LnNldENvbnRlbnQoaHRtbF9jMTIyZmQ0MTgwMDg0ZDlmOTc1NWY5NzM3Zjc2NjdmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZmIzODYzYmQ4ZDY0NWExYTM0NmU5ZDMxMDMwMWQ5MC5iaW5kUG9wdXAocG9wdXBfMzFiYzUwYWIwMWVkNGEyOGJkZTc4NjhhYmQwNTc0NjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCjwvc2NyaXB0Pg== onload="this.contentDocument.open();this.contentDocument.write(atob(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



## Step Final - Cluster Tables

### Cluster 1


```python
## Cluster 1 in downtown_merged
downtown_merged.loc[downtown_merged['Cluster Labels'] == 0, 
                     downtown_merged.columns[[0] + list(range(5, downtown_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PostalCode</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M5A</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Beer Store</td>
      <td>Mexican Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M7A</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Portuguese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Smoothie Shop</td>
      <td>Beer Bar</td>
      <td>Sandwich Place</td>
      <td>Distribution Center</td>
      <td>Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M5G</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Salad Place</td>
      <td>Department Store</td>
      <td>Japanese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M5J</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Scenic Lookout</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Plaza</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 2


```python
## Cluster 2 in downtown_merged
downtown_merged.loc[downtown_merged['Cluster Labels'] == 0, 
                     downtown_merged.columns[[1] + list(range(5, downtown_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Borough</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Downtown Toronto</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Beer Store</td>
      <td>Mexican Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Downtown Toronto</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Portuguese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Smoothie Shop</td>
      <td>Beer Bar</td>
      <td>Sandwich Place</td>
      <td>Distribution Center</td>
      <td>Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Downtown Toronto</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Salad Place</td>
      <td>Department Store</td>
      <td>Japanese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Downtown Toronto</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Scenic Lookout</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Plaza</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 3


```python
## Cluster 3 in downtown_merged
downtown_merged.loc[downtown_merged['Cluster Labels'] == 0, 
                     downtown_merged.columns[[2] + list(range(5, downtown_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighbourhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Beer Store</td>
      <td>Mexican Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Portuguese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Smoothie Shop</td>
      <td>Beer Bar</td>
      <td>Sandwich Place</td>
      <td>Distribution Center</td>
      <td>Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Central Bay Street</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Salad Place</td>
      <td>Department Store</td>
      <td>Japanese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Scenic Lookout</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Plaza</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 4


```python
## Cluster 4 in downtown_merged
downtown_merged.loc[downtown_merged['Cluster Labels'] == 0, 
                     downtown_merged.columns[[3] + list(range(5, downtown_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Latitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43.654260</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Beer Store</td>
      <td>Mexican Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>43.662301</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Portuguese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Smoothie Shop</td>
      <td>Beer Bar</td>
      <td>Sandwich Place</td>
      <td>Distribution Center</td>
      <td>Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>5</th>
      <td>43.657952</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Salad Place</td>
      <td>Department Store</td>
      <td>Japanese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>43.640816</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Scenic Lookout</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Plaza</td>
    </tr>
  </tbody>
</table>
</div>



### Cluster 5


```python
## Cluster 5 in downtown_merged
downtown_merged.loc[downtown_merged['Cluster Labels'] == 0, 
                     downtown_merged.columns[[4] + list(range(5, downtown_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-79.360636</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Pub</td>
      <td>Park</td>
      <td>Breakfast Spot</td>
      <td>Café</td>
      <td>Theater</td>
      <td>Beer Store</td>
      <td>Mexican Restaurant</td>
      <td>Spa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-79.389494</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Portuguese Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Smoothie Shop</td>
      <td>Beer Bar</td>
      <td>Sandwich Place</td>
      <td>Distribution Center</td>
      <td>Restaurant</td>
      <td>Diner</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-79.387383</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Italian Restaurant</td>
      <td>Salad Place</td>
      <td>Department Store</td>
      <td>Japanese Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Burger Joint</td>
      <td>Thai Restaurant</td>
    </tr>
    <tr>
      <th>8</th>
      <td>-79.381752</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Aquarium</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Fried Chicken Joint</td>
      <td>Scenic Lookout</td>
      <td>Brewery</td>
      <td>Restaurant</td>
      <td>Park</td>
      <td>Plaza</td>
    </tr>
  </tbody>
</table>
</div>



---
This notebook is part of a course on Coursera called Applied Data Science Capstone. This was done by Anderson Braz de Sousa.
