---
title: "predicting solar power at neom"
date: 2019-09-15T11:21:18+03:00
draft: false
---

## This is the jupyter notebook from our JunctionX kaust hackaton (24h forecast), [source is here ](https://github.com/davidrpugh/junctionx-kaust-2019)


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model, ensemble, metrics, model_selection, preprocessing
import joblib
```


```python
%matplotlib inline
```

# Predicting Solar Power Output at NEOM


```python
neom_data = (pd.read_csv("../data/raw/neom-data.csv", parse_dates=[0])
               .rename(columns={"Unnamed: 0": "Timestamp"})
               .set_index("Timestamp", drop=True, inplace=False))
```


```python
neom_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 96432 entries, 2008-01-01 00:00:00 to 2018-12-31 23:00:00
    Data columns (total 12 columns):
    mslp(hPa)          96432 non-null float64
    t2(C)              96432 non-null float64
    td2(C)             96432 non-null float64
    wind_speed(m/s)    96432 non-null float64
    wind_dir(Deg)      96432 non-null float64
    rh(%)              96432 non-null float64
    GHI(W/m2)          96432 non-null float64
    SWDIR(W/m2)        96432 non-null float64
    SWDNI(W/m2)        96432 non-null float64
    SWDIF(W/m2)        96432 non-null float64
    rain(mm)           96432 non-null float64
    AOD                96432 non-null float64
    dtypes: float64(12)
    memory usage: 9.6 MB
    


```python
neom_data.head()
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
      <th>mslp(hPa)</th>
      <th>t2(C)</th>
      <th>td2(C)</th>
      <th>wind_speed(m/s)</th>
      <th>wind_dir(Deg)</th>
      <th>rh(%)</th>
      <th>GHI(W/m2)</th>
      <th>SWDIR(W/m2)</th>
      <th>SWDNI(W/m2)</th>
      <th>SWDIF(W/m2)</th>
      <th>rain(mm)</th>
      <th>AOD</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <td>2008-01-01 00:00:00</td>
      <td>1012.751</td>
      <td>14.887</td>
      <td>2.606</td>
      <td>2.669</td>
      <td>105.078</td>
      <td>43.686</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2008-01-01 01:00:00</td>
      <td>1012.917</td>
      <td>14.429</td>
      <td>3.363</td>
      <td>2.667</td>
      <td>106.699</td>
      <td>47.442</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2008-01-01 02:00:00</td>
      <td>1012.966</td>
      <td>14.580</td>
      <td>3.778</td>
      <td>3.341</td>
      <td>112.426</td>
      <td>48.357</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2008-01-01 03:00:00</td>
      <td>1013.247</td>
      <td>14.390</td>
      <td>3.507</td>
      <td>3.141</td>
      <td>102.371</td>
      <td>48.125</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2008-01-01 04:00:00</td>
      <td>1013.083</td>
      <td>14.388</td>
      <td>3.869</td>
      <td>3.607</td>
      <td>111.300</td>
      <td>49.295</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
  </tbody>
</table>
</div>




```python
neom_data.tail()
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
      <th>mslp(hPa)</th>
      <th>t2(C)</th>
      <th>td2(C)</th>
      <th>wind_speed(m/s)</th>
      <th>wind_dir(Deg)</th>
      <th>rh(%)</th>
      <th>GHI(W/m2)</th>
      <th>SWDIR(W/m2)</th>
      <th>SWDNI(W/m2)</th>
      <th>SWDIF(W/m2)</th>
      <th>rain(mm)</th>
      <th>AOD</th>
    </tr>
    <tr>
      <th>Timestamp</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
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
      <td>2018-12-31 19:00:00</td>
      <td>1019.779</td>
      <td>14.653</td>
      <td>4.380</td>
      <td>3.587</td>
      <td>25.919</td>
      <td>50.340</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2018-12-31 20:00:00</td>
      <td>1019.578</td>
      <td>13.965</td>
      <td>2.853</td>
      <td>2.836</td>
      <td>35.203</td>
      <td>47.381</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2018-12-31 21:00:00</td>
      <td>1019.172</td>
      <td>13.624</td>
      <td>1.923</td>
      <td>1.922</td>
      <td>85.974</td>
      <td>45.275</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2018-12-31 22:00:00</td>
      <td>1018.610</td>
      <td>13.918</td>
      <td>1.512</td>
      <td>2.512</td>
      <td>103.656</td>
      <td>43.211</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
    <tr>
      <td>2018-12-31 23:00:00</td>
      <td>1018.611</td>
      <td>13.442</td>
      <td>0.733</td>
      <td>3.146</td>
      <td>91.084</td>
      <td>41.836</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.098</td>
    </tr>
  </tbody>
</table>
</div>




```python
neom_data.describe()
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
      <th>mslp(hPa)</th>
      <th>t2(C)</th>
      <th>td2(C)</th>
      <th>wind_speed(m/s)</th>
      <th>wind_dir(Deg)</th>
      <th>rh(%)</th>
      <th>GHI(W/m2)</th>
      <th>SWDIR(W/m2)</th>
      <th>SWDNI(W/m2)</th>
      <th>SWDIF(W/m2)</th>
      <th>rain(mm)</th>
      <th>AOD</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
      <td>96432.000000</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>1010.110794</td>
      <td>24.896298</td>
      <td>11.045605</td>
      <td>3.991582</td>
      <td>164.200525</td>
      <td>46.168410</td>
      <td>274.757261</td>
      <td>211.082623</td>
      <td>331.746291</td>
      <td>63.674490</td>
      <td>0.009041</td>
      <td>0.098086</td>
    </tr>
    <tr>
      <td>std</td>
      <td>5.613583</td>
      <td>6.382410</td>
      <td>7.153472</td>
      <td>2.485326</td>
      <td>102.793404</td>
      <td>17.874776</td>
      <td>355.287896</td>
      <td>296.287340</td>
      <td>390.765915</td>
      <td>91.856426</td>
      <td>0.173081</td>
      <td>0.000805</td>
    </tr>
    <tr>
      <td>min</td>
      <td>996.378000</td>
      <td>4.571000</td>
      <td>-22.946000</td>
      <td>0.076000</td>
      <td>0.672000</td>
      <td>5.708000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-0.037000</td>
      <td>0.096000</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>1005.539750</td>
      <td>20.221000</td>
      <td>5.889750</td>
      <td>2.152000</td>
      <td>62.935500</td>
      <td>32.173000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.098000</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>1010.050000</td>
      <td>25.421000</td>
      <td>11.324500</td>
      <td>3.437000</td>
      <td>149.692000</td>
      <td>44.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.098000</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>1014.316000</td>
      <td>29.466000</td>
      <td>16.581250</td>
      <td>5.342000</td>
      <td>265.977750</td>
      <td>58.859000</td>
      <td>579.205250</td>
      <td>429.275500</td>
      <td>788.745750</td>
      <td>121.765250</td>
      <td>0.000000</td>
      <td>0.099000</td>
    </tr>
    <tr>
      <td>max</td>
      <td>1029.022000</td>
      <td>44.186000</td>
      <td>27.196000</td>
      <td>16.716000</td>
      <td>359.620000</td>
      <td>99.929000</td>
      <td>1103.190000</td>
      <td>954.562000</td>
      <td>989.816000</td>
      <td>856.685000</td>
      <td>14.038000</td>
      <td>0.100000</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = neom_data.hist(bins=50, figsize=(20,15))
```


![png](/img/predicting_files/predicting_8_0.png)



```python
_hour = (neom_data.index
                  .hour
                  .rename("hour"))

hourly_averages = (neom_data.groupby(_hour)
                            .mean())

fig, ax = plt.subplots(1, 1)
_targets = ["GHI(W/m2)", "SWDIR(W/m2)", "SWDNI(W/m2)", "SWDIF(W/m2)"]
(hourly_averages.loc[:, _targets]
                .plot(ax=ax))
_ = ax.set_ylabel(r"$W/m^2$", rotation="horizontal")
```


![png](/img/predicting_files/predicting_9_0.png)



```python
months = (neom_data.index
                   .month
                   .rename("month"))
hours = (neom_data.index
                  .hour
                  .rename("hour"))

hourly_averages_by_month = (neom_data.groupby([months, hours])
                                     .mean())
```


```python
fig, axes = plt.subplots(2, 6, sharex=True, sharey=True, figsize=(12, 6))

for month in months.unique():
    if month <= 6:
        (hourly_averages_by_month.loc[month, _targets]
                                 .plot(ax=axes[0, month - 1], legend=False))
        _ = axes[0, month - 1].set_title(month)
    else:
        (hourly_averages_by_month.loc[month, _targets]
                                 .plot(ax=axes[1, month - 7], legend=False))
        _ = axes[1, month - 7].set_title(month)
    
    if month - 1 == 0: 
        _ = axes[0, 0].set_ylabel(r"$W/m^2$")
    if month - 7 == 0: 
        _ = axes[1, 0].set_ylabel(r"$W/m^2$")
   
```


![png](/img/predicting_files/predicting_11_0.png)


# Feature Engineering


```python
_dropped_cols = ["SWDIR(W/m2)", "SWDNI(W/m2)", "SWDIF(W/m2)"]

_year = (neom_data.index
                  .year)
_month = (neom_data.index
                   .month)
_day = (neom_data.index
                 .dayofyear)
_hour = (neom_data.index
                  .hour)

features = (neom_data.drop(_dropped_cols, axis=1, inplace=False)
                     .assign(year=_year, month=_month, day=_day, hour=_hour)
                     .groupby(["year", "month", "day", "hour"])
                     .mean()
                     .unstack(level=["hour"])
                     .reset_index(inplace=False)
                     .sort_index(axis=1)
                     .drop("year", axis=1, inplace=False))
```


```python
# want to predict the next 24 hours of "solar power"
efficiency_factor = 0.5

# square meters of solar cells required to generate 20 GW (231000 m2 will generate 7mW)
m2_of_solar_cells_required = 660000

target = (features.loc[:, ["GHI(W/m2)"]]
                  .mul(efficiency_factor)
                  .shift(-1)
                  .rename(columns={"GHI(W/m2)": "target(W/m2)"}))
```


```python
input_data = (features.join(target)
                      .dropna(how="any", inplace=False)
                      .sort_index(axis=1))
```


```python
input_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="10" halign="left">AOD</th>
      <th>...</th>
      <th colspan="10" halign="left">wind_speed(m/s)</th>
    </tr>
    <tr>
      <th>hour</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>...</td>
      <td>1.775</td>
      <td>0.800</td>
      <td>0.778</td>
      <td>0.993</td>
      <td>1.575</td>
      <td>1.606</td>
      <td>2.079</td>
      <td>2.887</td>
      <td>3.162</td>
      <td>3.315</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>2.255</td>
      <td>1.494</td>
      <td>0.595</td>
      <td>0.479</td>
      <td>2.143</td>
      <td>3.394</td>
      <td>3.208</td>
      <td>2.805</td>
      <td>3.436</td>
      <td>4.196</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>3.924</td>
      <td>1.952</td>
      <td>1.885</td>
      <td>1.834</td>
      <td>3.728</td>
      <td>5.187</td>
      <td>5.647</td>
      <td>6.324</td>
      <td>7.722</td>
      <td>8.740</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>5.802</td>
      <td>4.670</td>
      <td>3.535</td>
      <td>4.811</td>
      <td>5.417</td>
      <td>5.956</td>
      <td>7.445</td>
      <td>8.008</td>
      <td>8.297</td>
      <td>8.363</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>6.825</td>
      <td>5.146</td>
      <td>4.840</td>
      <td>3.051</td>
      <td>7.151</td>
      <td>4.439</td>
      <td>3.869</td>
      <td>6.069</td>
      <td>9.337</td>
      <td>10.455</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4012</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>0.098</td>
      <td>...</td>
      <td>2.152</td>
      <td>0.939</td>
      <td>0.578</td>
      <td>0.556</td>
      <td>0.548</td>
      <td>1.289</td>
      <td>2.356</td>
      <td>3.038</td>
      <td>3.378</td>
      <td>4.046</td>
    </tr>
    <tr>
      <td>4013</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>7.200</td>
      <td>6.153</td>
      <td>6.734</td>
      <td>4.120</td>
      <td>4.700</td>
      <td>5.639</td>
      <td>4.562</td>
      <td>3.876</td>
      <td>3.941</td>
      <td>4.160</td>
    </tr>
    <tr>
      <td>4014</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>3.531</td>
      <td>3.276</td>
      <td>3.971</td>
      <td>3.409</td>
      <td>2.173</td>
      <td>3.252</td>
      <td>4.387</td>
      <td>4.496</td>
      <td>4.268</td>
      <td>3.572</td>
    </tr>
    <tr>
      <td>4015</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>4.930</td>
      <td>4.024</td>
      <td>3.824</td>
      <td>3.115</td>
      <td>1.401</td>
      <td>2.460</td>
      <td>3.788</td>
      <td>4.351</td>
      <td>4.708</td>
      <td>4.673</td>
    </tr>
    <tr>
      <td>4016</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>0.097</td>
      <td>...</td>
      <td>6.060</td>
      <td>5.376</td>
      <td>4.654</td>
      <td>3.102</td>
      <td>3.931</td>
      <td>5.463</td>
      <td>5.687</td>
      <td>4.491</td>
      <td>5.557</td>
      <td>6.213</td>
    </tr>
  </tbody>
</table>
<p>4017 rows × 242 columns</p>
</div>




```python
input_data.to_csv("../results/day-ahead/input-data.csv")
```

# Train, Validation, Test Split


```python
# use first eight years for training data...
training_data = input_data.loc[:8 * 365]

# ...next two years for validation data...
validation_data = input_data.loc[8 * 365 + 1:10 * 365 + 1]

# ...and final year for testing data!
testing_data = input_data.loc[10 * 365 + 2:]
```


```python
training_data.shape
```




    (2921, 242)




```python
validation_data.shape
```




    (731, 242)




```python
testing_data.shape
```




    (365, 242)



# Preprocessing the training and validation data


```python
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    _numeric_features = ["GHI(W/m2)",
                         "mslp(hPa)",
                         "rain(mm)",
                         "rh(%)",
                         "t2(C)",
                         "td2(C)",
                         "wind_dir(Deg)",
                         "wind_speed(m/s)"]

    _ordinal_features = ["AOD",
                         "day",
                         "month",
                         "year"]

    standard_scalar = preprocessing.StandardScaler()
    Z0 = standard_scalar.fit_transform(df.loc[:, _numeric_features])
    
    ordinal_encoder = preprocessing.OrdinalEncoder()
    Z1 = ordinal_encoder.fit_transform(df.loc[:, _ordinal_features])
    transformed_features = np.hstack((Z0, Z1))
    
    return transformed_features


```


```python
training_features = training_data.drop("target(W/m2)", axis=1, inplace=False)
training_target = training_data.loc[:, ["target(W/m2)"]]
transformed_training_features = preprocess_features(training_features)

validation_features = validation_data.drop("target(W/m2)", axis=1, inplace=False)
validation_target = validation_data.loc[:, ["target(W/m2)"]]
transformed_validation_features = preprocess_features(validation_features)
```

# Find a few models that seem to work well

## Linear Regression


```python
# training a liner regression model
linear_regression = linear_model.LinearRegression()
linear_regression.fit(transformed_training_features, training_target)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
# measure training error
_predictions = linear_regression.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))
```




    16.18214503421171




```python
# measure validation error
_predictions = linear_regression.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))
```




    18.628289130716887




```python
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = linear_regression.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
```




    Text(0.5, 0, 'Hour')




![png](/img/predicting_files/predicting_31_1.png)


Linear regression is not bad but we an do better!

## MultiTask ElasticNet Regression


```python
# training a multi-task elastic net model
_prng = np.random.RandomState(42)
elastic_net = linear_model.MultiTaskElasticNet(random_state=_prng)
elastic_net.fit(transformed_training_features, training_target)
```




    MultiTaskElasticNet(alpha=1.0, copy_X=True, fit_intercept=True, l1_ratio=0.5,
                        max_iter=1000, normalize=False,
                        random_state=<mtrand.RandomState object at 0x1a386672d0>,
                        selection='cyclic', tol=0.0001, warm_start=False)




```python
# measure training error
_predictions = elastic_net.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))
```




    18.73221160297961




```python
# measure validation error
_predictions = elastic_net.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))
```




    17.864005311687215




```python
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = elastic_net.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
```




    Text(0.5, 0, 'Hour')




![png](/img/predicting_files/predicting_37_1.png)


ElasticNet is underfitting.

## MultiTask Lasso Regression


```python
# training a multi-task lasso model
_prng = np.random.RandomState(42)
lasso_regression = linear_model.MultiTaskLasso(random_state=_prng)
lasso_regression.fit(transformed_training_features, training_target)
```




    MultiTaskLasso(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=1000,
                   normalize=False,
                   random_state=<mtrand.RandomState object at 0x1a649d5870>,
                   selection='cyclic', tol=0.0001, warm_start=False)




```python
# measure training error
_predictions = lasso_regression.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))
```




    17.375320291371146




```python
# measure validation error
_predictions = lasso_regression.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))
```




    16.14489387119325




```python
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = lasso_regression.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
```




    Text(0.5, 0, 'Hour')




![png](/img/predicting_files/predicting_43_1.png)


Lasso Regression is underfitting.

## Random Forest Regression


```python
_prng = np.random.RandomState(42)
random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=100, random_state=_prng, n_jobs=2)
random_forest_regressor.fit(transformed_training_features, training_target)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=2,
                          oob_score=False,
                          random_state=<mtrand.RandomState object at 0x1284a3558>,
                          verbose=0, warm_start=False)




```python
# measure training error
_predictions = random_forest_regressor.predict(transformed_training_features)
np.sqrt(metrics.mean_squared_error(training_target, _predictions))
```




    6.540426384057337




```python
# measure validation error
_predictions = random_forest_regressor.predict(transformed_validation_features)
np.sqrt(metrics.mean_squared_error(validation_target, _predictions))
```




    17.10706983980618




```python
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = random_forest_regressor.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
```




    Text(0.5, 0, 'Hour')




![png](/img/predicting_files/predicting_49_1.png)


Random Forest with default parameters is over-fitting and needs to be regularized.

# Tuning hyper-parameters


```python
from scipy import stats
```

## MultiTask ElasticNet Regression


```python
_prng = np.random.RandomState(42)

_param_distributions = {
    "l1_ratio": stats.uniform(),
    "alpha": stats.lognorm(s=1),
}

elastic_net_randomized_search = model_selection.RandomizedSearchCV(
    elastic_net,
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

elastic_net_randomized_search.fit(transformed_training_features, training_target)
```

    Fitting 8 folds for each of 10 candidates, totalling 80 fits
    

    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    7.7s
    [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:   11.2s
    [Parallel(n_jobs=2)]: Done   9 tasks      | elapsed:   23.6s
    [Parallel(n_jobs=2)]: Done  14 tasks      | elapsed:   36.4s
    [Parallel(n_jobs=2)]: Done  21 tasks      | elapsed:  1.4min
    [Parallel(n_jobs=2)]: Done  28 tasks      | elapsed:  1.8min
    [Parallel(n_jobs=2)]: Done  37 tasks      | elapsed:  2.0min
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  2.1min
    [Parallel(n_jobs=2)]: Done  57 tasks      | elapsed:  3.0min
    [Parallel(n_jobs=2)]: Done  68 tasks      | elapsed:  3.5min
    [Parallel(n_jobs=2)]: Done  80 out of  80 | elapsed:  6.0min finished
    




    RandomizedSearchCV(cv=8, error_score='raise-deprecating',
                       estimator=MultiTaskElasticNet(alpha=1.0, copy_X=True,
                                                     fit_intercept=True,
                                                     l1_ratio=0.5, max_iter=1000,
                                                     normalize=False,
                                                     random_state=<mtrand.RandomState object at 0x1a386672d0>,
                                                     selection='cyclic', tol=0.0001,
                                                     warm_start=False),
                       iid='warn', n_iter=10, n_jobs=2,
                       param_distributions={'alpha': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a7b2845c0>,
                                            'l1_ratio': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a7b26f710>},
                       pre_dispatch='2*n_jobs',
                       random_state=<mtrand.RandomState object at 0x12808e318>,
                       refit=True, return_train_score=False,
                       scoring='neg_mean_squared_error', verbose=10)




```python
_ = joblib.dump(elastic_net_randomized_search.best_estimator_,
                "../models/tuned-elasticnet-regression-model.pkl")
```


```python
elastic_net_randomized_search.best_estimator_
```




    MultiTaskElasticNet(alpha=2.154232968599504, copy_X=True, fit_intercept=True,
                        l1_ratio=0.9699098521619943, max_iter=1000, normalize=False,
                        random_state=<mtrand.RandomState object at 0x1a48e31558>,
                        selection='cyclic', tol=0.0001, warm_start=False)




```python
(-elastic_net_randomized_search.best_score_)**0.5
```




    18.355092813714375




```python
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = elastic_net_randomized_search.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2017")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
```




    Text(0.5, 0, 'Hour')




![png](/img/predicting_files/predicting_58_1.png)



```python
# user requests forecast for last week of 2015
user_forecast_request = transformed_training_features[-7:, :]
user_forecast_response = elastic_net_randomized_search.predict(user_forecast_request)
actual_values_response = training_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2015")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")
```




    Text(0.5, 0, 'Hours')




![png](/img/predicting_files/predicting_59_1.png)


## MultiTask Lasso Regression


```python
_prng = np.random.RandomState(42)

_param_distributions = {
    "alpha": stats.lognorm(s=1),
}

lasso_regression_randomized_search = model_selection.RandomizedSearchCV(
    lasso_regression,
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

lasso_regression_randomized_search.fit(transformed_training_features, training_target)
```

    Fitting 8 folds for each of 10 candidates, totalling 80 fits
    

    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:    7.0s
    [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:    9.4s
    [Parallel(n_jobs=2)]: Done   9 tasks      | elapsed:   16.5s
    [Parallel(n_jobs=2)]: Done  14 tasks      | elapsed:   22.5s
    [Parallel(n_jobs=2)]: Done  21 tasks      | elapsed:   30.3s
    [Parallel(n_jobs=2)]: Done  28 tasks      | elapsed:   38.4s
    [Parallel(n_jobs=2)]: Done  37 tasks      | elapsed:   48.6s
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  1.0min
    [Parallel(n_jobs=2)]: Done  57 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=2)]: Done  68 tasks      | elapsed:  1.5min
    [Parallel(n_jobs=2)]: Done  80 out of  80 | elapsed:  1.8min finished
    




    RandomizedSearchCV(cv=8, error_score='raise-deprecating',
                       estimator=MultiTaskLasso(alpha=1.0, copy_X=True,
                                                fit_intercept=True, max_iter=1000,
                                                normalize=False,
                                                random_state=<mtrand.RandomState object at 0x1a649d5870>,
                                                selection='cyclic', tol=0.0001,
                                                warm_start=False),
                       iid='warn', n_iter=10, n_jobs=2,
                       param_distributions={'alpha': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a7af2b470>},
                       pre_dispatch='2*n_jobs',
                       random_state=<mtrand.RandomState object at 0x1a69473438>,
                       refit=True, return_train_score=False,
                       scoring='neg_mean_squared_error', verbose=10)




```python
_ = joblib.dump(lasso_regression_randomized_search.best_estimator_,
                "../models/tuned-lasso-regression-model.pkl")
```


```python
(-lasso_regression_randomized_search.best_score_)**0.5
```




    17.956920842745834




```python
# user requests forecast for last week of 2015
user_forecast_request = transformed_training_features[-7:, :]
user_forecast_response = lasso_regression_randomized_search.predict(user_forecast_request)
actual_values_response = training_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2015")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")
```




    Text(0.5, 0, 'Hours')




![png](/img/predicting_files/predicting_64_1.png)


## Random Forest Regressor


```python
_prng = np.random.RandomState(42)

_param_distributions = {
    "n_estimators": stats.geom(p=0.01),
     "min_samples_split": stats.beta(a=1, b=99),
     "min_samples_leaf": stats.beta(a=1, b=999),
}

_cv = model_selection.TimeSeriesSplit(max_train_size=None, n_splits=5)

random_forest_randomized_search = model_selection.RandomizedSearchCV(
    random_forest_regressor,
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

random_forest_randomized_search.fit(transformed_training_features, training_target)
```

    Fitting 8 folds for each of 10 candidates, totalling 80 fits
    

    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:   38.3s
    [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:  1.2min
    [Parallel(n_jobs=2)]: Done   9 tasks      | elapsed:  2.7min
    [Parallel(n_jobs=2)]: Done  14 tasks      | elapsed: 16.6min
    [Parallel(n_jobs=2)]: Done  21 tasks      | elapsed: 18.3min
    [Parallel(n_jobs=2)]: Done  28 tasks      | elapsed: 18.8min
    [Parallel(n_jobs=2)]: Done  37 tasks      | elapsed: 19.2min
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed: 19.7min
    [Parallel(n_jobs=2)]: Done  57 tasks      | elapsed: 20.4min
    [Parallel(n_jobs=2)]: Done  68 tasks      | elapsed: 22.3min
    [Parallel(n_jobs=2)]: Done  80 out of  80 | elapsed: 24.0min finished
    




    RandomizedSearchCV(cv=8, error_score='raise-deprecating',
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100, n_jobs=2,
                                                       oob_score=False,
                                                       random_state=<mt...
                       param_distributions={'min_samples_leaf': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a5ebd65c0>,
                                            'min_samples_split': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a625c2a58>,
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a5ebd49b0>},
                       pre_dispatch='2*n_jobs',
                       random_state=<mtrand.RandomState object at 0x1a3a76c2d0>,
                       refit=True, return_train_score=False,
                       scoring='neg_mean_squared_error', verbose=10)




```python
_ = joblib.dump(random_forest_randomized_search.best_estimator_,
                "../models/tuned-random-forest-regression-model.pkl")
```


```python
random_forest_randomized_search.best_estimator_
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=0.001249828663231378,
                          min_samples_split=0.0019415164208264953,
                          min_weight_fraction_leaf=0.0, n_estimators=75, n_jobs=2,
                          oob_score=False,
                          random_state=<mtrand.RandomState object at 0x125798480>,
                          verbose=0, warm_start=False)




```python
(-random_forest_randomized_search.best_score_)**0.5
```




    17.248545184577317




```python
# user requests forecast for 1 January 2016 which we predict using data from 31 December 2015!
user_forecast_request = transformed_training_features[[-1], :]
user_forecast_response = random_forest_randomized_search.predict(user_forecast_request)[0]
actual_values_response = training_target.values[[-1], :][0]

# this would be rendered in Tableau!
plt.plot(user_forecast_response, label="predicted")
plt.plot(actual_values_response, label="actual")
plt.legend()
plt.title("1 January 2016")
plt.ylabel("Solar Power (W/m2)")
plt.xlabel("Hour")
plt.savefig("../results/img/typical-actual-vs-predicted-solar-power.png")
```


![png](/img/predicting_files/predicting_70_0.png)


# Assess model performance on testing data


```python
testing_features = testing_data.drop("target(W/m2)", axis=1, inplace=False)
testing_target = testing_data.loc[:, ["target(W/m2)"]]
transformed_testing_features = preprocess_features(testing_features)
```


```python
elastic_net_predictions = elastic_net_randomized_search.predict(transformed_testing_features)
np.sqrt(metrics.mean_squared_error(testing_target, elastic_net_predictions))
```




    19.76022744338562




```python
lasso_regression_predictions = lasso_regression_randomized_search.predict(transformed_testing_features)
np.sqrt(metrics.mean_squared_error(testing_target, lasso_regression_predictions))
```




    19.73446980783998




```python
# random forest wins!
random_forest_predictions = random_forest_randomized_search.predict(transformed_testing_features)
np.sqrt(metrics.mean_squared_error(testing_target, random_forest_predictions))
```




    18.977074427410706




```python
# user requests forecast for last week of 2018
user_forecast_request = transformed_testing_features[-7:, :]
user_forecast_response = random_forest_randomized_search.predict(user_forecast_request)
actual_values_response = testing_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2018")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")
```




    Text(0.5, 0, 'Hours')




![png](/img/predicting_files/predicting_76_1.png)



```python
submission = (pd.DataFrame.from_dict({"Timestamp": pd.date_range("2018-01-01", end="2018-12-31 23:00:00", freq="H"),
                                     "Predicted Solar Power (W/m2)": random_forest_predictions.flatten(),
                                     "Actual Solar Power (W/m2)": testing_target.values.flatten()})
                          .set_index("Timestamp", inplace=False))
```


```python
submission.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 8760 entries, 2018-01-01 00:00:00 to 2018-12-31 23:00:00
    Data columns (total 2 columns):
    Predicted Solar Power (W/m2)    8760 non-null float64
    Actual Solar Power (W/m2)       8760 non-null float64
    dtypes: float64(2)
    memory usage: 205.3 KB
    


```python
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
_ = submission.plot(ax=ax)
```


![png](/img/predicting_files/predicting_79_0.png)



```python
submission.to_csv("../results/actual-vs-predicted-values-for-2018.csv", index=True)
```


```python
# combine the training and validtion data
combined_training_features = pd.concat([training_features, validation_features])
transformed_combined_training_features = preprocess_features(combined_training_features)
combined_training_target = pd.concat([training_target, validation_target])

# tune a random forest regressor using CV ro avoid overfitting
_prng = np.random.RandomState(42)

_param_distributions = {
    "n_estimators": stats.geom(p=0.01),
     "min_samples_split": stats.beta(a=1, b=99),
     "min_samples_leaf": stats.beta(a=1, b=999),
}

tuned_random_forest_regressor = model_selection.RandomizedSearchCV(
    ensemble.RandomForestRegressor(n_estimators=100, random_state=_prng),
    param_distributions=_param_distributions,
    scoring="neg_mean_squared_error",
    random_state=_prng,
    n_iter=10,
    cv=8,
    n_jobs=2,
    verbose=10
)

tuned_random_forest_regressor.fit(combined_training_features, combined_training_target)
```

    Fitting 8 folds for each of 10 candidates, totalling 80 fits
    

    [Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=2)]: Done   1 tasks      | elapsed:   59.2s
    [Parallel(n_jobs=2)]: Done   4 tasks      | elapsed:  1.9min
    [Parallel(n_jobs=2)]: Done   9 tasks      | elapsed:  4.2min
    [Parallel(n_jobs=2)]: Done  14 tasks      | elapsed:  5.4min
    [Parallel(n_jobs=2)]: Done  21 tasks      | elapsed:  7.6min
    [Parallel(n_jobs=2)]: Done  28 tasks      | elapsed:  8.4min
    [Parallel(n_jobs=2)]: Done  37 tasks      | elapsed:  9.1min
    [Parallel(n_jobs=2)]: Done  46 tasks      | elapsed:  9.6min
    [Parallel(n_jobs=2)]: Done  57 tasks      | elapsed: 10.7min
    [Parallel(n_jobs=2)]: Done  68 tasks      | elapsed: 13.9min
    [Parallel(n_jobs=2)]: Done  80 out of  80 | elapsed: 16.9min finished
    /Users/pughdr/Research/junctionx-kaust-2019/env/lib/python3.6/site-packages/sklearn/model_selection/_search.py:813: DeprecationWarning: The default of the `iid` parameter will change from True to False in version 0.22 and will be removed in 0.24. This will change numeric results when test-set sizes are unequal.
      DeprecationWarning)
    




    RandomizedSearchCV(cv=8, error_score='raise-deprecating',
                       estimator=RandomForestRegressor(bootstrap=True,
                                                       criterion='mse',
                                                       max_depth=None,
                                                       max_features='auto',
                                                       max_leaf_nodes=None,
                                                       min_impurity_decrease=0.0,
                                                       min_impurity_split=None,
                                                       min_samples_leaf=1,
                                                       min_samples_split=2,
                                                       min_weight_fraction_leaf=0.0,
                                                       n_estimators=100,
                                                       n_jobs=None, oob_score=False,
                                                       random_state=...
                       param_distributions={'min_samples_leaf': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a7b7b87f0>,
                                            'min_samples_split': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a7b7b8f28>,
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x1a7b7b8a20>},
                       pre_dispatch='2*n_jobs',
                       random_state=<mtrand.RandomState object at 0x12a6e0870>,
                       refit=True, return_train_score=False,
                       scoring='neg_mean_squared_error', verbose=10)




```python
tuned_random_forest_regressor.best_estimator_
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                          max_features='auto', max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=0.001249828663231378,
                          min_samples_split=0.0019415164208264953,
                          min_weight_fraction_leaf=0.0, n_estimators=75,
                          n_jobs=None, oob_score=False,
                          random_state=<mtrand.RandomState object at 0x12b1bddc8>,
                          verbose=0, warm_start=False)




```python
(-tuned_random_forest_regressor.best_score_)**0.5
```




    17.152712131824202




```python
# user requests forecast for last week of 2018
user_forecast_request = transformed_testing_features[-7:, :]
user_forecast_response = random_forest_randomized_search.predict(user_forecast_request)
actual_values_response = testing_target.values[-7:, :]

# this would be rendered in Tableau!
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
ax.plot(user_forecast_response.flatten(), label="predicted")
ax.plot(actual_values_response.flatten(), label="actual")
ax.legend()
ax.set_title("Last week of December 2018")
ax.set_ylabel("Solar Power (W/m2)")
ax.set_xlabel("Hours")
```




    Text(0.5, 0, 'Hours')




![png](/img/predicting_files/predicting_84_1.png)


# Forecasting the future of solar power at NEOM

Once the model is trained, the model can generate a new forecast for next day's solar power generation. Once actual values of solar power generation are observed, model can be automatically re-trained and improved.  Model can be retrained with weekly, monthly forecast horizons if longer forecasts are required.


```python
incoming_features = features.loc[[4017]]
new_predictions = tuned_random_forest_regressor.predict(incoming_features)[0]
solar_power_forecast = (pd.DataFrame.from_dict({"Timestamp": pd.date_range(start="2019-01-01", end="2019-01-01 23:00:00", freq='H'),
                                                "Predicted Solar Power (W/m2)": new_predictions})
                          .set_index("Timestamp", inplace=False))
```


```python
_ = solar_power_forecast.plot()
```


![png](/img/predicting_files/predicting_87_0.png)



```python
solar_power_forecast.to_csv("../results/solar-power-forecast-20190101.csv")
```


```python

```
