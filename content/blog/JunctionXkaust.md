---
title: "JunctionX KAUST 2019 : Predicting the Future of Solar Power Generation at NEOM"
date: 2019-09-15T09:23:18+03:00
draft: false
---


## Solar power generation forecasts will be a critical need if SOs are to balance NEOM's grid. We built an ML model to forecast solar power generation that takes into account NEOM's weather patterns.

[Devpost link](https://devpost.com/software/predicting-solar-power-generation-with-ml)

![ml_prediction](/img/junctionx/ml.jpg)
![dashboard](/img/junctionx/dashboard.jpg)

## Inspiration
Solar power generation forecasts will be a critical need if SOs are to balance NEOM's smart electricity grid with nearly 100% renewables. Even though NEOM is blessed with plenty of solar radiation, NEOM also experiences substantial fluctuations in temperature, wind, and dust and these factors can all have a substantial impact on solar power generation. We built an ML model capable of accurately forecasting solar power generation that takes into account NEOM's unique weather patterns and created a few prototype interactive dashboards to display the data.

## What it does
Our model uses ML techniques to predict future solar power generation at NEOM as a function of the weather data as well as the history solar power generation. After training our model on historical data, we can generate a new forecast for next day's solar power generation. Once the next day's actual values of solar power generation are observed, our model can be automatically re-trained and improved. Model can easily be retrained with weekly or monthly forecast horizons if longer forecasts are required by the SO.

Our interactive dashboard allows the user to interrogate the historical weather and solar power generation data for NEOM as well as to display forecasts of future solar power generation. Dashboard can generate user notifications via the Telegram mobile app to indicate significantly changes to either weather or solar energy forecasts.

Our web and mobile app prototype will allow users (SOs, residential, and industrial prosumers) at NEOM to interact with weather and solar energy forecasts and to receive alerts about significant upcoming changes to either.

## How we built it
We built the ML model using widely used open-source tools: Python, Jupyter, Scikit-learn, Keras, and Tensorflow. Our interactive dashboard leveraged another open-source tool called Grafana. We used Adobe XD to prototype our web and mobile apps.

## Challenges we ran into
* Finding the right machine learning approach. We evaluated a number of classical ML approaches including linear regression, multi-task elasticnet, multi-task lass regression, random forest regression, and GRU and LSTM deep neural networks. Random forest regression eventually emerged as the most performant.

* Finding the right dashboard tool. We attempted to roll our own but then decided this approach was too time consuming and explored a few off-the-shelf dashboard solutions before eventually settling on Grafana.

* Injecting the ML model into a web application turned out to be significantly more difficult than we had suspected.

## Accomplishments that we're proud of
* Our overall concept was very solid and useful.
* Our interactive dashboard with mobile notifications is really cool!
* Our ML modeling pipeline with good forecasting results could be put into production at NEOM.

## What we learned
* New machine learning approached and data analysis techniques.
* Serving machine learning models in web and mobile apps is hard!
* Teamwork!

## What's next for Forecasting Solar Power Generation for NEOM using ML?
Predicting the future or electricity demand at NEOM!

## Built With
* python
* keras
* react
* node.js
* tensorflow
* scikit-learn
* jupyter
* grafana
* pandas

## Try it out
* [GitHub Repo](https://github.com/davidrpugh/junctionx-kaust-2019)
* [GitHub Repo (fork on mine)](https://github.com/iliesaya/junctionx-kaust-2019)
 * [mybinder.org](https://mybinder.org/v2/gh/davidrpugh/junctionx-kaust-2019/master?filepath=notebooks%2Fpredicting-solar-power-at-neom.ipynb)