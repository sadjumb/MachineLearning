# 1. MachineLearning
## 1.1. About Dataset

### 1.1.1. Source & Acknowledgements
Dataset: [weather dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package/data)
Copyright Commonwealth of Australia 2010, Bureau of Meteorology.

### 1.1.2. Content
The dataset contains about 10 years of daily weather observations from different locations across Australia. Observations were drawn from numerous weather stations.

### 1.1.3. Task
In this project, I will use this data to predict whether or not it will rain the next day. There are 23 attributes including the target variable "RainTomorrow", indicating whether or not it will rain the next day or not.

### 1.1.4. Data description
* Date - The date of observation;
* Location - The common name of the location of the weather station;
* MinTemp - The minimum temperature in degrees celsius
* MaxTemp - The maximum temperature in degrees celsius
* Rainfall - The amount of rainfall recorded for the day in mm
* Evaporation - The so-called Class A pan evaporation (mm) in the 24 hours to 9am
* Sunshine - The number of hours of bright sunshine in the day.
* WindGustDir - The direction of the strongest wind gust in the 24 hours to midnight
* WindGustSpeed - The speed (km/h) of the strongest wind gust in the 24 hours to midnight
* WindDir9am - Direction of the wind at 9am
* WindDir3pm - Direction of the wind at 3pm
* WindSpeed9am - Wind speed (km/hr) averaged over 10 minutes prior to 9am
* WindSpeed3pm - Wind speed (km/hr) averaged over 10 minutes prior to 3pm
* Humidity9am - Humidity (percent) at 9am
* Humidity3pm - Humidity (percent) at 3pm
* Pressure9am - Atmospheric pressure (hpa) reduced to mean sea level at 9am
* Pressure3pm - Atmospheric pressure (hpa) reduced to mean sea level at 3pm
* Cloud9am - Fraction of sky obscured by cloud at 9am. This is measured in "oktas", which are a unit of eigths. It records how many
* Cloud3pm - Fraction of sky obscured by cloud (in "oktas": eighths) at 3pm. See Cload9am for a description of the values
* Temp9am - Temperature (degrees C) at 9am
* Temp3pm - Temperature (degrees C) at 3pm
* RainToday - Boolean: 1 if precipitation (mm) in the 24 hours to 9am exceeds 1mm, otherwise 0
* RainTomorrow - The amount of next day rain in mm. Used to create response variable RainTomorrow. A kind of measure of the "risk".

## 1.2. DATA VISUALIZATION AND CLEANING
!["Rain Tomorrow" statistic](./Images/RainTomorrow.png "Hello world"){text="Rain Tomorrow" statistic}
sadas