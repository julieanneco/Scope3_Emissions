<div class='tableauPlaceholder' id='viz1741490001253' style='position: relative'><noscript><a href='#'><img alt='Home ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sc&#47;Scope3EmissionsinGlobalBusiness&#47;Home&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Scope3EmissionsinGlobalBusiness&#47;Home' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sc&#47;Scope3EmissionsinGlobalBusiness&#47;Home&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>< br / >

[Go to Dasboard](https://public.tableau.com/app/profile/julie.anne.hockensmith/viz/Scope3EmissionsinGlobalBusiness/Home)< br / >< br / >

* * *

<img align="right" src="https://github.com/julieanneco/GHG_Emissions/blob/Photos/project_image.png?raw=true" alt="project image" width="500">

<!-- Table of Contents -->
<b>Table of Contents</b>
  <ol>
    <li><a href="#Project-Overview">Project Overview</a>
    <li><a href="#Data-Engineering">Data Engineering: Merging World Bank Data</a>
    <li><a href="#Outlier-Analysis">Outlier Analysis and Removal</a>
    <li><a href="#Skew-Transformation">Skew Transformation</a>
    <li><a href="#Machine-Learning">Machine Learning</a>
          <ul>
          <li><a href="#random-forest">Random Forest</a>
          <li><a href="#xgboost">XGBoost</a>
          </ul>
    <li><a href="#Analysis-Tableau">Interactive Analysis in Tableau</a>
    <li><a href="#conclusion">Conclusion</a>
    <li><a href="#acknowledgements">Acknowledgements</a>
  </ol>

<br />
<hr>
<br />

<!-- Project Overview -->
## Project Overview

As the global focus on sustainability intensifies, organizations are increasingly recognizing the importance of managing their environmental impact across all areas of operation. Among the various components of greenhouse gas emissions, Scope 3 emissions—those that occur in the value chain of an organization, both upstream and downstream—often represent the largest share of a business's carbon footprint. 

This data research project aims to provide a preliminary analysis of Scope 3 emissions within businesses, in order to answer specific research questions and build and improve upon various machine learning models to better predict and forecast future Scope 3 Emissions. 

**The project consists of various steps and processes to achieve a final prelimiary framework for this research effort:**
<ol>
  <li><b>Data Engineering:</b> Scraping, merging, cleaning, and normalizing data. </li>
  <li><b>Outlier Analysis (EDA):</b> Analyzing variables for correlation and regression to build final data frame(s). </li>
  <li><b>Skew Transformation:</b> Utilizing transformation techniques to examine and handle extreme skew. </li>  
  <li><b>Prediction with Machine Learning:</b> Analyzing variables for correlation and using analyzed features to build preliminary machine learning models.</li>
</ol>

This project utilizes both Tableau and Python. The ipynb files are uploaded to this repository and links to Tableau are provided where relevant.<br />


* * *


<!-- Data Engineering -->
## 1. Data Engineering
> [!NOTE]
> Before merging, I have done some preliminary data cleaning and normalizing on the original data:
> - Normalize all country names
> - Map and create regions

### Merging World Bank Data
[View the Python Markdown file for this step](https://github.com/julieanneco/Scope3_Emissions/blob/main/1%20-%20Merging%20World%20Bank%20Data.ipynb)

<details open="open">
  <summary><b><i>Using the WDI API to Scrape Development Data</b></i></summary>
The World Bank API (WDIAPI) has an integrated Python package that simplifies the extraction process of Development indicators and allows for download and use of the data directly in Python. Each indicator has a vector code that is used for querying and downloading functions. This file can serve as a repeatable template for merging any indicator available in the WDI API to the Scope 3 Data for potential analysis. The process maps the country codes to the country names (without requiring specific spelling) to easily pull data for any country.  

The intial merge file includes the indicators: Country GDP, Population, and Total Greenhouse Gas Emissions from 2013 to 2023 to coincide with the Scope 3 data.
</details>

The WDI library is installed and loaded like any standard package. Merging requires a set of steps that can be re-used for any indicator within the API. The steps include:
1. Get the specific indicator code. The *series.info* tool allows you to easily query a keyword to find indicators in the API
2. Create a dataframe using the API, defining the indicator code, country codes, and range/interval of years
3. Transpose the dataframe so that country becomes rows and years become columns
4. Merge to the helper code dataframe to get the incorporated country
5. Melt the dataframe so that every row is a combination of country and year with the indicator as the value
6. Join the final indicator dataframe to the dataset on country and year
7. 
```python
# find total emissions codes and then use the desired code
wb.series.info(q='greenhouse gas emissions')

# get total greenhouse gas emissions for all incorporated countries
country_total_ghg = wb.data.DataFrame(
'EN.GHG.ALL.LU.MT.CE.AR5',      # World Bank Indicator Code (using "Total greenhouse gas emissions including LULUCF (Mt CO2e)")
# country codes inside brackets
['GBR','ESP','USA','ITA','ZAF','IND','FRA','CHE','NLD','JPN','BRA','FIN'
,'CAN','NZL','TUR','DEU','IRL','AUS','AUT','LUX','NOR','SWE','PRT'
,'ARG','DNK','BEL','HKG','MEX','KOR','SGP','CHL','CHN','MYS','ISR','GRC'
,'COL','HUN','RUS','THA','BMU','PER','PAK','SWZ','IDN','NGA','KEN','ZWE'
,'PRY','PHL','POL','ARE','SVN','EGY','VEN','ROU','SVK','CYP','CRI','SLV'
,'ISL','JAM','HND','BLR','ECU','CZE','GHA','FJI','PAN','GTM','MAC','MLT'
,'VNM','KAZ','SAU','LKA','CYM','JOR','MNG','LTU','SMR','BGR','BGD','BOL'
,'SRB','MOZ','KWT','IMN','MUS','KHM','URY','QAT','EST','UKR','DOM','MAR'
,'LBY','CMR','GUY','OMN','MCO','MHL','LIE']
,range(2013, 2025, 1)           # range of years and interval
, index = 'time')

# transpose rows to columns and create dataframe
country_total_ghg = country_total_ghg.transpose()
country_total_ghg = country_total_ghg.reset_index()
country_total_ghg['WB_Code'] = country_total_ghg['index']
country_total_ghg.rename(columns={1: 'WB_Code'}, inplace=True)
country_total_ghg.drop(columns=['index'], inplace=True)
country_total_ghg.head()

# merge country code helper (converting country names to country codes for use as primary ID)
country_total_ghg = pd.merge(country_total_ghg, country_codes, on='WB_Code')

# rename year columns
country_total_ghg = country_total_ghg.rename(columns={'YR2013': '2013', 'YR2014': '2014', 'YR2015': '2015', 'YR2016': '2016',
    'YR2017': '2017', 'YR2018': '2018', 'YR2019': '2019', 'YR2020': '2020', 'YR2021': '2021', 'YR2022': '2022', 'YR2023': '2023'})

# drop index
country_total_ghg.drop(columns=['index'], inplace=True)

# Melt the dataframe to create a row for every country and every year
melted_ghg = country_total_ghg.melt(
    id_vars=['incorporated_country'],      # Keep 'incorporated_country' as identifier variable
    value_vars=['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023'],  # Year columns to pivot
    var_name='Year',                       # Name for the new column containing old column names
    value_name='country_total_ghg'         # Name for the new column containing values
)
```
Here you can how the data is pulled from and what happens once it is transposed to easily merge with the Scope 3 Data.

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/WDI1.png" alt="WDI1.key" width="620">
<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/WDI2.png" alt="WDI1.key" width="320">

<br />
* * *
<br />
<!-- Outlier-Analysis -->
## 2. Outlier Analysis and Removal
[View the Python Markdown file for this step](https://github.com/julieanneco/Scope3_Emissions/blob/main/2%20-%20Outlier%20Analysis.ipynb)

Due to the highly skewed nature of the Scope 3 Emissions data, where extreme variation, distribution, and outliers exist, outlier analysis and removal proved to be a complex and cruical step. While this process requires far more analysis, validation, and transformation, this preliminary outlier removal offers the beginning steps to this deep analytical process.

Tableau was better suited to vizualize and analyze outliers as it offers and easier way to interact with and drill into, in order to better understand the distribution and inherent issues within the dataset. 

**Click here to visit the [Outlier Dashboard] (https://public.tableau.com/app/profile/julie.anne.hockensmith/viz/OutlierDashboardScope3Emissions/Outlier).**
<div class='tableauPlaceholder' id='viz1741486011909' style='position: relative'><noscript><a href='#'><img alt='Outlier ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ou&#47;OutlierDashboardScope3Emissions&#47;Outlier&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='OutlierDashboardScope3Emissions&#47;Outlier' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ou&#47;OutlierDashboardScope3Emissions&#47;Outlier&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1741486011909');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='2000px';vizElement.style.height='2027px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='2000px';vizElement.style.height='2027px';} else { vizElement.style.width='100%';vizElement.style.height='4377px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>

The Outlier Dashboard offered valuable insight to understand the nature of outliers. Using Python, I then implemented a series of processes to handle outliers, including:
1. Extreme variation anlaysis at a company level
2. z-score analysis at various categorical levels
3. Custom IQR binning to reduce percentile volumes
4. Validating changes in Standard Deviation, Skew, and Kurtosis

#### Extreme Variation


#### Z-Score


### Custom IQR Binning


<br />
* * *
<br />
<!-- Skew-Transformation -->
## 3. Skew Transformation

[View the R Markdown file for this step](https://github.com/julieanneco/predictingHDI/blob/main/PredictHDI_Step2_EDA.Rmd)


<b>Correlation Matrix</b>

To begin analysis, I removed any rows with NULL values and all non-numerical columns from the key.ind data frame in order to create a correlation matrix. This matrix allowed me to understand variables that highly correlated to the Human Development Index (HDI). For the correlation matrix, I used the corrplot and color brewer packages.

```r
Matrix <-cor(key.corr)
corrplot(Matrix, type="upper", order="hclust", method="pie",
         col=brewer.pal(n=8, name="RdYlBu"))
```
<img src="https://github.com/julieanneco/predictingHDI/blob/photos/matrix1.png?raw=true" alt="Correlation Matrix" width="650">

The strength of the correlation is indicated by the pies. Blue indicates a positive correlation and red indicates a negative correlation. It is easy to see variables with strong correlation to HDI and I have outlined each of them. Using only these variables, I then took a deeper look at the regression. I created a data frame <b>predict.hdi</b> to further narrow down the data that will be used for building a prediction model. Looking at a matrix of scatterplots, there is obvious regression to HDI for the variables selected.

<img src="https://github.com/julieanneco/predictingHDI/blob/photos/matrix2.png?raw=true" alt="Scatterplot Matrix" width="650">

Individually, each variable shows strong linear regression and low p-values. The only variable with more of an exponential trend is GDP Per Capita. For the final model, I explored outliers and ultimately chose to include GDP per capita because, while not the only factor, it is a key economic development indicator. 

<img align="left" src="https://github.com/julieanneco/predictingHDI/blob/photos/birth.png?raw=true" alt="birth rate" width="350">
<br />
<br />
<b>Birth Rate and HDI</b>

Residual standard error: 0.07639 on 4676 degrees of freedom

Multiple R-squared:  0.7881,	Adjusted R-squared:  0.7881

F-statistic: 1.739e+04 on 1 and 4676 DF,  p-value: < 2.2e-16
<br />
<br />

<img align="left" src="https://github.com/julieanneco/predictingHDI/blob/photos/edu.png?raw=true" alt="edu index" width="350">
<br />

<b>Education Index and HDI</b>

Residual standard error: 0.05243 on 4676 degrees of freedom

Multiple R-squared:  0.9002,	Adjusted R-squared:  0.9002 

F-statistic: 4.217e+04 on 1 and 4676 DF,  p-value: < 2.2e-16
<br />
<br />
<br />

<img align="left" src="https://github.com/julieanneco/predictingHDI/blob/photos/gdp.png?raw=true" alt="gdp per capita" width="350">
<br />
<br />
<b>GDP Per Capita and HDI</b>

Residual standard error: 0.1197 on 4676 degrees of freedom

Multiple R-squared:   0.48,	Adjusted R-squared:  0.4798 

F-statistic:  4316 on 1 and 4676 DF,  p-value: < 2.2e-16
<br />
<br />
<br />
<img align="left" src="https://github.com/julieanneco/predictingHDI/blob/photos/infant.png?raw=true" alt="infant mortality rate" width="350">
<br />
<br />
<b>Infant Mortality Rate and HDI</b>

Residual standard error: 0.07095 on 4676 degrees of freedom

Multiple R-squared:  0.8172,	Adjusted R-squared:  0.8172 

F-statistic: 2.091e+04 on 1 and 4676 DF,  p-value: < 2.2e-16
<br />
<br />
<br />
<img align="left" src="https://github.com/julieanneco/predictingHDI/blob/photos/lifeexp.png?raw=true" alt="life expectancy" width="350">
<br />
<b>Life Expectancy and HDI</b>

Residual standard error: 0.06848 on 4676 degrees of freedom

Multiple R-squared:  0.8297,	Adjusted R-squared:  0.8297 

F-statistic: 2.279e+04 on 1 and 4676 DF,  p-value: < 2.2e-16
<br />
<br />
<br />
<br />
<br />


<br />
* * *
<br />
<!-- Machine-Learning -->
# 4. Machine Learning Models

[View the R Markdown file for this step](https://github.com/julieanneco/predictingHDI/blob/main/PredictHDI_Step3_ML.Rmd)

<!-- xgboost -->
## XGBoost

The <b>predict.hdi</b> data frame has been cleaned and validated for regression. Using this final data frame that resulted from steps 1 and 2, I decided to test a random forest prediction model. To begin, I split the data into 2 partitions using the caret package. I chose to partition 90% for training and 10% for testing because I wanted to have as much data to train as possible, though standard partitioning is often around 80/20.
```{r}
set.seed(123)
hdi.samples <- predict.hdi$hdi %>%
	createDataPartition(p = 0.9, list = FALSE)
train.hdi  <- predict.hdi[hdi.samples, ]
test.hdi <- predict.hdi[-hdi.samples, ]
```

Using the randomForest package, I fit a basic random forest regression model with 500 trees and a mtry of 3. I then plotted the error versus the number of trees.
```{r}
hdi.rf.1 <- randomForest(hdi ~ ., data = train.hdi, ntree=500, mtry = 3, 
	importance = TRUE, na.action = na.omit) 
print(hdi.rf.1) 
plot(hdi.rf.1) 
```

<img src="https://github.com/julieanneco/predictingHDI/blob/photos/trees.png?raw=true" alt="errors" width="500">

After tuning and testing for out of bag (OOB) error improvement and also looking at the significance of each variable for possible mean changes, I determined the original model was still the best fit with a <b>root-mean square error of .0087</b> and an <b>explained variance of 99.76%</b>, which both indicate a highly valid fit. Moving forward with this model, I made predictions on the test data, converted the predictions to a data frame, and merged them with the original test data to see a side-by-side comparison. This sample shows just how close the prediction model gets to the actual human development index based on the variables used in the random forest training.

<img src="https://github.com/julieanneco/predictingHDI/blob/photos/predictions1.png?raw=true" alt="predictions" width="450">

The mean distance of the prediction to the actual HDI is -.0051, which is very impressive given some of the variance in each variable dataset. I created a plot to visualize the prediction variance for the entire test data. The model seems to predict higher indices better, but only by a nominal amount. 

![alt text](https://github.com/julieanneco/predictingHDI/blob/photos/RF-R-Results.jpg?raw=true)

<!-- Analysis-Tablea -->
## Interactive Analysis in Tableau

<div class='tableauPlaceholder' id='viz1741234238142' style='position: relative'><noscript><a href='#'><img alt='Story 1 ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Em&#47;EmissionsinBusiness&#47;Story1&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='EmissionsinBusiness&#47;Story1' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Em&#47;EmissionsinBusiness&#47;Story1&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1741234238142');                    var vizElement = divElement.getElementsByTagName('object')[0];                    vizElement.style.width='2000px';vizElement.style.height='5027px';                    var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>


The random forest regression had surprisingly strong results, but I decided to also test classification since this is another common use for random forest prediction. To begin, I created 3 categories for HDI (Low, Med, High) and converted this column to a factor with 3 levels and then created an 80/20 partition using caTools, which is another package for creating partitions. I then fit the model with 500 trees and mtry of 2.

```{r}
predict.hdi.2$hdi.cat[predict.hdi.2$hdi < .650 ] = "Low"
predict.hdi.2$hdi.cat[predict.hdi.2$hdi > .850 ] = "High"
predict.hdi.2$hdi.cat[is.na(predict.hdi.2$hdi.cat)] <- "Mid"

(predict.hdi.2$hdi.cat = factor(predict.hdi.2$hdi.cat, levels=c("Low", "Mid", "High")))

set.seed(123)
split = sample.split(predict.hdi.2$hdi.cat, SplitRatio = 0.80)
hdi.training.set = subset(predict.hdi.2, split == TRUE)
hdi.test.set = subset(predict.hdi.2, split == FALSE)

hdi.rfc = randomForest(x = hdi.training.set[1:5],
y = hdi.training.set$hdi.cat,
ntree = 500, random_state = 0)
```

<img src="https://github.com/julieanneco/predictingHDI/blob/photos/rfc.png?raw=true" alt="predictions" width="650">

The model returned an <b>OOB error rate estimate of 1.84%</b>. Looking at a confusion matrix reveals just how well the classification prediction model performed on the test data with an error rate of 1.497.

<img src="https://github.com/julieanneco/predictingHDI/blob/photos/confusion.png?raw=true" alt="confusion matrix" width="280">

<!-- Conclusion -->
## Conclusion

The actual indicators predict even better, which is no surprise. What does feel like an accomplishment is how closely the original model also predicts the HDI. As stated in the project overview, the Human Development Indicator is meant to emphasize that people should be the ultimate criteria for assessing development, rather than economic growth alone. My assumption as to why the original prediction results based on regression are so close to the actual indicators is inherent of the relationship between the variables themselves (GNI and GDP, birth rate, infant mortality, and life expectancy). The interconnected nature of global development provides insight into what factors shed light into how we might continue to reduce poverty based on multiple dimensions that are economic, human, environmental, and so on.

<img src="https://github.com/julieanneco/predictingHDI/blob/photos/compare.png?raw=true" alt="compare results" width = "650">


<br />

<!-- Acknowledgements -->
## Acknowledgements

### Data 
* [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators)
* [UNDP Human Development Data](http://hdr.undp.org/en/data)

### R Packages Utilized
* [WDI](https://www.rdocumentation.org/packages/WDI/versions/2.7.1)
* [plyr](https://www.rdocumentation.org/packages/plyr/versions/1.8.6)
* [tidyr](https://www.rdocumentation.org/packages/tidyr/versions/0.8.3)
* [corrplot](https://www.rdocumentation.org/packages/corrplot/versions/0.84)
* [RColorBrewer](https://www.rdocumentation.org/packages/RColorBrewer/versions/1.1-2)
* [ggplot2](https://www.rdocumentation.org/packages/ggplot2/versions/3.3.2)
* [ggpubr](https://www.rdocumentation.org/packages/ggpubr/versions/0.4.0)
* [caret](https://www.rdocumentation.org/packages/caret/versions/6.0-86)
* [randomForest](https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest)
* [caTools](https://www.rdocumentation.org/packages/caTools/versions/1.17.1)

### References
Van der Mensbrugghe, Dominique. (2016). Using R to Extract Data from the World Bank's World Development Indicators. <i>Journal of Global Economic Analysis</i>. 1. 251-283. 10.21642/JGEA.010105AF.

UNDP. Human Development Index (HDI).](http://hdr.undp.org/en/content/human-development-index-hdi) http://hdr.undp.org/en/content/human-development-index-hdi
