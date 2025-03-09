<div class='tableauPlaceholder' id='viz1741490001253' style='position: relative'><noscript><a href='#'><img alt='Home ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sc&#47;Scope3EmissionsinGlobalBusiness&#47;Home&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Scope3EmissionsinGlobalBusiness&#47;Home' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Sc&#47;Scope3EmissionsinGlobalBusiness&#47;Home&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>

**[Click Here to Visit the Scope 3 Emissions Dasboard](https://public.tableau.com/app/profile/julie.anne.hockensmith/viz/Scope3EmissionsinGlobalBusiness/Home)**

* * *

<!-- Table of Contents -->
<b>Table of Contents</b>
  <ol>
    <li><a href="#Project-Overview">Project Overview</a>
    <li><a href="#Data-Engineering">Data Engineering: Merging World Bank Data</a>
    <li><a href="#Outlier-Analysis">Outlier Analysis and Removal</a>
          <ul>
          <li><a href="#variation">Extreme Variation Analysis (Company Level)</a>
          <li><a href="#zscore">Z-score Analysis (Primary Activity Level)</a>
          <li><a href="#iqr">Custom IQR Binning to Reduce Percentile Distribtion/Volume</a>
          <li><a href="#validation">Validating Changes in Standard Deviation, Skew, and Kurtosis</a>
	  </ul>
    <li><a href="#Skew-Transformation">Skew Transformation</a>
	  <ul>
	  <li><a href="#boxcox">Box Cox Transformation</a>
	  <li><a href="#quantile">Quantile Transformation</a>
	  </li>
	  </ul>
    <li><a href="#Machine-Learning">Machine Learning</a>
          <ul>
          <li><a href="#corelation">Correlation Analysis</a>
          <li><a href="#encoding">Feature Engineering and Encoding</a>
          <li><a href="#overview">Machine Learning Overview</a>
          <li><a href="#random-forest">Random Forest Results</a>
          <li><a href="#xgboost">XGBoost Results</a>
          <li><a href="#performance">Assessing Performance (MAE, Prediction Variance, Cross-fold Validation)</a>
          <li><a href="#performance">Comparing Models</a>
	  </ul>
    <li><a href="#conclusion">Conclusion</a>
    <li><a href="#acknowledgements">Acknowledgements</a>
  </ol>

<br />
<hr>

<!-- Project Overview -->
## Project Overview

As the global focus on sustainability intensifies, organizations are increasingly recognizing the importance of managing their environmental impact across all areas of operation. Among the various components of greenhouse gas emissions, Scope 3 emissions—those that occur in the value chain of an organization, both upstream and downstream—often represent the largest share of a business's carbon footprint. 

This data research project aims to provide a preliminary analysis of Scope 3 emissions within businesses, in order to answer specific research questions and build and improve upon various machine learning models to better predict and forecast future Scope 3 Emissions. 

This project utilizes both Tableau and Python. The ipynb files are uploaded to this repository and links to Tableau are provided where relevant.<br />

#### *Data Overview*

The inital data set includes 181,326 rows with the following unique columns and statistics:

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/summary.png" alt="summarydata.key" width="800">

#### *Data Challenges*

Initial analysis of the data shows significant peaks and valley when looking at the Average Scope 3 Emissions amount year over year (Yoy). The dataset indicates a high probability of discrepancy, inconsistency, extreme outliers, and skew. The lack of uniform/linear distributuion suggests unreliable data that creates consumption challenges for analysis and machine learning.

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/EDA.png" alt="EDA.key" width="700">

Highly skewed data with a very non-normal distribution can present a range of challenges, especially in statistical modeling and analysis. Some key difficulties include:
1. Statistical Analysis - Many standard statistical tests and models assume that the data follows a normal distribution or that the residuals of the model are normally distributed. When data is highly skewed or has a non-normal distribution, these assumptions are violated, which can lead to unreliable coefficients/parameter estimates, increased Type I and Type II errors with skewed distribution of residuals.
2. Impact to Model Performance - Highly skewed data can lead to prediction bias (prioritization of the majority class/values) as well as reduction in model interpretability (overfitting)
3. Difficulty in transformations, especially given the range of extreme negative values to extreme positive values.
4. Extreme outliers impact a model's ability to generalize and can lead to heteroscedasticity where the variance of errors or residuals is not constant across different levels of the independent variable(s), reducing confidence in standard errors.


**The project consists of various steps and processes to achieve a final prelimiary framework for this research effort:**
<ol>
  <li><b>Data Engineering:</b> Scraping, merging, cleaning, and normalizing data. </li>
  <li><b>Outlier Analysis (EDA):</b> Analyzing variables for correlation and regression to build final data frame(s). </li>
  <li><b>Skew Transformation:</b> Utilizing transformation techniques to examine and handle extreme skew. </li>  
  <li><b>Prediction with Machine Learning:</b> Analyzing variables for correlation and using analyzed features to build preliminary machine learning models.</li>
</ol>


* * *

<!-- Data Engineering -->
## 1. Data Engineering
> [!NOTE]
> Before merging, I have done some preliminary data cleaning and normalizing on the original data:
> - Normalize all country names
> - Map and create regions

### Merging World Bank Data
[View the Python Markdown file for this step](https://github.com/julieanneco/Scope3_Emissions/blob/main/1%20-%20Engineering%20-%20Merging%20World%20Bank%20Data.ipynb)

<details open="open">
  <summary><b><i>Using the WDI API to Scrape Development Data</b></i></summary>
The World Bank API (WDIAPI) has an integrated Python package that simplifies the extraction process of Development indicators and allows for download and use of the data directly in Python. Each indicator has a vector code that is used for querying and downloading functions. This file can serve as a repeatable template for merging any indicator available in the WDI API to the Scope 3 Data for potential analysis. The process maps the country codes to the country names (without requiring specific spelling) to easily pull data for any country.  
</details>
	  
The initial merge file includes the indicators: Country GDP, Population, and Total Greenhouse Gas Emissions from 2013 to 2023 to coincide with the Scope 3 data.

The WDI library is installed and loaded like any standard package. Merging requires a set of steps that can be re-used for any indicator within the API. The steps include:
1. Get the specific indicator code. The *series.info* tool allows you to easily query a keyword to find indicators in the API
2. Create a dataframe using the API, defining the indicator code, country codes, and range/interval of years
3. Transpose the dataframe so that country becomes rows and years become columns
4. Merge to the helper code dataframe to get the incorporated country
5. Melt the dataframe so that every row is a combination of country and year with the indicator as the value
6. Join the final indicator dataframe to the dataset on country and year

```python
# find total emissions codes and then use the desired code
wb.series.info(q='greenhouse gas emissions')

# get total greenhouse gas emissions for all incorporated countries
country_total_ghg = wb.data.DataFrame(
'EN.GHG.ALL.LU.MT.CE.AR5',      # World Bank Indicator Code
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
# rename year columns (not shown)
# drop index
country_total_ghg.drop(columns=['index'], inplace=True)

# Melt the dataframe to create a row for every country and every year
melted_ghg = country_total_ghg.melt(
    id_vars=['incorporated_country'],      # Keep 'incorporated_country' as identifier variable
    value_vars=['2013','2014','2015','2016','2017','2018','2019','2020','2021','2022','2023'],
    var_name='Year',                       
    value_name='country_total_ghg')
```

Here you can see how the data is pulled from the API and what happens once it is transposed to easily merge with the Scope 3 Data.

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/WDI1.png" alt="WDI1.key" width="580"> &rarr; <img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/WDI2.png" alt="WDI1.key" width="320">

* * *

<!-- Outlier-Analysis -->
## Outlier Analysis and Removal
[View the Python Markdown file for this step](https://github.com/julieanneco/Scope3_Emissions/blob/main/2%20-%20Outlier%20Analysis.ipynb)

Due to the highly skewed nature of the Scope 3 Emissions data, where extreme variation, distribution, and outliers exist, outlier analysis and removal proved to be a complex and cruical step. While this process requires far more analysis, validation, and transformation, this preliminary outlier removal offers the beginning steps to this deep analytical process.

Tableau was better suited to vizualize and analyze outliers as it offers and easier way to interact with and drill into, in order to better understand the distribution and inherent issues within the dataset. 

**[Click here to visit the Outlier Dashboard](https://public.tableau.com/app/profile/julie.anne.hockensmith/viz/OutlierDashboardScope3Emissions/Outlier)**
<div class='tableauPlaceholder' id='viz1741486011909' style='position: relative'><noscript><a href='#'><img alt='Outlier ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ou&#47;OutlierDashboardScope3Emissions&#47;Outlier&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='OutlierDashboardScope3Emissions&#47;Outlier' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Ou&#47;OutlierDashboardScope3Emissions&#47;Outlier&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>

<br />

The Outlier Dashboard offered valuable insight to understand the nature of outliers. Using Python, I then implemented a series of processes to handle outliers, including:
1. Extreme variation anlaysis at a company level
2. z-score analysis at a Primary Activity level 
3. Custom IQR binning to reduce percentile volumes
4. Validating changes in Standard Deviation, Skew, and Kurtosis

*Calculating baseline skewness and kurtosis before outlier removal*
```python
original_skew = outlier_df['Scope_3_emissions_amount'].skew()
original_kurtosis = outlier_df['Scope_3_emissions_amount'].kurtosis()
```
<table><tr><td>
Skew: 325.8187947316704

Kurtosis: 116921.06539086264
</td></tr></table>


<br />


*Looking at observation volume by year*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/YoY.png" alt="YoY.key" width="500">


<!-- variation -->
#### Extreme Variation Anlaysis (Company Level)


*Function to Find Accounts with Extreme Variation (inconsistent observations or extreme peaks/valleys from year to year*

```python
def find_extreme_variations(outlier_df, value_column, year_column='Year', 
                          activity_column='Primary activity', 
                          account_column='account_id',
                          z_score_threshold=2):
    # Calculate variation metrics for each account within a primary activity
    variation_stats = (outlier_df.groupby([activity_column, account_column])
                      .agg({value_column: ['std', 'mean', 'min', 'max', 'count'], year_column: list
                      }).reset_index())
    # Flatten column names
    variation_stats.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in variation_stats.columns]
    # Calculate coefficient of variation (CV)
    variation_stats['cv'] = (variation_stats[f'{value_column}_std'] / 
                           variation_stats[f'{value_column}_mean'].abs())
    # Calculate range ratio
    variation_stats['range_ratio'] = (variation_stats[f'{value_column}_max'].abs() / 
                                    variation_stats[f'{value_column}_min'].abs())
    # Calculate Z-scores of CV within each Primary Activity
    variation_stats['cv_zscore'] = (variation_stats
                                   .groupby(activity_column)['cv']
                                   .transform(lambda x: stats.zscore(x)))
    # Identify extreme accounts
    extreme_accounts = variation_stats[
        (variation_stats['cv_zscore'].abs() > z_score_threshold) &
        (variation_stats[f'{value_column}_count'] > 1)  # At least 2 years of data
    ].copy()
    # Sort and format results
    extreme_accounts = extreme_accounts.sort_values(
        ['cv_zscore'], 
        ascending=False)
    # Add year range information
    extreme_accounts['year_range'] = extreme_accounts[f'{year_column}_list'].apply(
        lambda x: f"{min(x)}-{max(x)}")
    return extreme_accounts
```
Sample of vizualizations that validated the inconsistent accounts and were subsequently removed from the data:


<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/extreme_variance1.png" alt="YoY.key" width="300"> <img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/extreme_variance2.png" alt="YoY.key" width="300"> <img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/extreme_variance3.png" alt="YoY.key" width="300">


<!-- zscore -->
#### Z-score Analysis (Primary Activity Level)


*Gettings the z-scores for observations within each Primary Activty and storing z-scores over 3*

```python
# Initialize list to store outlier records
outlier_records = []
# Process each Primary Activity
for activity in outlier_df['Primary activity'].unique():
    # Get data for this activity
    activity_data = outlier_df[outlier_df['Primary activity'] == activity]
    # Calculate z-scores and std dev for this activity
    z_scores = stats.zscore(activity_data['Scope_3_emissions_amount'])
    std_dev = activity_data['Scope_3_emissions_amount'].std()
    # Find outliers (z-score >= 3)
    outlier_mask = abs(z_scores) >= 3
    # Get outlier records
    outliers = activity_data[outlier_mask].copy()
    outliers['z_score'] = z_scores[outlier_mask]
    outliers['std_dev'] = std_dev
    # Add to records list
    outlier_records.append(outliers[['account_id', 'account_name', 'Primary activity', 
                                   'Scope_3_emissions_amount', 'Year', 'z_score', 'std_dev']])
# Combine all outliers into one dataframe
outliers = pd.concat(outlier_records)
# Sort by absolute z-score (highest to lowest)
outliers = outliers.sort_values(by='z_score', key=abs, ascending=False)
# Reset index
outliers = outliers.reset_index(drop=True)
```

*Vizualizing Outliers: Outliers Year over Year*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/outliers_yoy.png" alt="outliersyoy.key" width="600">


*Vizualizing Outliers: A sample of z-scores by Primary Activity*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/zscore_primary_activity.png" alt="zscoresbyprimaryactivity.key" width="800">



<!-- iqr -->
#### Custom IQR Binning to Reduce Percentile Distribtion/Volume

*Creating a Scatterplot to better understand volume and distribution within custom percentiles*

```
# calculate custom percentiles
percentiles = [25, 50, 80, 90, 95, 96, 97, 98, 99, 100]
percentile_values = {p: outlier_df_cleaned['Scope_3_emissions_amount'].quantile(p/100) for p in percentiles}
# Create custom color scale for different percentile ranges
color_scale = ['#d4e6f1','#a9cce3','#7fb3d5','#5499c7','#2980b9', '#2471a3', '#1f618d', '#1a5276', '#1b4f72', '#641e16']
# Initialize the figure
fig = go.Figure()

# Create percentile ranges and add traces for each range
prev_percentile = 0
for i, percentile in enumerate(percentiles):
    if i == 0:
        mask = outlier_df_cleaned['Scope_3_emissions_amount'] <= percentile_values[percentile]
        start = 0
    else:
        mask = (outlier_df_cleaned['Scope_3_emissions_amount'] > percentile_values[prev_percentile]) & \
               (outlier_df_cleaned['Scope_3_emissions_amount'] <= percentile_values[percentile])
        start = prev_percentile
    
    data = outlier_df_cleaned[mask].copy()
    data['percentile_range'] = f"{start}-{percentile}"
   
    if len(data) > 0:  # Only add trace if there is data in this range
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Scope_3_emissions_amount'],
                mode='markers',
                name=f"{start}-{percentile}th percentile",
                marker=dict(
                    color=color_scale[i] if i < len(color_scale) else color_scale[-1],
                    size=6),
                hovertemplate=(
                    '<b>Account ID:</b> %{customdata[0]}<br>' +
                    '<b>Account Name:</b> %{customdata[1]}<br>' +
                    '<b>Primary Activity:</b> %{customdata[2]}<br>' +
                    '<b>Scope 3 Emissions:</b> %{y:,.2f}<br>' +
                    '<b>Percentile Range:</b> %{customdata[3]}<br>'),
                customdata=np.column_stack((
                    data['account_id'],
                    data['account_name'],
                    data['Primary activity'],
                    data['percentile_range']))))
    prev_percentile = percentile
# Add lines for each percentile (not shown)
# Update layout (not shown)
fig.show()
```

**Percentile Results Before**

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/outliers_percentile_before.png" alt="outliers_percentile_before.key" width="800">

*Assessing Rows in the 99th percentile*

```
# how many accounts have at least 1 row with a scope 3 emissions amount in the 99th percentile or higher?
overall_99th = outlier_df_cleaned['Scope_3_emissions_amount'].quantile(0.99)
high_emission_accounts_99 = outlier_df_cleaned[
    outlier_df_cleaned['Scope_3_emissions_amount'] >= overall_99th
]['account_id'].unique()
n_high_accounts = len(high_emission_accounts_99)
```
<table><tr><td>
Number of accounts with emissions in 99th percentile or higher: 119
	
Percentage of total accounts: 9.7%

99th percentile threshold: 52,527,444.36
</td></tr></table>

**Percentile Results After Removing Values in the 99th Percentile**

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/outliers_percentile_after.png" alt="outliers_percentile_before.key" width="800">


<!-- validation -->
#### Validating Changes in Standard Deviation, Skew, and Kurtosis

*Comparing final data post-outlier removal*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/outlier_compare.png" alt="outlier_compare.key" width="300">


* * *

<!-- Skew-Transformation -->
## 3. Skew Transformation

[View the Python file for this step](https://github.com/julieanneco/Scope3_Emissions/blob/main/3%20-%20Skew%20Transformation.ipynb)

**Analyzing Skew**

*Data has a positive/right skew when*:
   - Long tail extends to the right
   - Most values cluster on the left
   - Mean > Median

*Interpretation*:
   - When skewness is > 1, the data is highly skewed
   - A Fisher kurtosis of 0 indicated perfect normal distribution. The farther from 0, the less normal-like tails.

While all features should be transformed within the Machine Learning pipeline, this step shows the analysis of Market Cap transformation using both Box Cox and Quantile transformation for use in future analysis and research. 

*Quantile Transformer*
  - Best for non-normal distributions where data is heavily skewed
  - Robust to outliers and reduces the impact of extreme values
  - Uses quantiles information and maps data to either a uniform or normal distribution
  - Disadvantge: loses absolute value relationships

*Box Cox*
 - Makes non-normal data more normal-like, reduces skewness, stabilizes variance
 - Helps meet assumptions for many statistical tests by assuming linear model assumptions
 - Disadvantage: Only works with positive values and cannot handle zeros
   

**Distributions Before and After Skew Transformation**

*Box Cox*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/boxcox.png" alt="BoxCox.key" width="820">

*Quantile*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/quantile.png" alt="Quantile.key" width="820">


* * *

<!-- Machine-Learning -->
# 4. Machine Learning Models

[View the Python file for this step](https://github.com/julieanneco/predictingHDI/blob/main/4%20-%20Machine%20Learning.ipynb)


<!-- correlation -->
## Correlation Analysis

*Heat Map Showing Correlation of Target Variable (y) to Features (X)*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/heatmap.png" alt="heatmap.key" width="820">

Removing highly correlated features is a crucial preprocessing step in machine learning to improve model performance, interpretability, generalization, and computational efficiency. It helps create more robust and efficient models that are better suited for real-world applications. The data available has many highly correlated features and this is due to many sub variables or different metrics for similar values, such as revenue/profit, liabilities, etc. I needed to remove a large portion of features to maintain performance. 


*Correlation of Features after Removing Sub-Variables and Highly Correlated Features*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/corr_features.png" alt="Quantile.key" width="500">

<!-- encoding -->
## Feature Engineering and Encoding

Most Machine Learning models require data encoding for categorical values. There are 2 main encoding methods used:


*1. Label encoding*

*2. One hot encoding*

Label encoding assigns a numerical value to every value in a categorical column. This method essentially converts a unique string into a unique number. This route can be problematic to the algorithm if the model assigns value to the numbers beyond categorical limitation. For instance, primary activity has over 200 values. It's safe to assume the model might interpret a value of 1 differently than a value of 200. One hot encoding converts every categorical value in each column to a binary value. For machine learning, this is the more ideal option, as it allows every category to be represented by its own binary feature. This prevents the model from assuming an ordinal relationship or ranking  between categories, which can cause inaccurate predictions. Because this dataset includes categorical columns wih many categories, one hot encoding will result in hundreds of features in order to capture every possible categorical value. This will likey result in overly high dimensionality and accumulated feature noise with very little correlation for reliable predictions. To decide which columns to keep for training, I one hot encoded Primary sector and Primary activity. I chose one hot encoded given the limitations of label encoding for categories with numerous values.

```python
# One-hot encode Primary Sector
sector_dummies = pd.get_dummies(df['Primary sector'], prefix='sector')
# Calculate correlations with Scope_3_emissions_amount
sector_correlations = sector_dummies.apply(lambda x: x.corr(df['Scope_3_emissions_amount']))
# Append one-hot encoded columns to original dataframe
df = pd.concat([df, sector_dummies], axis=1)

# One-hot encode Primary Activity
sector_dummies = pd.get_dummies(df['Primary activity'], prefix='activity')
# Calculate correlations with Scope_3_emissions_amount
sector_correlations = sector_dummies.apply(lambda x: x.corr(df['Scope_3_emissions_amount']))
# Append one-hot encoded columns to original dataframe
df = pd.concat([df, sector_dummies], axis=1)
```

I calculated correlations, but the results of one hot encoding is not going to provide any significant correlation. It is best tested in the model and out of the model to see if value is added. For this preliminary model, I used both encoded features in both models. 

## Machine Learning Overview

The models that best suit the nature of this data include:

##### 1. **Random Forest**
##### 2. **XGBoost**

Both models work well for the follwing reasons:
- Both are able to handle complex data relationships, which is a characteristic of Scope 3 emissions data.
- Able to capture non-linear relationships and interactions between variables.
- Robust to outliers, which common in emissions data and are less prone to overfitting.
- Work well with mixed data types (numerical and categorical), which is common in emissions reporting.
- Feature Importance: Both provide clear insights into which factors most influence emissions and can help identify key areas for emissions reduction.
- Scope 3 data frequently has gaps due to reporting challenges. While missing data was removed from the final dataset, it would be worthwhile to also test the models without data removal and compare the results as both models do well with handling missing values. (Random Forest can operate with missing data without imputation; XGBoost has built-in methods for missing value handling)

##### **Model Design**

The algorithms will train on a time-series of all previous years in conjunction with features in order to predict the Scope 3 Emissions Amount for the most recent year (2023) within each Emission type category.

##### **MAE (Mean Absolute Error)**

Performance will be analyzed using MAE. MAE measures the average magnitude of errors between predicted and actual values, without considering their direction (high or low). It is best in this prediction scenario because:

- The absolute scale of errors matters, as the absolute difference between actual and predicted emission amounts is highly important.
- MAE is in the same unit as the emission amount and both over and under-predictions are equally important.
- Direct, interpretable results and error measurements are needed in this scenario (MAE is more interpretable than percentage-based errors).
- Outliers exist but shouldn't dominate the error metric (MAE is more robust to outliers than squared metrics, such as MSE and RMSE).

##### *Code*
The code for each model will do the following (with slight variation specific to each model):
1. Creates a pipeline with preprocessing (StandardScaler for numeric features)
2. Uses TimeSeriesSplit for proper time series validation on Year
3. Processes each emission type separately
4. Calculates MAE for each emission type

This approach provides a robust way to analyze and predict emissions while maintaining the time series nature of the data and handling different emission types separately, given the variability within this category.

```python
# Initialize results storage
results_dict = {}
# Get unique emission types
emission_types = df['Scope_3_emissions_type'].unique()
# Identify numeric columns to scale (excluding Year, target, and encoded categorical features)
all_numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
columns_to_scale = [col for col in all_numeric_columns 
                   if col not in ['Year', 'Scope_3_emissions_amount','account_id']
                   and not col.startswith(('sector_', 'activity_'))]
categorical_columns = [col for col in df.columns 
                      if col.startswith(('sector_', 'activity_'))]

# Create preprocessing steps for StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columns_to_scale)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False)
# Get feature names after preprocessing
feature_names = (columns_to_scale + 
                ['Year'] + 
                categorical_columns)
# Create pipeline with preprocessor
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

for emission_type in emission_types:
    # Filter data for current emission type
    emission_df = df[df['Scope_3_emissions_type'] == emission_type].copy()
    # Sort by Year to ensure chronological order
    emission_df = emission_df.sort_values('Year')
    # Get the most recent year
    max_year = emission_df['Year'].max()
    
    # Split data into train (previous years) and test (most recent year)
    train_data = emission_df[emission_df['Year'] < max_year]
    test_data = emission_df[emission_df['Year'] == max_year]
    # Prepare features and target
    X_train = train_data[columns_to_scale + ['Year'] + categorical_columns].copy()
    y_train = train_data['Scope_3_emissions_amount']
    X_test = test_data[columns_to_scale + ['Year'] + categorical_columns].copy()
    y_test = test_data['Scope_3_emissions_amount']
    
    # Fit pipeline
    pipeline.fit(X_train, y_train)
    # Make predictions
    y_pred = pipeline.predict(X_test)
    # Calculate MAE
    mae = np.mean(np.abs(y_test - y_pred))
    # Store results in dictionary with emission_type as key
    results_dict[emission_type] = {
        'mae': mae,
        'actuals': y_test.tolist(),
        'predictions': y_pred.tolist(),
        'prediction_year': max_year}
```


<!-- random-forest -->
## Random Forest Results

*Results for All Emissions Types - Actuals vs Predictions *

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/RF1.png" alt="randomforest1.key" width="900">

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/RF2.png" alt="randomforest2.key" width="900">


<!-- xgboost -->
## XGBoost Results

*Results for All Emissions Types - Actuals vs Predictions *

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/XGB1.png" alt="XGBoost1.key" width="900">

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/XGB2.png" alt="XGBoost2.key" width="900">


<!-- performance -->
## Assessing Performance (MAE, Prediction Variance, Cross-fold Validation)

#### Prediction Variance

*Grouped Scatterplot*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/RF_Variance.png" alt="RFVariance.key" width="800">

*Distribution*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/RF_Variance_Dist.png" alt="RFVariance.key" width="800">

The remaining outliers are pushing the variance into higher percentiles, which is expected. The model has difficulty predicting the highest values and interestingly predicts extreme negative values in categories where extreme positive values are present.  


#### Cross Validation
*3 Fold Cross-Validation*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/RF_CV.png" alt="CV.key" width="800">

The MAE is generally consistent across 3 cross validation folds.

<!-- compare -->
## Comparing Models

**Final Metrics: Emissions Type Averages**

*Random Forest*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/RF_AVG.png" alt="RFavg.key" width="800">

*XGBoost*

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/XGB_AVG.png" alt="XGBavg.key" width="800">

**Which model performed better overall?**

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/model_compare_1.png" alt="compare.key" width="800">

<img src="https://github.com/julieanneco/Scope3_Emissions/blob/Photos/model_compare_2.png" alt="compare.key" width="500">


<!-- Conclusion -->
## Conclusion

**Analyzing MAE and Model Performance**:

MAE measures the average magnitude of errors between predicted and actual values. It is measured with the same unit as the target variable (CO2). Standard Deviation (σ) represents the natural variability and spread of the data and is used to indicate how much actual values deviate from the mean. In general, the higher the STD, the harder it is to predict a target. While the STD was dramatically reduced in the previous outlier and normalization process, the STD remained high across emission types and primary activities. A quick review of the plot outputs allows you to visually see how close each prediction was to the actual observation. The MAE metric for each emissions type is the average of all predictions for every observation within that category.

For both models, the MAE is significantly lower than the standard deviation, which indicates pretty decent prediction performance, but only in comparison to natural variation, especially considering the extreme variability of the data and how spread-out it is. It could likely be improved with more outlier removal and normalization since the STD is sensitive to outliers. Unfortunately, the features didn't offer a lot of strong correlation, so it is likely more the time-series trend that offered the most prediction power, but deeper analysis would help better understand feature correlation and importance for an improved model. The lowest possible level of detail in this dataset is the Primary Activity, which is categorical and requires encoding for use in most machine learning models. Features that offer deeper level of detail and are directly correlated to emission amount would vastly improve prediction strength. As an example, within the Business travel emission type, knowing the size of the company's vehicle fleet or the amount of annual airplane travel would certainly offer more information for the model to use. 

Comparing XGBoost to Random Forest, XGBoost seems to perform slightly better with an average MAE/STD ratio of .34 and is potentially a better suited model in this specific scenario. The ratios indicate the model has learned meaningful patterns in the data.

**Outcome**:

- The model's predictions are more accurate than what you'd expect from the natural variation in the data
- The model has successfully captured some underlying patterns beyond random variation
- The predictions are more reliable than simply using the mean value
- The model is performing well relative to the inherent variability and spread in the data

**Further Research and Testing on Highly Skewed Data**:

1. Benchmarking against XGBoost and Random Forest
2. Enhanced Feature Engineering - Dividing Emission Amount by Total Liabilities (Highest correlation), techniques to predict targets without time series evaluation
3. Industry validation, especially on extreme outliers
4. Continued normalization, skew transformation, and outlier analysis with possible imputation
5. Testing models on nulls, testing other ensemble approaches and non-parametric methods
6. Testing bootstrapping and resampling to ensure robust statistical inference, as well as potential techniques, such as cost-sensitive learning or SMOTE (Synthetic Minority Over-sampling Technique)
7. Testing Deep Learning and Neural Networks:
    - More complex pattern recognition (identify noin-linear relationships, adapt to changing patterns)
    - Better time series prediction
    - Better handling of high-dimensional data with automatic feature extraction
    - Better incorporate categorical data like Primary activity or Sector
<br />

<!-- Acknowledgements -->
## Acknowledgements

### Data 
* [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators)
* Scope 3 Emissions Data: Grant Funded Data for Research at Regis University

### Python Packages Utilized
* [wbgapi](https://www.rdocumentation.org/packages/WDI/versions/2.7.1)
* pandas
* numpy
* matplotlib
* seaborn
* scipy stats
* plotly express
* xgboost
* sklearn
