# [ML] How to Accurately Predict Home Prices? - Real Estate (Brooklyn)

<img width="1200" height="630" alt="image" src="https://github.com/user-attachments/assets/d718296c-56c6-432d-a9cd-c39c34bc781b" />

## 1) Introduction
The New York City real estate market is rapidly heating up, making it increasingly difficult for homebuyers to assess the fair market value of properties. Brooklyn, in particular, has experienced a significant rise in housing prices in recent years due to growing demand. This trend has created an urgent need for a reliable method to estimate home values accurately.

To address this challenge, our project aims to develop a machine learning model that can accurately predict the sale price of homes in Brooklyn. This model will serve as a valuable tool for homebuyers, sellers, and real estate agents, enabling them to make more informed and confident decisions.

The project focuses on building a robust, data-driven pricing prediction system. We'll approach the problem step by step, ensuring clean, realistic data is used to train the model for maximum accuracy.

A key advantage of this project is its ability to serve two types of clients:

* Homebuyers: These clients are searching for their ideal home at a fair price. They have preferred neighborhoods in mind and want to know whether the listed price reflects the home’s true value. This tool will help them understand which features (e.g., number of bathrooms, location, square footage) most influence price, ensuring they make sound investments.

* Homesellers (e.g., House-Flippers): These clients aim to buy undervalued properties and increase their resale value through strategic improvements. By identifying the features that most impact pricing, sellers can focus their investments (e.g., adding rooms or enhancing layout) to maximize profit.

In short, this project provides a practical, data-driven solution to one of the most pressing problems in Brooklyn’s real estate market: understanding what a home is truly worth.

## 2) Dataset
[Link to dataset](https://drive.google.com/drive/folders/1pLwJg3M3R9kbo5BcRVpbcBjYcMKW4hUT?usp=sharing)

* The dataset contains housing prices in Brooklyn from 2003–2017, and the features have been labeled to represent relevant information.

* The features with influence include:

  - neighborhood: the surrounding area or district
  - tax_class: tax classification
  - block: The tax block in which the tax lot is located
  - lot: The number of the tax lot.
  - residential_units: means a home, apartment, residential condominium unit or mobile home, serving as the principal place of residence.
  - AssessLand: The assessed land value for the tax lot.
  - ExemptLand: Tax-free land
  - ExemptTot: The exempt total value, which is determined differently for each exemption program, is the dollar amount related to that portion of the tax lot that has received an exemption.
  - HistDist: The name of the Historic District that the tax lot is within. Historic Districts are designated by the New York City Landmarks Preservation Commission.
  - BuiltFAR: the total floor area ratio, calculated as the total building floor area divided by the tax lot area.

More details:
https://www1.nyc.gov/assets/planning/download/pdf/data-maps/open-data/pluto_datadictionary.pdf?v=18v1

## 3) Import libraries

![image](https://github.com/user-attachments/assets/3fd7ee2d-78e7-4161-863e-ca0c53d3ab58)

## 4) Imports and Data Load
We make arrangement so that we can see the entire dataframe.

![image](https://github.com/user-attachments/assets/2e7f2f69-4237-4ef4-96dc-2006df6a40ac)

![image](https://github.com/user-attachments/assets/a9540147-dd35-4e9b-9762-68620bfbe83a)

![image](https://github.com/user-attachments/assets/52536349-1cdc-4f1e-8ab4-415908ab5e4f)

We describe the data and append in it the number of NaN's in each column in the last row. 

👉 We see that there are many columns with NaNs. Our next steps will be to clean this data as much as possible.

![image](https://github.com/user-attachments/assets/5811abfb-5b91-4275-90c6-ff954eb051d0)

![image](https://github.com/user-attachments/assets/4d23c29c-028b-4028-aa48-09a75b5121c9)

## 5) Data processing

A big simplification can be done by removing duplicate columns. Also there are some coumns refering various map ID or file numbers which supposedly have no impact on any of the predictions made. We can easily remove 59 columns on various grounds.



1.   'Unnamed: 0' It's the ID number associated with each sale.

1.   We remove both 'borough' & 'Borough' since we are working on Brooklyn borough only.

1.   'apartment_number' about 80% missing data

1.   'Ext' EXTENSION CODE

1.   'Landmark' we delete this because we also have address and neighbourhood to specify the importance of the location

1.   'AreaSource' A code indicating the source file that was used to determine the tax lot's TOTAL BUILDING FLOOR AREA (BldgArea)

1.   'UnitsRes' The sum of residential units in all buildings on the tax lot. Same as 'residential_units'

1.   'UnitsTotal' The sum of residential and non-residential (offices, retail stores, etc.) units in all buildings on the tax lot. Same as 'total_units'.

1.   'LotArea' Total area of the tax lot, expressed in square feet rounded to the nearest integer. This is same as 'lot'

1.   'BldgArea' The total gross area in square feet. Same as 'gross_sqft'

1.   'BldgClass' Same as 'building_class'

1.   'Easements', 'easement' we delete both. The number of easements on the tax lot. As this is not important for most of the sales.

1.   'OwnerType' A code indicating type of ownership for the tax lot

1.   'building_class_category' same as 'building_class'

1.   'sale_date' date of sale but we keep only the year in 'year_of_sale'

1.   'CT2010' The 2010 census tract that the tax lot is located in. Not considered to be important.

1.   'CB2010' The 2010 census block that the tax lot is located in. Not considered to be important.

1.   'ZipCode' same as 'zip_code'

1.   'ZoneDist1' The zoning district classification of the tax lot, 

     - ZONING DISTRICT 1 represents the zoning district classification occupying the greatest percentage of the tax lot’s area.

     - 'ZoneDist2' If the tax lot is divided by zoning boundary lines,Zoning, ZONING DISTRICT 2 represents the zoning classification occupying the second greatest percentage of the tax lot's area.

      - 'ZoneDist3' If the tax lot is divided by zoning boundary lines, ZONING, ZONING DISTRICT 3 represents the zoning classification occupying the third greatest percentage of the tax lot's area.

      - 'ZoneDist4' If the tax lot is divided by zoning boundary lines, Zoning, ZONING DISTRICT 4 represents the zoning classification occupying the fourth greatest percentage of the tax lot's area.

1.   'Overlay1' The commercial overlay assigned to the tax lot. 'Overlay2' A commercial overlay associated with the tax lot.

1.   'SPDist1' The special purpose district assigned to the tax lot.

     - SPECIAL PURPOSE DISTRICT 1 represents the special purpose district occupying the greatest percentage of the lot area.

     - 'SPDist2' SPECIAL PURPOSE DISTRICT 2 represents the special purpose district occupying the second greatest percentage of the lot area.

     - 'SPDist3' SPECIAL PURPOSE DISTRICT 3 represents the special purpose district occupying the smallest percentage of the lot area.

1.   'LtdHeight' Limited height districts are coded using the three to five character district symbols

1.   'YearBuilt' same as 'year_built'

1.   'BoroCode' same as 'Borough'

1.   'BBL' A concatenation of the borough code, tax block and tax lot.

1.   'Tract2010' The 2010 census tract that the tax lot is located in.

1.   'ZoneMap' The Department of City Planning Zoning Map Number associated with the tax lot’s X and Y Coordinates.

1.   'ZMCode' A code (Y) identifies a border Tax Lot, i.e., a Tax Lot on the border of two or more Zoning Maps.

1.   'Sanborn' The Sanborn Map Company map number associated with the tax block and lot.

1.   'TaxMap' The Department of Finance paper tax map Volume Number associated with the tax block and lot.

1.   'EDesigNum' The E-Designation number assigned to the tax lot.

1.   'PLUTOMapID' A code indicating whether the tax lot is in the PLUTO file and/or the modified DTM and/or the modified DTM Clipped to the Shoreline File.

1.   'FIRM07_FLA' A one character field. Code of 1 means that some portion of the tax lot falls within the 1% annual chance floodplain as determined by FEMA’s 2007 Flood Insurance Rate Map.

1.   'PFIRM15_FL' A one character field. Code of 1 means that some portion of the tax lot falls within the 1% annual chance floodplain as determined by FEMA’s 2015 Preliminary Flood Insurance Rate Map.

1.   'Version' The Version Number related to the release of PLUTO.

1.   'MAPPLUTO_F' No description found.

1.   'APPBBL' The originating Borough, Tax Block and Tax Lot from the apportionment prior to the merge, split or property’s conversion to a condominium. The Apportionment BBL is only available for mergers, splits and conversions since 1984.

1.   'APPDate' The date of the Apportionment.

1.   'SHAPE_Leng', 'SHAPE_Area no description of both so we drop

1.   'CD' The community district (CD) or joint interest area (JIA) that the tax lot is located in, or partially located in.

1.   'SchoolDist' The community school district that the tax lot is located in.

1.   'Council' The city council district that the tax lot is located in.

1.   'PolicePrct' The police precinct the tax lot is located in. This field contains a three digit police precinct number.

1.   'HealthCent' The health center district that the tax lot is located in.

1.   'SanitBoro' The Boro of the Sanitation District that services the tax lot.

1.   'SanitDistr' The Sanitation District that services the tax lot.

1.   'FireComp' The fire company that services the tax lot.

1.   'SanitSub' The Subsection of the Sanitation District that services the tax lot.

1.   'CondoNo' The condominium number assigned to the complex.

1.   'Address' same as 'address'

axis = 1 -> by column

axis = 0 -> by row

![image](https://github.com/user-attachments/assets/b55408a4-66f5-4855-a1d8-19f4b25bb250)

![image](https://github.com/user-attachments/assets/a1770427-b17e-4b51-8a5f-7c670e45a590)

We only consider rows for which the sale price is non zero. As zero sale price is associated with a transfer of property so we safely delete those.

We have replaced zeros/NaNs in many variables with median/mode.

Some other categorical variables were filled by 0 like 'OwnerName', 'IrrLotCode', 'SplitZone'. For 'XCoord' and 'YCoord' we have replaced the NaN and 0s by Mode(0). Same is for variables 'YearAlter1' and 'YearAlter2'. Mostly for continuous variables we have replaced the missing values by median/mean.

Mode: is the value that appears most often in the set.

Std: Calculate standard deviation.

```ruby
df_house = df_house[df_house['sale_price']!=0]

df_house['gross_sqft']=df_house['gross_sqft'].replace(0.0,df_house['gross_sqft'].median())
df_house['land_sqft']=df_house['land_sqft'].replace(0.0,df_house['land_sqft'].median())

df_house['NumBldgs']= df_house['NumBldgs'].fillna(df_house['NumBldgs'].median())
df_house['NumFloors']= df_house['NumFloors'].fillna(df_house['NumFloors'].median())
df_house['ProxCode']= df_house['ProxCode'].fillna(df_house['ProxCode'].mode()[0])
df_house['LotType']= df_house['LotType'].fillna(df_house['LotType'].mode()[0])
df_house['BsmtCode']= df_house['BsmtCode'].fillna(df_house['BsmtCode'].mode()[0])
df_house['LandUse']= df_house['LandUse'].fillna(df_house['LandUse'].mode()[0])
df_house['AssessLand']= df_house['AssessLand'].fillna(df_house['AssessLand'].median())
df_house['AssessTot']= df_house['AssessTot'].fillna(df_house['AssessTot'].median())
df_house['ExemptLand']= df_house['ExemptLand'].fillna(df_house['ExemptLand'].median())
df_house['ExemptTot']= df_house['ExemptTot'].fillna(df_house['ExemptTot'].median())
df_house['BuiltFAR']= df_house['BuiltFAR'].fillna(df_house['BuiltFAR'].median())
df_house['ResidFAR']= df_house['ResidFAR'].fillna(df_house['ResidFAR'].median())
df_house['CommFAR']= df_house['CommFAR'].fillna(df_house['CommFAR'].median())
df_house['FacilFAR']= df_house['FacilFAR'].fillna(df_house['FacilFAR'].mean())
df_house['OwnerName']= df_house['OwnerName'].fillna(value=0)
df_house['IrrLotCode']= df_house['IrrLotCode'].fillna(value=0)
df_house['SplitZone']= df_house['SplitZone'].fillna(value=0)
df_house['XCoord']= df_house['XCoord'].fillna(df_house['XCoord'].mode()[0])
df_house['YCoord']= df_house['YCoord'].fillna(df_house['YCoord'].mode()[0])
df_house['XCoord']= df_house['XCoord'].replace(0.0,df_house['XCoord'].mode()[0] )
df_house['YCoord']= df_house['YCoord'].replace(0.0,df_house['YCoord'].mode()[0] )

df_house['ComArea']= df_house['ComArea'].fillna(df_house['ComArea'].median())
df_house['ResArea']= df_house['ResArea'].fillna(df_house['ResArea'].median())
df_house['OfficeArea']= df_house['OfficeArea'].fillna(df_house['OfficeArea'].median())
df_house['RetailArea']= df_house['RetailArea'].fillna(df_house['RetailArea'].median())
df_house['GarageArea']= df_house['GarageArea'].fillna(df_house['GarageArea'].median())
df_house['OtherArea']= df_house['OtherArea'].fillna(df_house['OtherArea'].median())
df_house['StrgeArea']= df_house['StrgeArea'].fillna(df_house['StrgeArea'].median())
df_house['FactryArea']= df_house['FactryArea'].fillna(df_house['FactryArea'].median())
df_house['LotFront']= df_house['LotFront'].fillna(df_house['LotFront'].median())
df_house['LotDepth']= df_house['LotDepth'].fillna(df_house['LotDepth'].median())
df_house['BldgFront']= df_house['BldgFront'].fillna(df_house['BldgFront'].median())
df_house['BldgDepth']= df_house['BldgDepth'].fillna(df_house['BldgDepth'].median())
df_house['HealthArea']= df_house['HealthArea'].fillna(df_house['HealthArea'].median())
df_house['YearAlter1']= df_house['YearAlter1'].fillna(df_house['YearAlter1'].mode()[0])
df_house['YearAlter2']= df_house['YearAlter2'].fillna(df_house['YearAlter2'].mode()[0])
```

✔️ The categorical variables are categorized.

a.astype (new type): Converts the data type of the elements

Unique: checks how many different elements there are in the data set

![image](https://github.com/user-attachments/assets/4a02b39e-d8cf-42ce-ac0f-9d7b080553fb)

![image](https://github.com/user-attachments/assets/c54a6577-a4fa-472a-9d8a-77f15a034d88)

✍️ We split the 'address' into number and 'street name'. Convert the 'street name' to categorical variable and delete the number column along with 'address'.

![image](https://github.com/user-attachments/assets/a81015ef-115d-4741-9619-f0c82b77cc4a)

👉🏼 We see the missing values in 'tax_class' and replace them by the corresponding values from 'tax_class_at_sale'. The convert 'tax_class' to categorical. Same is done for the missing values in 'building_class' by replacing the missing values from 'building_class_at_sale'.

map: Perform conversion to integer

![image](https://github.com/user-attachments/assets/86697b45-86f8-40e0-93f9-af5e4328e9b8)

![image](https://github.com/user-attachments/assets/94b62498-e118-4cb8-ab7e-4f6d5f812052)

👉🏽 We categorize several other variables 'OwnerName', 'IrrLotCode', 'SplitZone'

![image](https://github.com/user-attachments/assets/e2d189dc-8467-4d86-b254-2e2456e35aab)

![image](https://github.com/user-attachments/assets/344f5bed-fd34-4491-8bc1-73eb66369459)

![image](https://github.com/user-attachments/assets/089e8506-1f48-4828-99c5-d0c9209eb6fa)

✔️ We have thus removed all the null values from the dataframe

![image](https://github.com/user-attachments/assets/e870255e-43df-491c-be95-604045357fe8)

We locate and remove all the outliers in each column. Which makes the data more streamlined.

gross_sqft = Gross Square Feet is the total area of enclosed space measured to the exterior walls of a building.

![image](https://github.com/user-attachments/assets/3b8a9d1a-4cde-4bce-b6f1-da7ca4c17984)

![image](https://github.com/user-attachments/assets/5b93b473-afe8-47e0-95c9-ac1a0a656a77)

![image](https://github.com/user-attachments/assets/a3563ac0-baa3-4d8b-9fdd-9a7656b7e21e)

👉🏻 Next we remove variables those are highly correlated by evaluating their Variable Inflation Factor. We remove one variable each step and again calculate the VIF and then remove the next. We remove all variables with VIF>5 by this way to minimize correlation among the variables. We show a few and the rest are done in a similar way.

* Variance inflation factor (VIF) is a measure of the amount of multicollinearity in a set of multiple regression variables. Mathematically, the VIF for a regression model variable is equal to the ratio of the overall model variance to the variance of a model that includes only that single independent variable. This ratio is calculated for each independent variable. A high VIF indicates that the associated independent variable is highly collinear with the other variables in the model.

```ruby
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
```

X: vector/matrix used to represent the features of the data.
* the first dimension: number of elements of the training dataset (number of training data samples)
* the remaining dimensions: features

y: desired data label (index of target name)

![image](https://github.com/user-attachments/assets/80ef0e0f-f1c2-4b30-ad32-9e870bf8ad6a)

![image](https://github.com/user-attachments/assets/a2324a9a-dfeb-4d6f-8b0c-99a24e17fb01)

👉🏻 It should continue this way resulting in the deletion of the following variables.

![image](https://github.com/user-attachments/assets/d17639e5-fec7-4a96-be53-e5433bb405c7)

* neighborhood: surrounding area

* tax_class: tax class

* block: The tax block in which the tax lot is located

* lot: The number of the tax lot.

* residential_units: means a home, apartment, residential condominium unit or mobile home, serving as the principal place of residence.

* AssessLand: The assessed land value for the tax lot.

* ExemptLand: Tax-free land

* ExemptTot: The exempt total value, which is determined differently for each exemption program, is the dollar amount related to that portion of the tax lot that has received an exemption.

* HistDist: The name of the Historic District that the tax lot is within. Historic Districts are designated by the New York City Landmarks Preservation Commission.

* BuiltFAR: total construction floor area divided by tax lot area.

![image](https://github.com/user-attachments/assets/c52345fc-3e3c-4b29-9821-91e20a699a72)

Visualising Numeric Variables

![image](https://github.com/user-attachments/assets/76002deb-9819-47cf-ba90-9cb4847eed96)

![image](https://github.com/user-attachments/assets/dfceae2b-ac02-4e66-864b-30be7f812757)

![image](https://github.com/user-attachments/assets/e5a50e75-0dce-410c-9e88-e7de0c157b8f)

![image](https://github.com/user-attachments/assets/f1c2c439-d7cd-47e5-82cf-92abd7c4d07e)

## 6) Build model

```ruby
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
```

* random_state : The result is the same every time

* r2 : evaluate the accuracy of the model, the closer r2 is to 1 the better

```ruby
# Split data
X = df_house.drop('sale_price',axis=1)
y = df_house['sale_price']
Xtrn, Xtest, Ytrn, Ytest = train_test_split(X,y,test_size=0.3, random_state=42)

# Define models
models = [LinearRegression(), 
          linear_model.Lasso(alpha=0.1), 
          Ridge(alpha=100.0),
          #DecisionTreeClassifier(random_state=0),
          RandomForestRegressor(n_estimators=100, max_features='sqrt'), 
          KNeighborsRegressor(n_neighbors=6), # số láng giềng bằng 6
          DecisionTreeRegressor(max_depth=4), 
          ensemble.GradientBoostingRegressor()]

# Evaluate models
results = []

for model in models:
    print(model)
    model_name = model.__class__.__name__
    model.fit(Xtrn, Ytrn)
    
    r2 = r2_score(Ytest, model.predict(Xtest))
    
    print('Score on training:', model.score(Xtrn, Ytrn))
    print('R² score on test:', r2)
    print('\n')
    
    results.append({'Model': model_name, 'R2_Price': r2})

# Create results DataFrame
TestModels = pd.DataFrame(results).set_index('Model')
TestModels
```

![image](https://github.com/user-attachments/assets/9bece2a0-b714-4fd9-950f-dc3711ae70af)

![image](https://github.com/user-attachments/assets/a9691772-2e84-4455-be60-07a8946b45ef)

We see that GradientBoostingRegressor is the best model for us as scores on both training and testing sets are close, indicating no overfitting. So we choose it and apply grid search CV to find out best set of parameters. We dont run this step as it is really time consuming. We state the final result and use it in prediction.

## 7) Prediction
```ruby
feature_cols =['neighborhood', 'tax_class', 'block', 'lot', 'residential_units', 'AssessLand', 'ExemptLand', 'ExemptTot', 'HistDist', 'BuiltFAR']
target=['sale_price']

# dropna() : Remove observations with missing values
X = df_house[feature_cols].dropna()

# ravel is used to convert Matrix to Vector
y = np.array(df_house[target].dropna()).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# use linear regression as the model
model = ensemble.GradientBoostingRegressor()
model.fit(X_train, y_train)

# Plot feature importance
feature_importance = model.feature_importances_

# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
```

![image](https://github.com/user-attachments/assets/0c8f70ac-ced8-4b59-bacc-ece7fb43543f)

* We do grid search to find out the best parameters for Gradient Boosting Regressor. Since there is a trade off between the parameters 'learning_rate' and 'n_estimators' we try to find out the best values for them. Other parameters are accepted at their default values.

* Evaluate the points on the grid to see which position will help the training model achieve the best results. => Helps find an effective parameter set

![image](https://github.com/user-attachments/assets/a83284a0-018f-4dfc-8c61-7e11a37bdf2b)

![image](https://github.com/user-attachments/assets/8e9197d6-f13d-4916-bb2b-78c24a947812)

![image](https://github.com/user-attachments/assets/8ae7b564-c039-4ecb-999e-c8591605704d)

![image](https://github.com/user-attachments/assets/812beff2-5dfc-48e3-bab9-2930ee6fc707)

```ruby
# Plot training deviance
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, model.train_score_, 'b-',label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-', label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations') # x-axis: features
plt.ylabel('Deviance') # y-axis: predict
```

![image](https://github.com/user-attachments/assets/efa7eba0-1f05-4454-b5d7-00f527c19702)

![image](https://github.com/user-attachments/assets/edcdb81c-fc9b-476e-9d51-5117fed94ccb)

## 8) Conclusion

After conducting a thorough analysis on the Brooklyn House dataset, we have a better understanding of the features that impact the sale price of a house. With the preprocessed and cleaned data, we trained a predictive model to estimate the sale price of a house in Brooklyn. This information can be valuable for real estate agents, investors and home buyers to make informed decisions.

