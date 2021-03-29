 # Kaggle - House Prices



## " Stacked Regressions to predict House Prices " Notebooks 정리





- **불필요한 컬럼 삭제** : ID 컬럼 삭제

```python
# ID 컬럼 삭제
train.drop('Id', axis = 1, inplace = True)
```





## Data Processing





### outliers

- **이상치 제거** : grlivarea가 넓으나 salepirce가 낮은 이상치 데이터 제거

  ( 단, 이상치 제거는 항상 안전 한 것은 아니다. 테스트 데이터에 이상치가 있는 경우, 모두 제거하면 모델에 나쁜 영향 미칠 수 있기에 일부만 제거 )

```python
# 하나의 설명변수와 종속변수 간의 산점도 확인
fig, ax = plt.subplots()
ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
plt.show()
```





### target variable

- **종속변수 분포 확인 및 정규화 진행** : 로그 변환, 박스콕스 변환

```python
# 로그 변환 후 분포 확인
train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'] , fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.title('SalePrice distribution')

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```





### Features engineering

- **결측치 확인**

```python
# 각 변수 별 결측치 비율 확인
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
```



- **상관계수 확인**

```python
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
sns.heatmap(corrmat, annot=True, fmt='d', linewidth=1)
```



- **결측치 처리**

  : 수치형, 범주형 변수 특성에 맞춰서 결측치 처리



- **more features**

  - 수치형이나 의미상 명목형인 경우 타입 변경

  - 순서 집합인 범주형 변수는 label encoding

  - 파생변수 생성

  - 설명변수 왜도 확인 : 지나치게 왜곡된 피처 존재할 경우 예측 성능 저하                      

    ​                                  : 0.75 이상이라면 박스콕스 정규 변환

``` python
# 수치형 => 명목형 타입 변경
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

# 범주형 => 수치형 타입 변경
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))
    
# 설명변수(수치형) 왜도 확인 후 0.75 이상이면 박스콕스 정규 변환
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index

lam = 0.15
for feat in skewed_features:
     all_data[feat] = boxcox1p(all_data[feat], lam)

```



- **더미화**

```python
all_data = pd.get_dummies(all_data)
```





## Modeling





### base models

- Lasso
- ElasticNet
- KernelRidge
- GradientBoostingRegressor
- XGBRegressor
- LGBRegressor





### stacking models

- simple stacking : averaging base models
- less simple stacking : adding a meta model





### ensembling stackedregressor

- ensemble = stacked_pred\*0.70 + xgb_pred\*0.15 + lgb_pred\*0.15

