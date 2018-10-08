
# coding: utf-8

# # Introduction

# This will be an explanation of a dataset of properties of cars. The goal of the exploration is to use the information to predict the fuel efficiency (in miles per gallon). The dataset is contained in the file `data/auto-mpg.data` and a description of the data is in the file `data/auto-mpg.names`.

# Loading required libraries:

# In[84]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from math import floor
from os.path import join
from os import getcwd


# Save random seed for reproduceability

# In[85]:


random_seed = 42


# Graphics save directory

# In[86]:


imdir = join(getcwd(), 'report', 'images')


# # Load the data

# The data were loaded into a `pandas.DataFrame` object for ease of processing the mix of continuous and categorical data.

# In[87]:


column_names = [
    'mpg', 'cylinders', 'displacement',
    'horsepower', 'weight', 'acceleration',
    'model year', 'origin', 'car name'
]
df = pd.read_csv('data/auto-mpg.data', header=None,
                 names=column_names, na_values='?',
                 sep='\s+')


# In[88]:


df.describe()


# In[89]:


df.head()


# The dataset description calls out six missing values from the horsepower column. Since these records are relatively small compared to the rest of the dataset, for now I will delete the rows.

# In[90]:


df_clean = df.dropna(axis=0)


# In[91]:


df_clean.describe()


# Sure enough, 6 rows of data had NA values and have now been dropped.

# # Split the data for cross-validation

# Physically we would not expect that any individual car's properties should depend on any other car's properties. Therefore, if we want to divide the dataset to allow for **training**, **testing**, and **valdiation** data, we can simply randomly sample from the rows. I will use 50% of the data for training, 25% for validation, and 25% for testing and reporting.

# In[92]:


def train_test_val_split(data, train_frac=0.5, test_frac=0.25, val_frac=0.25, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    else:
        np.random.seed()

    # Normalize fractions
    tot = train_frac + test_frac + val_frac
    train_frac /= tot
    test_frac /= tot
    val_frac /= tot

    numrows = data.shape[0]

    rows = np.arange(numrows)
    np.random.shuffle(rows)

    train_row_count = floor(train_frac * numrows)
    test_row_count = floor(test_frac * numrows)
    val_row_count = floor(val_frac * numrows)

    train_test_boundary = train_row_count
    test_val_boundary = train_test_boundary + test_row_count
    val_train_boundary = test_val_boundary + val_row_count

    train_rows = rows[:train_test_boundary]
    test_rows = rows[train_test_boundary:test_val_boundary]
    val_rows = rows[test_val_boundary:val_train_boundary]

    # Add leftover rows to training
    np.append(train_rows, rows[val_train_boundary:])

    train = data.loc[data.index.intersection(train_rows)]
    test = data.loc[data.index.intersection(test_rows)]
    val = data.loc[data.index.intersection(val_rows)]

    return train, test, val


# In[93]:


train, test, val = train_test_val_split(df_clean, random_seed=random_seed)


# # Dataset description

# From the description of the dataset above (`df_clean.describe()`) we see some information fairly quickly. First, the `pandas` library is treating discrete values as continuous (luckily it was smart enough to ignore the 'car name' column).

# In[94]:


continuous_columns = ['mpg', 'displacement', 'horsepower',
                      'weight', 'acceleration']


# In[95]:


train_continuous = train[continuous_columns]
val_continuous = val[continuous_columns]


# # Initial regression

# First examine the performance of a regression using only the continuous values without rescaling the data columns.

# In[96]:


class LinearRegression():
    def __init__(self):
        self.coefficients_ = None
        self.means_ = None
        self.stds_ = None
        self.x_columns_ = None
        self.y_column_ = None
    
    def fit(self, data, x_columns, y_column):
        x = data[x_columns].values
        y = data[y_column].values
        
        self.x_columns_ = x_columns
        self.y_column_ = y_column
        
        xs = self.zscore(x)
        
        # One-pad xs
        
        
        self.coefficients_ = np.linalg.inv(xs.T @ xs) @ xs.T @ y
        
    def zscore(self, x):
        # Make sure x is 2-d
        if len(x.shape) < 2:
            x = x.reshape((-1, 1))
        
        if self.means_ is None and self.stds_ is None:
            self.means_ = np.mean(x, axis=0)
            self.stds_ = np.std(x, axis=0)
        
        xs = (x - self.means_) / self.stds_
        
        # Add column of ones
        o = np.ones_like(xs[:,0]).reshape((-1, 1))
        xs = np.hstack([xs, o])
        
        return xs
        
    def predict(self, data):
        x = data[self.x_columns_].values
        xs = self.zscore(x)
        
        y_pred = xs @ self.coefficients_
        
        return y_pred
    
    def rmse(self, data):
        yp = self.predict(data)
        yt = data[self.y_column_].values
        
        se = (yp - yt) ** 2
        mse = np.mean(se)
        rmse = np.sqrt(mse)
        
        return rmse


# In[97]:


lr = LinearRegression()
lr.fit(train, continuous_columns[1:], ['mpg'])


# In[98]:


sns.regplot(x=test['mpg'], 
            y=lr.predict(test).ravel())
plt.xlabel('mpg (actual)')
plt.ylabel('mpg (predicted)')
plt.show()


# A perfect fit would have a slope of one and an intercept of zero. Looks like more digging is necessary

# In[99]:


sns.pairplot(train_continuous)
plt.savefig(join(imdir, 'continuous_pairplot.png'), dpi=300)
plt.show()


# This shows us that the relationship between mpg and displacement, horsepower, and weight do not appear to be linear--it looks like either a $\mathrm{e}^{-x}$ or $1/x$ relationship. To validate, let's plot mpg vs. horsepower^-1 and mpg vs log(horsepower)

# In[100]:


f = plt.figure(figsize=(15, 8))

ax1 = f.add_subplot(121)
ax1.plot(1/train['horsepower'], train['mpg'], '.')
ax1.set_title('Inverse')
ax1.set_xlabel(r'horsepower$^{-1}$')
ax1.set_ylabel('mpg')

ax2 = f.add_subplot(122, sharey=ax1)
ax2.plot(np.log(train['horsepower']), train['mpg'], '.')
ax2.set_title('Exponential')
ax2.set_xlabel('log(horsepower)')

plt.savefig(join(imdir, 'inverse_and_exponential_horsepower.png'), dpi=300)
plt.show()


# These look more linear, but as a guess I am going to assume that the parameter to use is $1/horsepower$. Similarly, I will fit $1/displacement$ and $1/weight$

# In[101]:


for dataset in (train, test, val):
    dataset['inverse horsepower'] = dataset['horsepower'] ** -1
    dataset['inverse displacement'] = dataset['displacement'] ** -1
    dataset['inverse weight'] = dataset['weight'] ** -1
continuous_columns.extend(['inverse horsepower', 
                           'inverse displacement', 'inverse weight'])
train_continuous = train[continuous_columns]


# In[102]:


train.columns


# In[103]:


sns.pairplot(train_continuous)
plt.show()


# In[104]:


sns.pairplot(train)
plt.show()


# There appear to be some funny things going on in the "cylinders" column. These are typically paired, so odd numbers are unusual. Let's look at that.

# In[105]:


train[train.cylinders%2!=0]


# A wikipedia search for the [Mercedes 300D](https://en.wikipedia.org/wiki/Mercedes-Benz_W123) reveals that it was, in fact, offered in an inline-5 cylinder configuration.
# 
# On the other hand, the [Mazda RX-4](https://en.wikipedia.org/wiki/Mazda_Luce#Mazda_RX-4) was not offered in a 3-cylinder configuration. It's possible that this refers to the rotary Wankel engine. Cars with this engine were sold during this time period, which provides some coroborating evidence.

# In[106]:


f = plt.figure(figsize=(20,20))
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap='coolwarm', square=True, mask=mask)
plt.show()


# In[107]:


f = plt.figure(figsize=(20,20))
mask = np.zeros_like(train_continuous.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(train_continuous.corr(), vmin=-1, vmax=1, center=0, annot=True, cmap='coolwarm', square=True, mask=mask)
plt.show()


# From this wee can see that displacement, horsepower, and weight are negatively correlated with mpg. Counterintuitively, acceleration is positively correlated with mpg.

# Just for fun, let's try to figure out what the origin codes ("1", "2", and "3") correspond to

# In[108]:


train[train.origin==1].head()


# In[109]:


train[train.origin==2].head()


# In[110]:


train[train.origin==3].head()


# It appears that origin "1" is the United States, origin "2" is Europe, and origin "3" is Japan (or possibly Asia).

# There are two 6-cylinder cars that appear to be outliers in mpg. Let's take a look  and validate the maximum of those.

# In[111]:


train[train.cylinders==6].loc[train[train.cylinders==6]['mpg'].idxmax()]


# In fact, let's look at boxplots to examine the categorical attributes and determine 

# In[112]:


sns.boxplot(x='origin', y='mpg', data=train)
plt.savefig(join(imdir, 'mpg_vs_origin.png'), dpi=300)
plt.show()


# Here we can see that Japan has the highest gas mileage, America the lowest, and Europe somewhere in between. Based on these differences it is probably worthwhile to develop a category-aware model, or three separate models (one for each origin).

# In[113]:


sns.boxplot(x='cylinders', y='mpg', data=train)
plt.show()


# This shows the inverse trend of cylinder numbers to mpg. It also suggests that the 3- and 5-cylinder cars may not be representative of this overall trend and arguably should be thrown out.

# In[114]:


sns.boxplot(x='model year', y='mpg', data=train)
plt.savefig(join(imdir, 'mpg_vs_year.png'), dpi=300)
plt.show()


# Here we can see the trend of the quasi-continuous (but actually discrete) variable `model year` vs. gas mileage. There's no clean relationship here, which belies the correlation calculated above.

# It looks like 1978 has some high mpg examples--several sigma above the mean. Let's look more closely at those.

# In[115]:


train[train['model year'] == 78].sort_values('mpg')


# In[116]:


train[train['model year'] == 78].describe()


# Did the Ford Fiesta *really* get 36.1 mpg? While it is the lightest car of the year, this is also worth checking out and potentially rejecting. As it turns out, the car was [reported](https://www.cargurus.com/Cars/1978-Ford-Fiesta-Reviews-c9312) to get excellent gas mileage--this is probably a real data point.

# Based on this, it appears that outliers are not especially concerning.

# We should check to see how many American cars there are vice the other two origins. If the dataset is weighted it may be necessary to use a more advanced sampling procedure.

# In[117]:


len(train[train['origin'] == 1])


# In[118]:


len(train[train['origin'] == 2])


# In[119]:


len(train[train['origin'] == 3])


# Sure enough, this is unbalanced. This is an excellent argument for generating three different models.

# In[120]:


f = plt.figure(figsize=(20, 8))

f.add_subplot(131)
sns.distplot(train['model year'], kde=False)

f.add_subplot(132)
sns.distplot(train['origin'], kde=False)

f.add_subplot(133)
sns.distplot(train['cylinders'], kde=False)

f.suptitle('Distribution of categorical parameters in training dataset')

plt.savefig(join(imdir, 'categorical_distribution.png'), dpi=300)
plt.show()


# Distribution across model years number of cylinders is fairly even.

# # Summary

# - Split the origin of the cars by country
#     - Can make 3 models (one for each region)
#     - Can one-hot encode
# - Include year as an input, it's correlated 
# - Calculate the inverse of the weight, displacement, and horsepower
# - Cylinders is also strongly correlated
#     - Can use as input but throw out cars with odd numbers of cylinders
#     - Can one-hot encode
#     - Can ignore as input

# # Make a linear regression tool

# In[121]:


train_continuous.columns


# # Regression on one column

# In[122]:


one_var_regression = []
for column in train_continuous.columns[1:]:
    model = LinearRegression()
    model.fit(train, [column], ['mpg'])
    one_var_regression.append({
        'name': column,
        'model': model,
        'performance': model.rmse(test)
    })


# In[123]:


for m in one_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# So inverse horsepower is apparently the best single predictor, even though it was not the most strongly correlated.

# In[124]:


sns.regplot(x=one_var_regression[4]['model'].predict(test).ravel(), y=test['mpg'])
plt.xlabel(f"Regression on {one_var_regression[4]['name']}")
plt.show()


# ## Regression on two columns

# Now to find the best combination of two input columns

# In[125]:


from itertools import combinations


# In[126]:


cs = [list(_) for _ in combinations(continuous_columns[1:] + ['model year'], 2)]


# In[127]:


two_var_regression = []
for c in cs:
    model = LinearRegression()
    model.fit(train, c, ['mpg'])
    two_var_regression.append({
        'name': c,
        'model': model,
        'performance': model.rmse(test)
    })


# In[128]:


for m in two_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# The best two variable model appears to be a combination of `model year` and `inverse displacement`, with a test set RMSE of 3.12. It does appear that the model loses precision at higher mpg cars.

# In[129]:


sns.regplot(x=two_var_regression[-1]['model'].predict(test).ravel(), y=test['mpg'])
plt.xlabel(f"Regression on {two_var_regression[-2]['name']}")
plt.show()


# In[130]:


sns.regplot(x=two_var_regression[12]['model'].predict(test).ravel(), y=test['mpg'])
plt.xlabel(f"Regression on {two_var_regression[-1]['name']}")
plt.show()


# ## Three variable regression

# In[131]:


cs = [list(_) for _ in combinations(continuous_columns[1:] + ['model year'], 3)]
three_var_regression = []
for c in cs:
    model = LinearRegression()
    model.fit(train, c, ['mpg'])
    three_var_regression.append({
        'name': c,
        'model': model,
        'performance': model.rmse(test)
    })
for m in three_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# In[132]:


three_var_perf = [m['performance'] for m in three_var_regression]
three_var_best_model = three_var_regression[three_var_perf.index(min(three_var_perf))]
print(three_var_best_model)


# ## Performance of best model vs. input size

# In[133]:


input_columns = continuous_columns[1:] + ['model year']
best_model_perf = []
best_model_train_perf = []
for i in range(1,len(input_columns)+1):
    cs = [list(_) for _ in combinations(input_columns, i)]
    #print(cs)
    best_perf = 1000000
    best_train_perf = 1000000
    for c in cs:
        model = LinearRegression()
        model.fit(train, c, ['mpg'])
        best_perf = min([model.rmse(test), best_perf])
        best_train_perf = min([model.rmse(train), best_train_perf])
    best_model_perf.append(best_perf)
    best_model_train_perf.append(best_train_perf)
plt.plot(best_model_perf, label="Test set error")
plt.plot(best_model_train_perf, label="Training set error")
plt.plot(best_model_perf.index(min(best_model_perf)), 
         min(best_model_perf), 'ro', alpha=0.5)
plt.plot(3, best_model_perf[3], 'go', alpha=0.5)
plt.xlabel('Number of regression parameters')
plt.ylabel('Test set RMSE')
xticks = [_ for _ in range(len(best_model_perf))]
xtick_labels = [_+1 for _ in range(len(best_model_perf))]
plt.xticks(xticks, xtick_labels)
plt.legend(loc='upper right')
plt.savefig(join(imdir, 'performance_vs_model_size.png'), dpi=300)
plt.show()


# In[134]:


best_model_perf[3]


# In[135]:


best_model_perf[5]


# While it appears that the best model is made when six parameters are included, the performance of the 6-input model is not markedly better (on the test set) than the 4-input model. At this point I would select the 4-input model as the best based on its combination of accuracy and (presumably) higher generalizability.

# In[136]:


input_columns = continuous_columns[1:] + ['model year']
cs = [list(_) for _ in combinations(input_columns, 4)]
best_model_perf = []
models = []
#print(cs)
best_perf = 1000000
for c in cs:
    model = LinearRegression()
    model.fit(train, c, ['mpg'])
    best_perf = min([model.rmse(test), best_perf])
    best_model_perf.append(model.rmse(test))
    models.append(model)
print(f"The best model had an RMSE of {best_perf}")
index = best_model_perf.index(best_perf)
inputs = cs[index]
print(f"It had a performance of {best_perf}")
print(f"Its inputs were {inputs}")


# In[137]:


input_columns = continuous_columns[1:] + ['model year']
cs = [list(_) for _ in combinations(input_columns, 6)]
best_model_perf = []
models = []
#print(cs)
best_perf = 1000000
for c in cs:
    model = LinearRegression()
    model.fit(train, c, ['mpg'])
    best_perf = min([model.rmse(test), best_perf])
    best_model_perf.append(model.rmse(test))
    models.append(model)
print(f"The best model had an RMSE of {best_perf}")
index = best_model_perf.index(best_perf)
inputs = cs[index]
print(f"It had a performance of {best_perf}")
print(f"Its inputs were {inputs}")


# In[138]:


tmp = train[
    ['acceleration',
     'inverse horsepower',
     'inverse displacement',
     'model year']
].values
np.linalg.cond(tmp)


# In[139]:


tmp = train[
    ['displacement',
     'horsepower',
     'acceleration',
     'inverse horsepower',
     'inverse displacement',
     'model year']
].values
np.linalg.cond(
    tmp
)


# ## Regression on all continuous columns

# In[140]:


all_col_model = LinearRegression()
all_col_model.fit(train, continuous_columns[1:], ['mpg'])
all_col_model.rmse(test)


# In[141]:


sns.regplot(x=all_col_model.predict(test).ravel(), y=test['mpg'])
plt.xlabel(f"Regression on all continuous parameters")
plt.show()


# In[142]:


all_col_plus_year = LinearRegression()
all_col_plus_year.fit(train, continuous_columns[1:] + ['model year'], ['mpg'])
all_col_plus_year.rmse(test)


# In[143]:


sns.regplot(x=all_col_plus_year.predict(test).ravel(), y=test['mpg'])
plt.xlabel('Regression on all continuous parameters plus model year')
plt.show()


# ## Individual region models

# Repeat the analysis above but treat each country separately to try for better accuracy

# ### United States

# In[144]:


cs = [list(_) for _ in combinations(continuous_columns[1:] + ['model year'], 2)]
us_two_var_regression = []
for c in cs:
    model = LinearRegression()
    model.fit(train[train.origin == 1], c, ['mpg'])
    us_two_var_regression.append({
        'name': c,
        'model': model,
        'performance': model.rmse(test)
    })


# In[145]:


for m in us_two_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# Now the best fit model uses `inverse weight` and `model year`, with similar accuracy. Since `inverse displacement` and `inverse weight` are so correlated (0.93) this is not surprising.

# In[146]:


us_all_col_plus_year = LinearRegression()
us_all_col_plus_year.fit(train[train.origin == 1], continuous_columns[1:] + ['model year'], ['mpg'])
us_all_col_plus_year.rmse(test)


# ### Europe

# In[147]:


cs = [list(_) for _ in combinations(continuous_columns[1:] + ['model year'], 2)]
eu_two_var_regression = []
for c in cs:
    model = LinearRegression()
    model.fit(train[train.origin == 2], c, ['mpg'])
    eu_two_var_regression.append({
        'name': c,
        'model': model,
        'performance': model.rmse(test)
    })


# In[148]:


for m in eu_two_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# European car mileage is best predicted by `inverse weight` and `model year`

# In[149]:


eu_all_col_plus_year = LinearRegression()
eu_all_col_plus_year.fit(train[train.origin == 2], continuous_columns[1:] + ['model year'], ['mpg'])
eu_all_col_plus_year.rmse(test)


# ### Japan

# In[150]:


cs = [list(_) for _ in combinations(continuous_columns[1:] + ['model year'], 2)]
jp_two_var_regression = []
for c in cs:
    model = LinearRegression()
    model.fit(train[train.origin == 3], c, ['mpg'])
    jp_two_var_regression.append({
        'name': c,
        'model': model,
        'performance': model.rmse(test)
    })


# In[151]:


for m in jp_two_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# Japanese cars are like US cars in that they are also best predicted using `inverse weight` and `model year`

# In[152]:


jp_all_col_plus_year = LinearRegression()
jp_all_col_plus_year.fit(train[train.origin == 3], continuous_columns[1:] + ['model year'], ['mpg'])
jp_all_col_plus_year.rmse(test)


# # Unscaled regression

# In[153]:


class UnscaledLinearRegression():
    def __init__(self):
        self.coefficients_ = None
        self.means_ = None
        self.stds_ = None
        self.x_columns_ = None
        self.y_column_ = None
    
    def fit(self, data, x_columns, y_column):
        x = data[x_columns].values
        y = data[y_column].values
        
        self.x_columns_ = x_columns
        self.y_column_ = y_column
        
        xs = self.zscore(x)
        
        self.coefficients_ = np.linalg.inv(xs.T @ xs) @ xs.T @ y
        
    def zscore(self, x):
        # Make sure x is 2-d
        if len(x.shape) < 2:
            x = x.reshape((-1, 1))
        
        if self.means_ is None and self.stds_ is None:
            self.means_ = np.mean(x, axis=0)
            self.stds_ = np.std(x, axis=0)
        
        #xs = (x - self.means_) / self.stds_
        xs = x
        
        # Add column of ones
        o = np.ones_like(xs[:,0]).reshape((-1, 1))
        xs = np.hstack([xs, o])
        
        return xs
        
    def predict(self, data):
        x = data[self.x_columns_].values
        xs = self.zscore(x)
        
        y_pred = xs @ self.coefficients_
        
        return y_pred
    
    def rmse(self, data):
        yp = self.predict(data)
        yt = data[self.y_column_].values
        
        se = (yp - yt) ** 2
        mse = np.mean(se)
        rmse = np.sqrt(mse)
        
        return rmse


# In[154]:


unscaled_one_var_regression = []
for column in train_continuous.columns[1:]:
    model = UnscaledLinearRegression()
    model.fit(train, [column], ['mpg'])
    unscaled_one_var_regression.append({
        'name': column,
        'model': model,
        'performance': model.rmse(test)
    })
for m in one_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# Comparison of Scaled and Unscaled for two-variable regression. Positive values indicate that scaled is better; negative values indicate that unscaled is better

# In[155]:


one_var_scaled_perf = np.array([m['performance'] for m in one_var_regression])
one_var_unscaled_perf = np.array([m['performance'] for m in unscaled_one_var_regression])
print(one_var_scaled_perf - one_var_unscaled_perf)


# In[156]:


cs = [list(_) for _ in combinations(continuous_columns[1:] + ['model year'], 2)]
unscaled_two_var_regression = []
for c in cs:
    model = UnscaledLinearRegression()
    model.fit(train, c, ['mpg'])
    unscaled_two_var_regression.append({
        'name': c,
        'model': model,
        'performance': model.rmse(test)
    })
for m in unscaled_two_var_regression:
    print(f"{m['name']}\t{m['performance']}")


# In[157]:


two_var_scaled_perf = np.array([m['performance'] for m in two_var_regression])
two_var_unscaled_perf = np.array([m['performance'] for m in unscaled_two_var_regression])
print(one_var_scaled_perf - one_var_unscaled_perf)


# Since these regressions are linear, centering and scaling the data has no effect on the outcome!

# # Polynomial features

# In[158]:


class PolynomialRegressor(LinearRegression):
    def __init__(self, degree=1):
        super().__init__()
        self.degree_ = degree
        if degree < 1:
            raise ValueError(f"Degree must be 1 or greater, not {degree}")
    
    def make_poly_cols(self, data, x_cols):
        x_poly = data.copy()
        
        new_x_cols = x_cols[:]
        
        for i in range(2, self.degree_ + 1):
            for c in x_cols:
                colname = f"{c}^{i}"
                x_poly[colname] = x_poly[c] ** i
                new_x_cols.append(colname)
        
        return x_poly, new_x_cols
    
    def fit(self, data, x_cols, y_col):
        self.x_columns_ = x_cols
        self.y_column_ = y_col
        
        x, xc = self.make_poly_cols(data, x_cols)
        x = x[xc].values
        
        xs = self.zscore(x)
        
        y = data[y_col].values
        
        self.coefficients_ = np.linalg.inv(xs.T @ xs) @ xs.T @ y
    
    def predict(self, data):
        x, xc = self.make_poly_cols(data, self.x_columns_)
        x = x[xc].values
        
        xs = self.zscore(x)
        
        y_pred = xs @ self.coefficients_
        
        return y_pred


# In[159]:


# Verify that this is the same for the degree=1 case
pr = PolynomialRegressor(1)
lr = LinearRegression()
pr.fit(train, ['weight', 'model year'], ['mpg'])
lr.fit(train, ['weight', 'model year'], ['mpg'])
assert pr.rmse(train) == lr.rmse(train)


# In[160]:


best_model_vs_degree = []
for degree in range(1,6):
    input_columns = continuous_columns[1:] + ['model year']
    best_model_perf = []
    for i in range(1,len(input_columns)+1):
        cs = [list(_) for _ in combinations(input_columns, i)]
        #print(cs)
        best_perf = 1000000
        for c in cs:
            model = PolynomialRegressor(degree)
            model.fit(train, c, ['mpg'])
            best_perf = min([model.rmse(test), best_perf])
        best_model_perf.append(best_perf)
    best_model_vs_degree.append(best_model_perf)


# In[161]:


perf_array = np.array(best_model_vs_degree)
print(f"Error of best model: {np.min(perf_array)}")
print(f"Location of best model: {np.where(perf_array == np.min(perf_array))}")


# In[162]:


print(perf_array)


# It appears the most accurate model on the test set had five column inputs and was of degree three. 

# In[163]:


for i in range(perf_array.shape[0]):
    best_model_perf = perf_array[i]
    plt.plot(best_model_perf, label=f"Degree {i+1}")
    plt.xlabel('Number of regression parameters')
    plt.ylabel('Test set RMSE')
    xticks = [_ for _ in range(len(best_model_perf))]
    xtick_labels = [_+1 for _ in range(len(best_model_perf))]
    plt.legend(loc='upper right')
plt.plot(4, perf_array[2, 4], 'ro', alpha=0.5)
plt.plot(3, perf_array[2, 3], 'go', alpha=0.5)
plt.xticks(xticks, xtick_labels)
plt.savefig(join(imdir, 'polynomial_regression_performance.png'), dpi=300)
plt.show()


# In[164]:


input_columns = continuous_columns[1:] + ['model year']
best_model_perf = []
cs = [list(_) for _ in combinations(input_columns, 4)]
#print(cs)
best_perf = 1000000
best_model = None
for c in cs:
    model = PolynomialRegressor(3)
    model.fit(train, c, ['mpg'])
    best_perf = min([model.rmse(test), best_perf])
    if best_perf == model.rmse(test):
        best_model = model
best_model_perf.append(best_perf)
index = best_model_perf.index(min(best_model_perf))
print(f"The best model had inputs: {cs[index]}")


# In[165]:


sns.regplot(test[('mpg')], best_model.predict(test).ravel())
plt.show()


# In[166]:


best_model = LinearRegression()
best_model.fit(train, ['acceleration', 'inverse horsepower', 'inverse displacement', 'model year'], ['mpg'])
best_model.rmse(val)

