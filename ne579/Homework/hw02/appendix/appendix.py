#!/usr/bin/env python
# coding: utf-8

# # NE 579 Homework Number 2: Data Statistics

# ## Import required libraries

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.io as sio
from scipy.stats import trim_mean, skew, kurtosis, norm


# In[2]:


sns.color_palette('coolwarm')


# ## Load the data from the file

# In[3]:


fn = 'hwkdata.mat'
data_dict = sio.loadmat(fn)
print(data_dict.keys())


# In[4]:


data_dict['x'].shape


# In[5]:


data_dict['y'].shape


# In[6]:


data = np.append(data_dict['x'], data_dict['y'], 1)


# In[7]:


data.shape


# Now the 2-d array `data` contains 252 samples, each with 14 input variables and 1 output variable

# I will want to label the columns later...

# In[8]:


parameter_names = [
    'Age',
    'Weight',
    'Height',
    'Adiposity Index',
    'Neck circumference',
    'Chest circumference',
    'Abdomen circumference',
    'Hip circumference',
    'Thigh circumference',
    'Knee circumference',
    'Ankle circumference',
    'Extended bicep circumference',
    'Forearm circumference',
    'Wrist circumference',
    '% Bodyweight'
]


# ## Calculate statistical properties of the data:
# - Maximum
# - Minimum
# - Mean
# - Median
# - 20% trimmed mean
# - Standard deviation
# - Variance
# - Skewness
# - Kurtosis

# ### Maximum

# In[9]:


np.max(data, axis=0)


# ### Minimum

# In[10]:


np.min(data, axis=0)


# ### Mean

# In[11]:


np.mean(data, axis=0)


# ### Median

# In[12]:


np.median(data, axis=0)


# ### 20% Trimmed Mean

# In[13]:


trim_mean(data, 0.2, axis=0)


# ### Standard Deviation

# In[14]:


np.std(data, axis=0)


# ### Variance

# In[15]:


np.var(data, axis=0)


# ### Skewness

# In[16]:


skew(data, axis=0)


# ### Kurtosis

# In[17]:


kurtosis(data, axis=0)


# #### Also calculate $Kurt - 3$

# In[18]:


kurtosis(data, axis=0) - 3 * np.ones_like(kurtosis(data, axis=0))


# #### Difference between mean and median

# In[70]:


with sns.color_palette('coolwarm'):
    f = plt.figure()
    s = np.divide(np.mean(data, axis=0) - np.median(data, axis=0), np.std(data, axis=0))
    ax = plt.subplot(121)
    plt.bar(np.arange(s.shape[0]), s, label=r'$\mu$ - median')
    #plt.bar(np.arange(s.shape[0]), np.mean(data, axis=0) - np.median(data, axis=0), label=r'$\mu$ - median')
    #plt.bar(np.arange(s.shape[0]), kurtosis(data, axis=0) / kurtosis(data, axis=0).max(axis=0),
    #        alpha=0.5, label='Kurtosis')
    #plt.bar(np.arange(s.shape[0]), skew(data, axis=0),
    #        alpha=0.5, label='Skewness')
    plt.legend(loc="upper right")
    plt.savefig('images/mu_minus_median.png', dpi=300, bbox_inches='tight')
    plt.show()


# From this plot we see that the mean (sensitive to outliers) exceeds the median (insensitive to outliers) 

# In[69]:


with sns.color_palette('coolwarm'):
    plt.bar(np.arange(np.mean(data, axis=0).shape[0]),
        np.divide(np.mean(data, axis=0) - trim_mean(data, 0.2, axis=0), np.std(data, axis=0)), label=r'$\mu$ - trim mean')
    plt.legend(loc='upper right')
    #plt.bar(np.arange(trim_mean(data, 0.2, axis=0).shape[0]),
    #        trim_mean(data, 0.2, axis=0))
    plt.savefig('images/mu_minus_trimmed_mean.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[85]:


with sns.color_palette('coolwarm'):
    f = plt.figure()
    ax1 = f.add_subplot(121)
    ax1.bar(np.arange(np.mean(data, axis=0).shape[0]),
            np.divide(np.mean(data, axis=0) - trim_mean(data, 0.2, axis=0),
                      np.std(data, axis=0)), 
            label=r'$\mu$ - trim mean')
    ax1.legend(loc='lower center')
    
    ax2 = f.add_subplot(122, sharey=ax1)
    s = np.divide(np.mean(data, axis=0) - np.median(data, axis=0), np.std(data, axis=0))
    ax2.bar(np.arange(s.shape[0]), s, label=r'$\mu$ - median')
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.legend(loc="lower center")
    
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    ax1.set_xlabel('Parameter number')
    ax2.set_xlabel('Parameter number')
    
    plt.savefig('images/robust_statistics.png', dpi=300, bbox_inches='tight')
    


# 
# # Now look at covariance and correlation

# In[103]:


plt.figure(figsize=(20,20))
sns.heatmap(np.corrcoef(data.T), vmin=-1, vmax=1, center=0, cmap='coolwarm',
            square=True, annot=True)
plt.savefig('images/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# ## ...and covariance

# In[108]:


plt.figure(figsize=(20,20))
sns.heatmap(np.cov(data.T), cmap='Blues',
            square=True, annot=True)
plt.savefig('images/covariance_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# ## Distribution of covariance

# In[120]:


with sns.color_palette('coolwarm'):
    plt.hist(np.cov(data.T).flatten(), bins=50)
    plt.xlabel('Covariance')
    plt.savefig('images/covariance_dist.png', dpi=300, bbox_inches='tight')
    plt.show()


# # Pandas

# In[23]:


stat_names = [
    "Parameter", "Max", "Min", "Mean", "Median", "20% Trimmed Mean",
    "Standard Deviation", "Variance", "Skewness", "Kurtosis"
]


# In[24]:


summary_statistics = pd.DataFrame(columns=stat_names)
summary_statistics['Parameter'] = parameter_names
summary_statistics['Max'] = np.max(data, axis=0)
summary_statistics['Min'] = np.min(data, axis=0)
summary_statistics['Mean'] = np.mean(data, axis=0)
summary_statistics['Median'] = np.median(data, axis=0)
summary_statistics['20% Trimmed Mean'] = trim_mean(data, 0.2, axis=0)
summary_statistics['Standard Deviation'] = np.std(data, axis=0)
summary_statistics['Variance'] = np.var(data, axis=0)
summary_statistics['Skewness'] = skew(data, axis=0)
summary_statistics['Kurtosis'] = kurtosis(data, axis=0)


# In[25]:


summary_statistics


# In[58]:


with open('summary_statistics_table.tex', 'w') as f:
    f.write(summary_statistics.to_latex(index=True))


# In[26]:


(summary_statistics['Max'] - summary_statistics['Min'] ) / summary_statistics['Standard Deviation']


# In[27]:


summary_statistics['Mean']


# In[86]:


props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
with sns.color_palette('coolwarm'):
    f, ax = plt.subplots(4, 4, figsize=(20,15), sharey=True)
#    for i in range(len(parameter_names)):
    for i in range(4):
        for j in range(4):
            if (i*4+j) < len(parameter_names):
                x = np.linspace(np.min(data[:,i*4+j]), np.max(data[:,i*4+j]))
                y = data[:,i*4+j].shape[0]*norm.pdf(x, loc=np.mean(data[:,i*4+j]), scale=np.std(data[:,i*4+j]))
                ax[i,j].plot(x, y)
                ax[i,j].hist(data[:,i*4+j], bins=15)
                ax[i,j].axvline(x=np.mean(data[:,i*4+j]), linestyle='dotted')
                ax[i,j].set_title(f'{parameter_names[i*4+j]}')
                textstr = f'Skewness {skew(data[:,i*4+j]):0.2f}\nKurtosis {kurtosis(data[:,i*4+j]):0.2f}'
                ax[i,j].text(0.5, 0.95, textstr, transform=ax[i,j].transAxes,
                             fontsize=14, verticalalignment='top', bbox=props)
plt.savefig('images/histogram_array.png', dpi=300, bbox_inches='tight')
plt.show()


# In[101]:


N = [5, 25, 100, 250]
D = [norm.rvs(size=n) for n in N]

f, axs = plt.subplots(1, len(N), sharey='all')

for _ in range(len(N)):
    axs[_].hist(D[_], density=True)
    axs[_].set_title(f'N={N[_]}')

plt.savefig('sample_size.png', dpi=300, bbox_inches='tight')
plt.show()


# In[ ]:




