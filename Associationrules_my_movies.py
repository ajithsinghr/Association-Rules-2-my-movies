# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:37:53 2023

@author: ramav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\Assignments\\Association rules\\my_movies.csv")
df.shape
df.head()
df.info()
df = df.iloc[:,5:]
df

# Apriori Algorithm
# 1. Association rules with 10% Support and 70% confidence

from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder

# with 10% support
frequent_itemsets = apriori(df,min_support = 0.1,use_colnames=True)
frequent_itemsets

# 70% confidence
rules=association_rules(frequent_itemsets,metric='lift',min_threshold=0.7)
rules

# Lift Ratio>1 is a good influential rule is selecting the associated
rules[rules.lift>1]

# Visualization of obtained rule
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()

# 2. Association rules with 5% Support and 90% confidence
# with 5% support
frequent_itemsets2=apriori(df,min_support=0.05,use_colnames=True)
frequent_itemsets2

# 90% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.9)
rules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


# 3. Association rules with 20% Support and 80% confidence
# with 20% support
frequent_itemsets2=apriori(df,min_support=0.20,use_colnames=True)
frequent_itemsets2

# 80% confidence
rules2=association_rules(frequent_itemsets2,metric='lift',min_threshold=0.8)
rules2

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
rules2[rules2.lift>1]

# visualization of obtained rule
plt.scatter(rules2['support'],rules2['confidence'])
plt.xlabel('support')
plt.ylabel('confidence')
plt.show()


