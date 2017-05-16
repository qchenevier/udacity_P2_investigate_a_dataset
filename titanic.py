import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

%matplotlib inline

# load data
filename = 'titanic-data.csv'
df_raw = pd.read_csv(filename)

# clean Age & Fare columns
col_to_analyze = ['Pclass', 'Sex', 'Age', 'Fare']
df = (
    df_raw
    .loc[:, col_to_analyze + ['Survived']]
    .assign(Young=lambda df: df.Age <= 16)
)

g = sns.barplot(
    data=df,
    x='Sex',
    y='Survived',
    hue='Pclass',
    palette='Set1',
)


g = sns.FacetGrid(
    df,
    col='Sex',
    row='Pclass',
    hue='Survived',
    hue_kws={'marker': ['+', 'x']},
    palette='Set1',
)
g.map(plt.scatter, 'Fare', 'Age', alpha=.5)
g.add_legend()


g = sns.FacetGrid(
    df[df.Pclass != 1],
    col='Sex',
    row='Pclass',
    hue='Survived',
    hue_kws={'marker': ['+', 'x']},
    palette='Set1',
)
g.map(plt.scatter, 'Fare', 'Age', alpha=.5)
g.add_legend()


g = sns.factorplot(
    data=df,
    x='Sex',
    y='Survived',
    col='Young',
    hue='Pclass',
    palette='Set1',
    kind='bar',
)

## Data analysis: t-test
# We compute various t-test for the following alternative hypotheses:
# - Pclass = 1 or 2; Sex = female (rich or middle-class girls & women): high survival rate
# - Pclass = 1 or 2; Sex = male; Young = True (rich or middle-class boys): high survival rate
# - Pclass = 2 or 3; Sex = male; Young = False (poor or middle-class men): low survival rate

def ttest_selection_and_print_result(df, selection, alpha):
    ttest = ttest_ind(
        df.loc[selection, 'Survived'],
        df.loc[~selection, 'Survived'],
        equal_var=False
    )
    print('p value: {pvalue:.2e} | t stat: {tstat:.2e}'
          .format(pvalue=ttest.pvalue, tstat=ttest.statistic))
    print('Null hypothesis is {maybe}rejected.'
          .format(maybe='' if ttest.pvalue < alpha else 'not '))
    print('Selection size: {}'.format(len(selection[selection])))



# - Pclass = 1 or 2; Sex = female (rich or middle-class girls & women): high survival rate
alpha = 0.05
selection = (df.Pclass != 3) & (df.Sex == 'female')
ttest_selection_and_print_result(df, selection, alpha)
len(selection[selection])

# - Pclass = 1 or 2; Sex = male; Young = True (rich or middle-class boys): high survival rate
alpha = 0.05
selection = (df.Pclass != 3) & (df.Sex == 'male') & (df.Young == True)
ttest_selection_and_print_result(df, selection, alpha)
selection.value_counts()


# - Pclass = 2 or 3; Sex = male; Young = False (poor or middle-class men): low survival rate
alpha = 0.05
selection = (df.Pclass != 1) & (df.Sex == 'male') & (df.Young == False)
ttest_selection_and_print_result(df, selection, alpha)
selection.value_counts()
