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
    .assign(Youngness=lambda df: (df.Age <= 16).map({True: 'Below 16', False: 'Above 16'}))
    .assign(Survival=lambda df: df.Survived.map({0: 'Died', 1: 'Survived'}))
    .assign(Class=lambda df: df.Pclass.map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'}))
    .sort_values(by=['Class', 'Survived'])
)

g = sns.distplot(
    df.Age.fillna(-40),
    rug=True
)


g = (
    df.Age
    .isnull()
    .map({False: 'Age data available', True: 'Age data missing'})
    .value_counts()
    # .index.
    .plot.pie(autopct='%.2f %%', explode=(0, 0.1))
)


g = sns.barplot(
    data=df,
    x='Sex',
    y='Survived',
    hue='Class',
    palette='Set1',
)


g = sns.FacetGrid(
    df,
    col='Sex',
    row='Class',
    hue='Survival',
    hue_kws={'marker': ['+', 'x']},
    palette='Set1',
)
g.map(plt.scatter, 'Fare', 'Age', alpha=.5)
g.add_legend()


g = sns.FacetGrid(
    df[df.Class != '1st Class'],
    col='Sex',
    row='Class',
    hue='Survival',
    hue_kws={'marker': ['+', 'x']},
    palette='Set1',
)
g.map(plt.scatter, 'Fare', 'Age', alpha=.5)
g.add_legend()


g = sns.factorplot(
    data=df,
    x='Sex',
    y='Survived',
    col='Youngness',
    hue='Class',
    palette='Set1',
    kind='bar',
)


df.info()



# Chi-Squared Tests
def apply_chi2_contingency_and_print_results(df, independent_variable, dependent_variable, alpha=0.05):
    from scipy.stats import chi2_contingency
    # Pclass to Survivability
    pivot = (
        df[[dependent_variable, independent_variable]]
        .pipe(
            pd.pivot_table,
            index = dependent_variable,
            columns = independent_variable,
            aggfunc = len
        )
    )
    chi2, p_value, dof, expected = chi2_contingency(pivot)
    print("Chi-Squared test on {} to {}.".format(independent_variable, dependent_variable))
    print("Does {} have a significant effect on {}?".format(independent_variable, dependent_variable))
    print("chi2 score: {:.2e} | p value: {:.2e}".format(chi2, p_value))
    print('Null hypothesis is {maybe}rejected: {} and {} are {maybe}correlated.'
          .format(independent_variable, dependent_variable, maybe='' if p_value < alpha else 'not '))
    print()

alpha = 0.05
apply_chi2_contingency_and_print_results(df, 'Class', 'Survival', alpha=0.05)
apply_chi2_contingency_and_print_results(df, 'Youngness', 'Survival', alpha=0.05)

## Data analysis: t-test
# We compute various t-test for the following alternative hypotheses:
# - Pclass = 1 or 2; Sex = female (rich or middle-class girls & women): high survival rate
# - Pclass = 1 or 2; Sex = male; Youngness = True (rich or middle-class boys): high survival rate
# - Pclass = 2 or 3; Sex = male; Youngness = False (poor or middle-class men): low survival rate

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

# - Pclass = 1 or 2; Sex = male; Youngness = True (rich or middle-class boys): high survival rate
alpha = 0.05
selection = (df.Pclass != 3) & (df.Sex == 'male') & (df.Youngness == True)
ttest_selection_and_print_result(df, selection, alpha)
selection.value_counts()


# - Pclass = 2 or 3; Sex = male; Youngness = False (poor or middle-class men): low survival rate
alpha = 0.05
selection = (df.Pclass != 1) & (df.Sex == 'male') & (df.Youngness == False)
ttest_selection_and_print_result(df, selection, alpha)
selection.value_counts()
