---
layout: post
title:  "Hypothesis Testing"
keywords: "Data Science, Hypothesis Testing, Python, Sqlite, Flatiron School"
categories: [machine-learning]
tags: [Data Science, Hypothesis Testing, Python, Sqlite, Flatiron School]
icon: icon-cogs
---

***Do discounts have a statistically significant effect on the number of products customers order? If so, at what level(s) of discount?***

![Northwind Database Schema](https://cdn-images-1.medium.com/max/2426/1*I3bp6yGM27SyMZYv3kqIwA.png)

## OVERVIEW

The [Northwind database](https://relational.fit.cvut.cz/dataset/Northwind) contains the sales data for a fictitious company called Northwind Traders, which imports and exports specialty foods from around the world.

The [Null Hypothesis](https://www.investopedia.com/terms/n/null_hypothesis.asp) claims that there is no correlation between the number of products ordered and whether or not there is a discount applied to the order.

The [Alternative Hypothesis](https://www.thoughtco.com/null-hypothesis-vs-alternative-hypothesis-3126413) claims that the amount of products ordered is related to whether or not there is a discount applied to the order.

The [Significance Level](https://blog.minitab.com/blog/adventures-in-statistics-2/understanding-hypothesis-tests-significance-levels-alpha-and-p-values-in-statistics)( Alpha α ) is 0.05, meaning that I am comfortable with a 5% chance of rejecting the Null Hypothesis when it shouldn’t be rejected.

The [Effect Size](https://www.leeds.ac.uk/educol/documents/00002182.htm) will be determined using [Cohen’s d](https://www.socscistatistics.com/effectsize/default3.aspx) and the understanding that we are only interested in a mean difference of ±3 products/order in relation to the [control](https://www.thoughtco.com/what-is-a-control-group-606107) and [experimental group](https://explorable.com/scientific-control-group).

An [ANOVA](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/anova/) test will be used to determine the correlation between the number of products ordered at the discount levels 5%, 10%, 15%, 20% and 25%.

Through the application of the [Central Limit Theorem](https://towardsdatascience.com/understanding-the-central-limit-theorem-642473c63ad8), an [Independent Samples T-Test](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/t-test/) will be used to determine if we can reject the Null Hypothesis.

## OBTAINING THE RELEVANT DATA

The hypothesis experiment will be performed primarily in [Python](https://www.python.org/) within a [Jupyter Notebook](https://jupyter.org/). But, I found the [DB Browser for SQLITE](https://sqlitebrowser.org/) to be a helpful tool for navigating around the SQL database and finding paths to all the variables of interest before pulling the necessary data into a [Pandas](https://pandas.pydata.org/) dataframe for analysis. Side Note: Later on in the process, I am able to check [*Glass’ Delta ](https://www.statisticshowto.datasciencecentral.com/glasss-delta/)*and *[Hedges’ G](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/hedgeg.htm) *along with* [Cohen’s D](https://www.statisticshowto.datasciencecentral.com/cohens-d/) *very quickly **using this *[*effect size calculator](https://www.socscistatistics.com/effectsize/default3.aspx).

    # create a connection to the database
    conn = sqlite3.connect('Northwind_small.sqlite', detect_types=sqlite3.PARSE_COLNAMES)
    c = conn.cursor()

    # query the database
    c.execute("SELECT * FROM OrderDetail;")

    # store results in a dataframe
    df_orders = pd.DataFrame(c.fetchall())
    df_orders.columns = [i[0] for i in c.description]

    # double check everything imported correctly
    df_orders.head()

**Studying the variables of interest’s and their respective distributions.**

    #dist plot of Quantity values
    plt.subplot(211)

    sns.distplot(df_orders['Quantity'], hist='density')
    plt.title('Distribution of Quantity', fontsize=16)
    plt.xlabel('Quantity', fontsize=16)

    #dist plot of discount values
    plt.subplot(212)

    sns.distplot(df_orders['Discount'], hist='density')
    plt.title('Distribution of Discount', fontsize=16)
    plt.xlabel('Discount', fontsize=16)

    plt.tight_layout()
    plt.show()

![Positively Skewed](https://cdn-images-1.medium.com/max/2000/1*h8GCerd0sOcjeGptFTm9cQ.png)

The [Dependent Variable](https://en.wikipedia.org/wiki/Dependent_and_independent_variables)** — Quantity** is positively [skewed](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/skewed-distribution/) with a minimum order quantity of 1 and the largest order was 130. Meaning that many orders include only a few items with fewer large volume orders.

The Independent Variable**-Discount** is an [ordinal](https://stats.idre.ucla.edu/other/mult-pkg/whatstat/what-is-the-difference-between-categorical-ordinal-and-interval-variables/)** **variable that has a very near, but not quite, [uniform distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3662.htm)** **across the 5%, 10%, 15%, 20%, 25% increments and a large number of orders with no discount applied.

### Setting up control and experimental groups

* The **control** group is comprised of orders that did not receive a discount

* The **experimental** group is comprised of orders that did receive a discount

    # create control and experimental groups
    # put items without the variable we are curious about into control group
    control = df_orders[df_orders['Discount'] == 0]
    ctrl = control['Quantity']

    # put items with the variable we are curious about into experimental group
    experimental = df_orders[df_orders['Discount'] != 0]
    exp= experimental["Quantity"]

## Check Normality

    # visual check for normality
    plt.figure(figsize=(20, 10))
    sns.distplot(ctrl, label='Control')
    sns.distplot(exp, label='Experimental',kde=True, hist=True )
    plt.title('Visual Check for Normality', fontsize=25)
    plt.xlabel('Quantity/Order', fontsize=20)
    plt.legend(fontsize=55);

![Positively Skewed](https://cdn-images-1.medium.com/max/3340/1*h3zehs4M6HBOw6NnW2ukUg.png)

### Creating a Sampling Distribution of Sample Mean

    # find the difference of means before sampling
    original_diff = abs(ctrl.mean()-exp.mean())

    #check the amount of samples and using store half into variable
    i = np.round(len(df_orders)/2,0)

    #create 2 sample groups
    sample_a = df_orders["Quantity"].sample(1078)
    sample_b = df_orders.drop(sample_a.index)["Quantity"]

    #find the difference of means after sampling
    sample_diff = abs(np.mean(sample_a) - np.mean(sample_b))

    #grab a sample from the ctrl and exp group 10,000 times
    my_diffs = []
    for i in range(10000):
        sample_a = df_orders["Quantity"].sample(1078)
        sample_b = df_orders.drop(sample_a.index)["Quantity"]
        diff = np.mean(sample_a) - np.mean(sample_b)
        my_diffs.append(diff)

    # visualize the data
    #sampling of control group compared to mean of experimental
    plt.title('Sampling of Control Group compared to Mean of Experimental')

    plt.hist(my_diffs, color="orange", label="Control Group Sampling")
    plt.axvline(original_diff, color = 'k', linewidth = 5, label="Mean of Experimental Group")
    plt.xlabel(ttest_ind(exp.values, ctrl.values ))

    plt.legend()
    plt.show();

![](https://cdn-images-1.medium.com/max/2000/1*7LY6ZatnnFxTEt7G_A3WzA.png)

### Beginning to draw conclusions

The large gap between the distribution of the control sampling and the mean of the experimental visually confirms our [T-Test results](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html). The p-value is less than 0.05, we can reject our null hypothesis. There is not, “ *no correlation between the number of products ordered and whether or not there is a discount applied to the order.”*

**There probably is*** “*a correlation between the number of products ordered and whether or not there is a discount applied to the order.”

### Determine the effect size using Cohen’s d.

    def Cohen_d(group1, group2):

    # Compute Cohen's d.

    # group1: Series or NumPy array
        # group2: Series or NumPy array

    # returns a floating point number

    diff = group1.mean() - group2.mean()

    n1, n2 = len(group1), len(group2)
        var1 = group1.var()
        var2 = group2.var()

    # Calculate the pooled threshold as shown earlier
        pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

        # Calculate Cohen's d statistic
        d = diff / np.sqrt(pooled_var)

        return d

    # Run both samplings through function
    abs(Cohen_d(ctrl_sample,exp_sample))

Cohen’s d = 6.33 is well above the 0.8 [large threshold](http://core.ecu.edu/psyc/wuenschk/docs30/EffectSizeConventions.pdf). Meaning that the difference between the control and experimental groups is large and very much worth using discounts to increase the number of items per order.

### Comparing the various discount percentages using [Tukey’s HSD](https://en.wikipedia.org/wiki/Tukey%27s_range_test).

    df_orders['Quantity'].groupby(df_orders['Discount']).describe()

![](https://cdn-images-1.medium.com/max/2000/1*7sDzo9KYbQRVSpdZmJM3zw.png)

    mc = MultiComparison(df_orders['Quantity'], df_orders['Discount'])
    mc_results = mc.tukeyhsd()
    print(mc_results)

    mc_results = mc_results.plot_simultaneous(figsize=(16,14))
    plt.show()

![Comparing the Different Discount Percentages Mean Quantity as a Box and Whisker’s Plot](https://cdn-images-1.medium.com/max/2000/1*swhC_iV2QK1xTTVBPUb8fg.png)

### Final Conclusions

![](https://cdn-images-1.medium.com/max/2000/1*UyRoD7eEHl8VHJckcUoV6w.png)

I’d recommend the imaginary Northwind Traders continues offering discounts to increase the number of items per order. I’d also recommend offering a 5% or 15% and larger discount while avoiding a 10% discount. As 10% was the only discount percentage that failed to reject the null hypothesis.
