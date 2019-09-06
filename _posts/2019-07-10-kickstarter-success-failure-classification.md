---
layout: post
title:  "Kickstarter Success/Failure Classification"
keywords: "Machine Learning"
categories: [machine-learning]
tags: [Machine Learning]
icon: icon-cogs
---

I worked with the Kickstarter dataset found on Kaggle.com to build a model that would predict the outcome of a Kickstarter campaign. I chose to work with this dataset because I just went through a small business incubator and while the lessons learned were extremely valuable I will probably use crowdfunding platforms on future.

What I like about Kickstarter:

 1. The small business owner retains full ownership of the business.

 2. The money received is used to deliver a product directly into the hands of a responsive and diverse test populace that is highly likely to provide feedback! This is huge, even selling a new product at a loss makes sense with this sort of model. I gave out so many free Minimal Viable Products to people at a 100% loss and was lucky to receive any feedback.

 3. A successful kickstarter campaign is free-publicity. A large portion of the marketing is done just by setting up a project. Sure, you are unlikely to succeed without supplemental marketing. However, the Kickstarter website will promote some products to an audience that you would have no way of reaching through traditional means and backers are likely to follow future projects if they like the outcome of this one.

This was my first classification project and as such, I was not entirely happy with the outcome. When I redo this project I will seek out higher quality data through web scraping as I was only able to create a classification model with less than 63% accuracy. Don’t get me wrong, this is a huge step up from the 54% I was getting originally before feature engineering and hyperparameter tuning.

The most valuable insights were made not through the classification of the outcome but through looking at the relationships of different features to the outcome. As these observations could be used to increase the statistical likelihood of a successful Kickstarter campaign.

Let’s take a look at when most projects start and end. You can click the link below to go to plotly and see the distribution of successful and failed campaigns throughout the calendar year. You may notice that the majority of projects fail in December. So, avoid trying to campaign around the winter holiday season. Instead, wait until the end of February or beginning of March.

[**What Day of the Year Do Film Projects Start and Finish? | histogram made by Burton-david | plotly**
*Burton-david's interactive graph and data of "What Day of the Year Do Film Projects Start and Finish?" is a histogram…*plot.ly](https://plot.ly/~Burton-David/5.embed)

When dealing with the data I originally used all of the available features and created a model that could predict the outcome with over a 99% accuracy. This model was next to useless because it included features that would not be available at the onset of a project. Such as the number of backers and amount pledged.

So, I took those features out and used the Launched and Deadline features to engineer 16 other variables of data. Such as campaign length, Day of the week project was started, ended along with others.

My new model was able to predict the outcome with a 54% accuracy and was highly unstable. Meaning that the results were different each time I ran the model. This problem was solved by using an ensemble of Random Forest, KNN and Logistic Regression to predict the outcome with 61% accuracy.

While 61% is not great, I wouldn’t place money on those odds the real insight came when observing which features were correlated with the outcome. For instance the day of the week a project starts on is significant but the day it ends seems not to matter nearly as much.

As one might expect smaller goals are more likely to be met. And, dance fundraisers are the most likely to succeed. Having this new bit of information I rushed to my friend to let her know that she should campaign for her dance company on Kickstarter as it is the most likely type of campaign to succeed!

She burst the bubble by explaining that dance companies are required to put down money to reserve revenues long before a kickstarter campaign makes sense to do. And because kickstarter has an all or none approach. Meaning that the campaign succeeds and you get all the money, or it falls short and you get none. Therefore, if the project has any money in it the person leading the campaign is likely to just donate the rest themselves because the money has already been spent. So, those high numbers are likely due to a lot of creators bankrolling their own projects to secure the less than goal amount.

You can look forward to future posts about how different elements of a kickstarter campaign are correlated with its outcome. But, remember to take them with a grain of salt because there are priors such as the aforementioned situation worth taking into consideration.
