---
layout: post
title:  "Working with neural networks has become more user-friendly over the past few years."
desc: "Working with neural networks has become more user-friendly over the past few years. The most challenging aspect for newcomers is running the algorithm because a lot of processing power is needed."
keywords: "Machine Learning, Keras, Eda, Exploratory Data Analysis, Student Performance"
categories: [machine-learning]
tags: [Machine Learning, Keras, Eda, Exploratory Data Analysis, Student Performance]
icon: icon-cogs
---

Working with neural networks has become more user-friendly over the past few years. The most challenging aspect for newcomers is running the algorithm because a lot of processing power is needed. There are several solutions to this hurdle, the most popular of which is to run the models on Amazon’s cloud service or Google’s Gcloud (currently offering a $300 credit to new users). The data set used in this blog was able to be processed locally, thanks to its small size. However, data sets with a more significant number of instances would have required the use of several GPUs.

There is a saturation of tutorials on Keras throughout the internet. This article primarily focuses on giving a client what they need as a supplement to what they have requested. A link to the raw data set [https://archive.ics.uci.edu/ml/datasets/student+performance](https://archive.ics.uci.edu/ml/datasets/student+performance) there is also an article on data mining to predict school performance [http://www3.dsi.uminho.pt/pcortez/student.pdf](http://www3.dsi.uminho.pt/pcortez/student.pdf). By recreating the models using traditional Machine Learning techniques, it is possible to achieve similar outcomes with little to no data munging. This project took the same data and went a step further by running it through several Keras models that were able to outperform earlier iterations of the project. The real advantage of using a neural network on a regression classification is that feature engineering is performed “under the hood” and neural networks tend to be more robust against imbalanced features than traditional machine learning models.

{% include figure.html image="/images/1.png" caption="Most Binary Features are Heavily Skewed" %}

{% include figure.html image="/images/2.png" caption="Traditional ML Techniques on Very Lightly Processed Data (only scores of zero removed)" %}

{% include figure.html image="/images/3.png" caption="Keras Model Outperforms all ML models" %}

{% include figure.html image="/images/4.png" caption="After Tuning the Model" %}

The most significant issue with Neural Networks is their lack of interpretability. Moreover, when dealing with data such as this student data set, it is our job as the data scientist to provide the client with what they need on top of whatever it is that the client requested. So a predictor such as the model created in Keras would only be usable as a tool for an admissions board at a highly selective (probably illegal) school. As someone who has taught in the public school system for over a decade, the idea of leaving the project “as-is” and just providing the client with what they asked for would have been amoral.
Taking a step back from the project and asking, “what information would be of the most use to the client?” Was the next logical step on this project.
 In an ideal world, this would be asked and answered before any work has begun. However, we do not live in an ideal world. Clients have an idea and want to use AI as the solution because AI is the shiny new toy that all their friends are using. No problem, give them a neural network model (they are easy enough to build), and include what you believe to be the more useful analytics within the final report.
 The EDA phase of the project provided a lot of useful information so let us walk through that phase now.

[Download the Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance) at the UCI Machine Learning Repository (a wonderful source of data sets to practice with).[ Link to original paper](http://www3.dsi.uminho.pt/pcortez/student.pdf) by Paulo Cortez and Alice Silva is well worth a read. The original project had a primary objective of predicting student achievement and secondary objective to identify the key variables that affect educational success/failure.

We can answer the secondary objective with one line of code.

{% include figure.html image="/images/5.png" %}

The features with the largest absolute value are the key variables that affect educational success and failure. I wanted to make the data useful to a teacher who is not able to choose which students they teach. So, lets take a closer look at each of the variables listed above that are managable and try to determine where students or their families should focus their efforts to minimize their chances of failure.

Failures is displaying -0.3474. Number of past failures is inversely correlated to success. Well, students cannot change the past. However, teachers can use this information on future classes to stress the importance of doing well from the start of their educational journey. It could also mean that these children are less intelligent and are always going to score low. The point being, we do not know for certain why these students are failing early on as well just that there is a correlation between past and future failure.

Mother’s educational level (Medu) and number of absences are the next most important features. As an teacher I was constantly asked by parents what they could do to better their child’s chances of doing well in school. I would always tell parents to get their children to school everyday and I would do what I could to help. But, I have also noticed that children who study with their parents tend to be the most successful. Especially if the parent is studying to better themselves not just helping out with their homework. So, it is interesting that a higher educational level of both the mother (Medu) and father (Fedu) are strongly correlated to the student’s final grade.

There are some correlations that appear to be common sense and others that defy logic. Such as, alcohol consumption. It is surprising, to me, to see weekday drinking as less harmful than weekend drinking. Or, free-time having an adverse affect on student performance. But, a project like this would be the perfect starting block for future studies. I would suggest that each of these strongly correlated features (that are manageable) be studied in a series of hypothesis tests.
