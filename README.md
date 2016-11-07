# Customer-Journey-LDA-Modeling

## Overview
<a href ='http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf'> Latent Dirichlet Allocation </a> (LDA) is a statistical model that extracts various topics from documents. It is usually used to cluster words by theme but I used it to cluster customer journeys for <a href='https://www.clickfox.com/'> Clickfox </a>. I found this problem incredibly interesting because it extends Natural Language Processing algorithms to non-natural language data. The goal of this project was to develop insights from modeling these customer journies with LDA.

## The Data
The data used for this was 480,000 rows by 11,000 columns of customer journey events. Due to the size of this data, manipulation was done via numpy with multiprocessing. The data were pickled along the way to ensure that code was not run redundantly. 

## Status
Current status of this project is open. This means that this README is still under construction and is subject to change.

## Process
<img src='images/optimal.png' height="300", width="375"><br>
This process involved numerous steps:
<ul>
    <li>Formatting the data properly
    <li>Building a prototype LDA model using Gensim
    <li>Developing preliminary visualizations for the model results
    <li>Tweaking the LDA model for optimization
    <li> Interpreting the model results
    <li> Gaining insights from these results
    <li>Creating visualizations to communicate these insights
</ul>
