#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 14:28:09 2017

@author: Viggy
"""
#!pip install newspaper3k

from newspaper import Article

url = 'http://www.cnn.com/2017/10/11/politics/donald-trump-is-everywhere/index.html'
article = Article(url)

article.download()
article.html
article.parse()
article.authors
article.publish_date
article.text 

article.top_image

article.movies

article.nlp()
article.keywords
article.summary

import newspaper

cnn_paper = newspaper.build('http://cnn.com')

for article in cnn_paper.articles:
    print(article.url)

for category in cnn_paper.category_urls():
    print(category)
    
cnn_article = cnn_paper.articles[0]
cnn_article.download()
cnn_article.parse()
cnn_article.nlp()

from newspaper import fulltext

html = requests.get(...).text
text = fulltext(html)
