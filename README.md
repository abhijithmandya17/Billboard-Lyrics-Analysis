# Text Mining Song Lyrics
Using text mining techniques on song lyrics to predict success

Duncan Rule, Abhijith Mandya, and Nitesh Prakash

****

## Hypothesis, Objective & Metrics

Our objective is to build models that can accurately predict whether a song will be on the year-end Billboard Top 100 chart using only its lyrics. Our hypothesis is that the sentiment, topics, and words present in a song’s lyrics influence its performance in a predictable way. For example, we may find that sad songs about certain topics, or songs with topics pertaining to love are likely to place on the Billboard chart. More specifically, we believe that supervised learning models that include a song’s lyric features can perform significantly better than chance. Since our outcome is binary, we will examine our models using ROC curves and use the area under these curves as our primary evaluation metric.

## Summary of Results

![alt text](https://github.com/duncanrules/lyric_analysis/blob/master/output_plots/joint_roc.png "Joint ROC Plot")

Our hypothesis that a song’s lyrics are predictive of its success is supported by our results. We were able to predict with high accuracy which songs would place on the year-end Billboard Top 100 chart, but all of our models suffered from the class imbalance of our response variable - predicting a successful song proved to be much more difficult than correctly labelling an unsuccessful song, as shown by the confusion matrices. Despite this weakness of our models, they were all shown to be valuable predictors with relatively high AUC values.

## [Data](https://virginia.box.com/s/2ssuphywc78su1cjtwahn5w7lkv5nyvg)

The data was obtained from multiple Kaggle datasets of songs and their lyrics; the first was a dataset in which the host took the year end Billboard charts and scraped song lyrics websites. The second and third are more exhaustive song lyric datasets scraped from similar sites, but include both songs that were present on the Top 100 chart and those that were not. By combining these datasets, we can get a large amount of songs and their lyrics, as well as labels of “success” as determined by whether or not they made the chart. Since the Billboard Top 100 chart ranges from 1964 to 2015, there will likely be a time series bias while working with the data and recency bias with more songs being released closer to current day. 

[[1]](https://www.kaggle.com/rakannimer/billboard-lyrics)[[2]](https://www.kaggle.com/mousehead/songlyrics)[[3]](https://www.kaggle.com/artimous/every-song-you-have-heard-almost)

## Methods - Pre-processing

Since the lyrics were spread across three datasets, our first task was to join the data into a single dataframe. This was accomplished by removing all non-alphanumeric characters and whitespace so that a join could be made by matching a song’s title and artist. The songs were then labelled with a 0 or 1 according to whether or not they were present in the Billboard chart dataset - this was used as our ground truth label of “success”. Once we had a single dataframe, we further cleaned the data by removing duplicates - because of the existence of covers, we ignored lyrics when matching duplicates and only considered title and artist. Lastly, we used a Python port of Google’s language detection library [[4]](https://pypi.python.org/pypi/langdetect) to remove all non-English songs, allowing us to use feature extraction methods with English-specific dictionaries. A random split into a train/test (80/20) format was made for model evaluation purposes and to avoid cross contamination. The train set had 34,930 songs while the test set had 11,446.

## Methods - Feature Extraction

Because our team is working exclusively with text data, we employed different yet well-known text mining transformations: 

#### Latent Dirichlet Allocation (LDA)

LDA is a generative statistical model that allows sets of observations to be explained by unobserved groups that explain why some parts of the data are similar. It posits that each document is a mixture of a small number of clusters or topics and that each word's creation is attributable to one of the document's topics. A combination of validation and computing architecture constraints, led to an optimal choice of 20 topics based on which every song in our sample set was divided. First, we removed any stopwords (commonly recurring, that do not add value to predictions e.g. the, a, etc.). To reduce vector size, we also ignored words that occured in just 1 song. The train set was used to initialize the LDA model using the Gensim package [[6]](https://radimrehurek.com/gensim/). This model was then used to calculate the distribution weights of 20 topics across the whole data set to maintain consistency.

#### Sentiment Analysis

To analyze the general positive or negative sentiment of a song, we used NLTK’s VADER lexicon [[7]](http://comp.social.gatech.edu/papers/icwsm14.vader.hutto.pdf). The VADER lexicon uses human-verified labels for words and phrases, and scores them not only as positive or negative, but also according to the intensity of that sentiment. By aggregating the scores of the words within a song’s lyrics, we were able to calculate a score for both positive and negative sentiment (as well as a compound score) for each song.

#### Empath Categories

To determine the degree to which each song’s lyrics are associated with various categories such as love or violence, we used the Empath lexicon [[8]](https://hci.stanford.edu/publications/2016/ethan/empath-chi-2016.pdf). Empath created these category lexicons by starting with seed-words for each category and using word embeddings to identify similar words. These categories were then verified by humans on a large scale. By checking each word in a song’s lyrics against these categories, we were able to determine how much of each song is “about” each of these 198 categories, similar to the LDA topic weights.

#### Word2Vec

Word2Vec is a technique to reduce a high dimensionality vector of one-hot encoded words in a text mining problem to one with a reduced set of features. This is done by finding the words associated with each other with high probability. This allows each word to be mapped on a vector space based on how close they are to our list of features. In our scenario we generated 300 features with each song being sampled up to the first 500 words to create a uniform set of features. Any song that had fewer than 500 words were padded with empty cells on the left.

#### Term Frequency/Inverse Document Frequency (TF-IDF)

TF-IDF is a popular transformation used to weigh each word in the text document according to how unique it is. In other words, the TF-IDF approach captures the relevancy among words, text documents and particular categories [[9]](http://ieeexplore.ieee.org/document/5778755/?part=1). This approach produces an enormous sparse matrix which is computationally challenging to model. Thus, we limited our terms to those that occurred in at least 20 songs in our data set. The matrix dimensions were 46376X9370 i.e., there were 9370 terms that occured in at least 20 songs in our dataset.

## Methods - Modeling

By extracting the features of the song lyrics, the text was transformed into a format that could be used by many supervised modeling techniques to predict Billboard success. Because of computing restraints, we split our modeling into two approaches: the first method was to use the TF-IDF weights, sentiment scores, Empath category scores, and LDA topic weights as a combined predictor set. These predictors were then used as inputs to Random Forests (RF), a multinomial Naive Bayes (NB) classifier, and an XGBoost model. 

The NB classifier was chosen as a baseline model because of its common use in text classification due to its simplicity and efficiency. In more recent text classiﬁcation studies, however, RF is being increasingly chosen over other text classification techniques because of the theoretical guarantees for optimal classiﬁcation performance RF provides. In relation to the factors to consider when choosing a text classiﬁcation method, RF are increasingly seen to outperform other classiﬁers in text mining applications such as document categorisation problems like ours [[10]](http://www.academia.edu/11059601/Random_Forest_and_Text_Mining). We then chose to implement XGBoost model since it is an improvement over RF to see if that would increase our prediction accuracy. All supervised learning models were tuned and cross validated using GridSearchCV.

The second method was to build a convolutional neural net (CNN) with Long Short Term Memory (LSTM) to carry out text classification. Using the word2vec representation of the songs, an embedding matrix was created to which all the words were mapped  This mapping was then used as the input features for the model. The model here used a 1D CNN with an LSTM layer(This allows the model to retain information about previous portions in the song) to carry out binary text classification with a sigmoid activation function [[13]](https://github.com/rantsandruse/lstm_word2vec).

## Methods - Computing Platform

The sheer size of the data led us to consider service-oriented architecture to accommodate it. The initial feature engineering was solely done on Google Cloud Platform (GCP) with an 8 CPU-2 GPU Debian backend. We found constraints in two forms that made it unavoidable to abandon GCP and revert to local architecture. First, Some of our feature extraction models required a package called gensim which was crucial for LDA. After multiple trials, gensim failed to load on our virtual machine. Second, our virtual machine deprecated our SSH keys locking us out of the instance. The technical support for the free tier we used is yet to resolve the issue, at the time of writing.

## Results

#### Multinomial Naive Bayes

The NB classifier was by far our most efficient model, completing in under a minute. However, this efficiency came at the cost of accuracy and AUC - the model labelled just 13 “hit” songs correctly with an overall accuracy of 89.51%, and had an AUC of .7 for the test set.

#### Random Forest

The RF classifier was the best of both worlds. It had an accuracy of 89.43% while correctly classifying only 18 songs as a hit and had an AUC of 0.79 for the test set. These metrics were achieved with a computational cost which was comparable to NB’s runtime.

#### XGBoost

The XGBoost was the best performer in terms of the supervised models and computation cost with an accuracy of 90.56% and an AUC of 0.81. It correctly identified 107 hit songs. 

#### Convolutional Neural Net

While the neural network based approach gave us the highest accuracy of 92.55% and an AUC of 0.86. This was a good result in light of the fact that class imbalances in our response variable lead to any accuracy below 90% to be only marginally significant. However the long time required to train the model and the opacity of the implementation itself meant it was difficult to tune the parameters to obtain the best possible results.

## Reflections

Class imbalance plays a crucial role in model accuracy and we need to develop more robust ways of downplaying its effect. We tuned class weights and downsampled our data set to improve our predictions but the problem continued to affect our models.

Working with large datasets has opened us up to a plethora of new avenues and challenges through which we initially struggled but overcame. Asymptotically, many if not all our basic functions (saving a file, version control etc.) need to be redefined to the problem forcing us to adopt service based architectures and third party computing resources. 

Although we engaged in extensive feature engineering, it was more horizontal than vertical i.e., we applied multiple transformations but did little to optimize them.  For instance, Latent Semantic Analysis (LSA) or Singular Vector Decomposition (SVD) for text mining are great feature selection methods which work wonders in this dimensionality cursed problem.

This problem ideally needs to find a balance between interpretability and predictive power such that an Artist can, based on learnings, write a successful song. Our approach, given the data, solely focused on predicting. Features like the genre, single/album song, beats/min can effectively improve our predictive power as well as shine a light on the interpretability of our models. This ensemble can be used by record companies, in addition to other available data to predict the probability of a song/artist, of their choosing, making it to the Billboard Top 100. In the future, this can be extended to the artist by improving interpretability of the model helping them write chart topping songs.
