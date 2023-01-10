# %%
import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
import re
from bs4 import BeautifulSoup
 

# %%
! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Jewelry_v1_00.tsv.gz

# %% [markdown]
# ## Read Data

# %%
import pandas as pd

tsv_file='amazon_reviews_us_Jewelry_v1_00.tsv'
 
# reading given tsv file
amazon_data_table=pd.read_csv(tsv_file,sep='\t',usecols = ['star_rating','review_body'],low_memory=False)
amazon_data_table['star_rating'] = pd.to_numeric(amazon_data_table['star_rating'],errors='coerce')

# %% [markdown]
# ## Keep Reviews and Ratings

# %%
import pandas as pd

tsv_file='amazon_reviews_us_Jewelry_v1_00.tsv'
 
# reading given tsv file
amazon_data_table=pd.read_csv(tsv_file,sep='\t',usecols = ['star_rating','review_body'],low_memory=False)
amazon_data_table['star_rating'] = pd.to_numeric(amazon_data_table['star_rating'],errors='coerce')

# %% [markdown]
#  ## We select 20000 reviews randomly from each rating class.
# 
# 

# %%
import pandas as pd
amazon_data_table_records_5_rating = amazon_data_table[amazon_data_table['star_rating'] == 5].sample(frac = 1).iloc[:20000]
amazon_data_table_records_4_rating = amazon_data_table[amazon_data_table['star_rating'] == 4].sample(frac = 1).iloc[:20000]
amazon_data_table_records_3_rating = amazon_data_table[amazon_data_table['star_rating'] == 3].sample(frac = 1).iloc[:20000]
amazon_data_table_records_2_rating = amazon_data_table[amazon_data_table['star_rating'] == 2].sample(frac = 1).iloc[:20000] 
amazon_data_table_records_1_rating = amazon_data_table[amazon_data_table['star_rating'] == 1].sample(frac = 1).iloc[:20000]
table = pd.concat([amazon_data_table_records_5_rating,amazon_data_table_records_4_rating, amazon_data_table_records_3_rating,amazon_data_table_records_2_rating,amazon_data_table_records_1_rating])
final_amazon_table = table.sample(frac=1).reset_index(drop=True)
mean_of_review_body = final_amazon_table.review_body.str.len().mean()
print(mean_of_review_body)

# %% [markdown]
# # Data Cleaning
# 
# 

# %% [markdown]
# # Pre-processing

# %%
import re
mean_of_review_body = final_amazon_table.review_body.str.len().mean()
print(mean_of_review_body)
final_amazon_table['review_body'] = final_amazon_table.review_body.str.lower()

# %%
final_amazon_table['Review_body'] = final_amazon_table['review_body'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
final_amazon_table = final_amazon_table.drop('review_body', axis=1)

# %%
from bs4 import BeautifulSoup
import contractions
final_amazon_table['Review_body'].apply(lambda x : re.sub(r' +',' ',x))
final_amazon_table['Review_body'].apply(lambda x : " ".join(re.sub('[^A-Za-z]+','', split) for split in x.split()))
final_amazon_table['Review_body'].apply(lambda x: [contractions.fix(word) for word in x.split()])
final_amazon_table['Review_body'] = [BeautifulSoup(text).get_text() for text in final_amazon_table['Review_body'] ]
mean_of_review_body = final_amazon_table.Review_body.str.len().mean()
print(mean_of_review_body)

# %% [markdown]
# ## remove the stop words 

# %%
import nltk
nltk.download('punkt')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))
# final_amazon_table = final_amazon_table['Review_body'].apply(lambda x: [word for word in x if word not in stop_words])
# final_amazon_table.head(100000)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_the_stop_words_from_the_amazon_data(text):
    main_words=word_tokenize(text)
    filtered_chars=[w for w in main_words if w not in stopwords.words('english')]
    return filtered_chars
final_amazon_table.Review_body=final_amazon_table.Review_body.apply(remove_the_stop_words_from_the_amazon_data)

# %% [markdown]
# ## perform lemmatization  

# %%
import nltk
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

amazon_word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
amazon_data_lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_words_of_amazon_data(text):
    lemma = [amazon_data_lemmatizer.lemmatize(w) for w in text]
    return ' '.join(lemma)

final_amazon_table['Review_body'] = final_amazon_table.Review_body.apply(lemmatize_words_of_amazon_data)

mean_of_review_body = final_amazon_table.Review_body.str.len().mean()
print(mean_of_review_body)

# %% [markdown]
# # TF-IDF Feature Extraction

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
tf_idf_extraction_of_amazon_data = TfidfVectorizer()
x_train_amazon_data,x_test_amazon_data,y_train_amazon_data,y_test_amazon_data = train_test_split(final_amazon_table['Review_body'],final_amazon_table['star_rating'],test_size = 0.2)
x_train__amazon_data_tfidf = tf_idf_extraction_of_amazon_data.fit_transform(x_train_amazon_data)
x_test__amazon_data_tfidf = tf_idf_extraction_of_amazon_data.transform(x_test_amazon_data)

# %% [markdown]
# # Perceptron

# %%
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
perceptron_prediction_for_the_dataset = Perceptron()
perceptron_prediction_for_the_dataset.fit(x_train__amazon_data_tfidf,y_train_amazon_data)
predictions_test_of_amazon_data = perceptron_prediction_for_the_dataset.predict(x_test__amazon_data_tfidf)
accuracy_score(predictions_test_of_amazon_data, y_test_amazon_data)
precision = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[0]
recall = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[1]
f1_score = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[2]
print(str(precision[0])+","+ str(recall[0])+","+ str(f1_score[0]))
print(str(precision[1])+","+ str(recall[1])+","+ str(f1_score[1]))
print(str(precision[2])+","+ str(recall[2])+","+ str(f1_score[2]))
print(str(precision[3])+","+ str(recall[3])+","+ str(f1_score[3]))
print(str(precision[4])+","+ str(recall[4])+","+ str(f1_score[4]))
print(str(np.average(precision))+","+ str(np.average(recall))+","+str(np.average(f1_score)))

# %% [markdown]
# # SVM

# %%
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
classification_of_dataset_with_svm = LinearSVC()
classification_of_dataset_with_svm.fit(x_train__amazon_data_tfidf, y_train_amazon_data)
prediction_with_svm = classification_of_dataset_with_svm.predict(x_test__amazon_data_tfidf)
accuracy_score(prediction_with_svm, y_test_amazon_data)
precision_svm = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[0]
recall_svm = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[1]
f1_score_svm = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[2]
print(str(precision_svm[0])+","+ str(recall_svm[0])+","+ str(f1_score_svm[0]))
print(str(precision_svm[1])+","+ str(recall_svm[1])+","+ str(f1_score_svm[1]))
print(str(precision_svm[2])+","+ str(recall_svm[2])+","+ str(f1_score_svm[2]))
print(str(precision_svm[3])+","+ str(recall_svm[3])+","+ str(f1_score_svm[3]))
print(str(precision_svm[4])+","+ str(recall_svm[4])+","+ str(f1_score_svm[4]))
print(str(np.average(precision_svm))+","+ str(np.average(recall_svm))+","+str(np.average(f1_score_svm)))

# %% [markdown]
# # Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
import numpy as np
# instantiate the model (using the default parameters)
classification_of_dataset_with_logreg = LogisticRegression(random_state=16,max_iter=100000)
classification_of_dataset_with_logreg.fit(x_train__amazon_data_tfidf,y_train_amazon_data)
prediction_with_logreg = classification_of_dataset_with_logreg.predict(x_test__amazon_data_tfidf)
accuracy_score(prediction_with_logreg, y_test_amazon_data)
precision_lg = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[0]
recall_lg = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[1]
f1_score_lg = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[2]
print(str(precision_lg[0])+","+ str(recall_lg[0])+","+ str(f1_score_lg[0]))
print(str(precision_lg[1])+","+ str(recall_lg[1])+","+ str(f1_score_lg[1]))
print(str(precision_lg[2])+","+ str(recall_lg[2])+","+ str(f1_score_lg[2]))
print(str(precision_lg[3])+","+ str(recall_lg[3])+","+ str(f1_score_lg[3]))
print(str(precision_lg[4])+","+ str(recall_lg[4])+","+ str(f1_score_lg[4]))
print(str(np.average(precision_lg))+","+ str(np.average(recall_lg))+","+str(np.average(f1_score_lg)))

# %% [markdown]
# # Naive Bayes

# %%
from sklearn.naive_bayes import MultinomialNB
import numpy as np
classification_of_dataset_with_naive_bayes_multinomial = MultinomialNB()
classification_of_dataset_with_naive_bayes_multinomial.fit(x_train__amazon_data_tfidf, y_train_amazon_data)
prediction_with_naive_bayes = classification_of_dataset_with_naive_bayes_multinomial.predict(x_test__amazon_data_tfidf)
accuracy_score(prediction_with_naive_bayes, y_test_amazon_data)
precision_nb = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[0]
recall_nb = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[1]
f1_score_nb = precision_recall_fscore_support(y_test_amazon_data, predictions_test_of_amazon_data)[2]
print(str(precision_nb[0])+","+ str(recall_nb[0])+","+ str(f1_score_nb[0]))
print(str(precision_nb[1])+","+ str(recall_nb[1])+","+ str(f1_score_nb[1]))
print(str(precision_nb[2])+","+ str(recall_nb[2])+","+ str(f1_score_nb[2]))
print(str(precision_nb[3])+","+ str(recall_nb[3])+","+ str(f1_score_nb[3]))
print(str(precision_nb[4])+","+ str(recall_nb[4])+","+ str(f1_score_nb[4]))
print(str(np.average(precision_nb))+","+ str(np.average(recall_nb))+","+str(np.average(f1_score_nb)))

# %%



