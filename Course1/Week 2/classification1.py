import graphlab


#Loading the data
products = graphlab.SFrame('amazon_baby.gl/');
products.head();

products['word_count'] = graphlab.text_analytics.count_words(products['review']);

products.show()

giraffe_reviews = products[products['name']=='Vulli Sophie the Giraffe Teether']

giraffe_reviews['rating'].show(view='Categorical')

#now we will define which review have positive or negative comments

products['rating'].show(view = 'Categorical')

#defining what is postive and negative

#ignore all three star reviews

products_modified = products[products['rating']!=3];

#positive sentiment are 4* and 5*
products_modified['sentiment'] = products_modified['rating'] >=4

#sentiments will be either 0 or 1
products_modified[products_modified['sentiment']==1]
products_modified[products_modified['sentiment']==0]

#train the model

train_data,test_data = products_modified.random_split(.8,seed=0)

sentiment_model = graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=test_data)
sentiment_model.evaluate(test_data,metric='roc_curve')

sentiment_model.show(view='Evaluation')



#Apply the model to understand sentiment for Giraffe


giraffe_reviews['mypredictedsentiment'] = sentiment_model.predict(giraffe_reviews,output_type='probability')
giraffe_reviews.head()


giraffe_reviews = giraffe_reviews.sort('mypredictedsentiment')