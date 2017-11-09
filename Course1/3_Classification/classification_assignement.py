import graphlab


#Loading the data
products = graphlab.SFrame('amazon_baby.gl/');
products.head();

#getting the count of every word in review 
products['word_count'] = graphlab.text_analytics.count_words(products['review']);

def awesome_count(input):
	if 'awesome' in input:
		return input['awesome'];
	else:
	     return 0;

#selecting our own features
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

products['awesome'] = products['word_count'].apply(awesome_count)	   	


#function to check the count of a particular word in dictionary
def mywordcount(input1,input2):
	if selected_words[input2] in input1:
		return input1[selected_words[input2]]
	else:
	     return 0;


i = 0;

#using myword count to add a column corresponding to every new feature in the document

while i<len(selected_words):
	products[selected_words[i]]=products['word_count'].apply(lambda x:mywordcount(x,i))
	i = i+1	;

print("hello")

#adding the word count corresponding to every feature in selected word in a python dictionary
selected_words_count = dict()


#counting occurence of various features across various data set
while i<len(selected_words):
	selected_words_count[selected_words[i]]=products[selected_words[i]].sum()
	i = i+1;

#
train_data,test_data = products.random_split(.8, seed=0)

 my_features= selected_words 

 #setting classes 
 products['sentiment'] = products['rating'] >=4

 #eliminating useless data i.e we can't take this in our data
 products = products[products['rating']!=3];


 #making classifier models on the basis of our features and word count
 selected_words_model = graphlab.logistic_classifier.create(train_data,target='sentiment',features=my_features,validation_set=test_data)
 sentiment_model = graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word_count'],validation_set=test_data)

#evaluating model on test data
 sentiment_model.evaluate(test_data)
#predicting from our model
 diaper_champ = products[products['name']=='Baby Trend Diaper Champ']
 diaper_champ['my_predicted_sentiment'] = selected_words_model.predict(diaper_champ,output_type='probability')
  diaper_champ['original_predicted_sentiment'] = sentiment_model.predict(diaper_champ,output_type='probability')

sentiment_model.show('Coefficients')
selected_words_model.show('Coefficients')

#evaluating model on test data
 sentiment_model.evaluate(test_data)
