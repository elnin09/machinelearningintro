import graphlab


#load some data from wikipedia,pages on people

people = graphlab.SFrame('people_wiki.gl/')

len(people)


#explore the data set and check the text it contains

obama = people[people['name']== 'Barack Obama']

obama['text'];


#Get the word count for obama article
obama['word_count'] = graphlab.text_analytics.count_words(obama['text'])


#sort the word count for obama article
obama_word_count_table = obama[['word_count']].stack('word_count',new_column_name =['word','count'])


obama_word_count_table.sort('count',ascending = False)

#computing and exploring tf-idf

people['word_count'] = graphlab.text_analytics.count_words(people['text'])
# a list array with dictionary elements

tf_idf = graphlab.text_analytics.tf_idf(people['word_count'])

people['tf_idf'] = tf_idf
#tf_idf is a list of dictionaries

obama = people[people['name']== 'Barack Obama']


obama[['tf_idf']].stack('tf_idf',new_column_name=['word','tfidf']).sort('tfidf',ascending = False)

clinton = people[people['name']== 'Bill Clinton']
beckham = people[people['name']== 'David Beckham']

#is obama closer to clinton than to beckham

graphlab.distances.cosine(obama['tf_idf'][0],clinton['tf_idf'][0])
graphlab.distances.cosine(obama['tf_idf'][0],beckham['tf_idf'][0])



#build a nearest neighbour model for document retrieval
Knn_model= graphlab.nearest_neighbors.create(people,features=['tf_idf'],label='name');


#make our first query
Knn_model.query(obama)


swift =  people[people['name']== 'Taylor Swift']
Knn_model.query(swift)

jolie = people[people['name']== 'Angelina Jolie']


ans = Knn_model.query(jolie);


arnold = people[people['name']== 'Arnold Schwarzenegger'] 


Knn_model.query(arnold)
