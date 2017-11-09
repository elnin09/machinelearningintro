import graphlab

people = graphlab.SFrame('people_wiki.gl/')
people['word_count'] = graphlab.text_analytics.count_words(people['text'])
tf_idf = graphlab.text_analytics.tf_idf(people['word_count'])
people['tf_idf'] = tf_idf


elton = people[people['name']== 'Elton John']

elton_word_count_table = elton[['word_count']].stack('word_count',new_column_name =['word','count'])
  #the,in,and
  elton_word_count_table.sort('count',ascending = False)

elton_tfidf_count_table = elton[['tf_idf']].stack('tf_idf',new_column_name =['word','count'])
#furnish,elton,billboard
elton_tfidf_table.sort('count',ascending = False)


Knn_model1= graphlab.nearest_neighbors.create(people,features=['word_count'],label='name',distance='cosine');
Knn_model2= graphlab.nearest_neighbors.create(people,features=['tf_idf'],label='name',distance='cosine');

victoria =  people[people['name']== 'Victoria Beckham']
graphlab.distances.cosine(victoria['tf_idf'][0],elton['tf_idf'][0])

#0.9567006376655429
 paul = people[people['name']== 'Paul McCartney']
graphlab.distances.cosine(elen['tf_idf'][0],paul['tf_idf'][0])

#0.9603060844514637

Knn_model2.query(elton)
Knn_model1.query(elton)


