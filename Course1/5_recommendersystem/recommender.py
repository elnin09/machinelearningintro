import graphlab

song_data = graphlab.SFrame('song_data.gl/')

song_data['song'].show()


len(song_data)

users = song_data['user_id'].unique()

len(users)

#create a song recommender


train_data,test_data = song_data.random_split(.8,seed=0)

popularity_model = graphlab.popularity_recommender.create(train_data,user_id='user_id',item_id='song');


 popularity_model.recommend(users=[users[1]])

similarity_model = graphlab.item_similarity_recommender.create(train_data,user_id='user_id',item_id='song');

#applying the personalised

personalised_model = similarity_model

personalised_model.recommend(users=[users[0]])

personalised_model.recommend(users=[users[1]])

personalised_model.get_similar_items(['With Or Without You - U2'])

personalised_model.get_similar_users([])


#quantitative comparison between models


model_performance = graphlab.recommender.util.compare_models(test_data,[popularity_model,personalised_model],user_sample=0.05)