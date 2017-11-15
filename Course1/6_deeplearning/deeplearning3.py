import graphlab

image_train = graphlab.SFrame('image_train_data/')


image_train['label'].sketch_summary()

dog_data = image_train[image_train['label'] == 'dog']
cat_data = image_train[image_train['label'] == 'cat']
bird_data = image_train[image_train['label'] == 'bird']
automobile_data = image_train[image_train['label'] == 'automobile']

knn_model_dog = graphlab.nearest_neighbors.create(dog_data,features=['deep_features'],label='id')
knn_model_cat = graphlab.nearest_neighbors.create(cat_data,features=['deep_features'],label='id')
knn_model_bird = graphlab.nearest_neighbors.create(bird_data,features=['deep_features'],label='id')
knn_model_automobile = graphlab.nearest_neighbors.create(automobile_data,features=['deep_features'],label='id')


image_test = graphlab.SFrame('image_test_data/')


def get_myimages_ids(query_result , mydata):
	return mydata.filter_by(query_result['reference_label'],'id')

knn_model_dog.query(image_test[0:1])
cat_neighbors = get_myimages_ids(knn_model_cat.query(image_test[0:1]),cat_data)
cat_neighbors.show()

dog_neighbors_1 = get_myimages_ids(knn_model_dog.query(image_test[0:1]) , dog_data)
dog_neighbors_1.show()


ans1 = knn_model_dog.query(image_test[0:1]);
ans1['distance'].mean()
ans2 = knn_model_cat.query(image_test[0:1]);
 ans2['distance'].mean()
#split the test data into labels as well


dog_test_data = image_test[image_test['label'] == 'dog']
cat_test_data = image_test[image_test['label'] == 'cat']
bird_test_data = image_test[image_test['label'] == 'bird']
automobile_test_data = image_test[image_test['label'] == 'automobile']


dog_cat_neighbors = knn_model_cat.query(dog_test_data, k=1)
dog_dog_neighbors = knn_model_dog.query(dog_test_data, k=1)
dog_automobile_neighbors = knn_model_automobile.query(dog_test_data, k=1)
dog_bird_neighbors = knn_model_bird.query(dog_test_data, k=1)


dog_distances[‘dog-dog’] = dog_dog_neighbors[‘distance’]

dog_distances[‘dog-cat’] = dog_cat_neighbors[‘distance’]

dog_distances[‘dog-automobile’] = dog_automobile_neighbors[‘distance’]

dog_distances[‘dog-bird’] = dog_bird_neighbors[‘distance’]


dog_distances = graphlab.SFrame(
	{'dog-dog':dog_dog_neighbors['distance'],'dog-cat':dog_cat_neighbors['distance'],'dog-automobile':dog_automobile_neighbors['distance'],'dog-bird':dog_bird_neighbors['distance']})


#ideally dog_dog distance must be smallest because by distance we find label and as such
#dog should be labelled as dog

def is_dog_correct_output(row):
	if(row['dog-dog'] < row['dog-cat']   and row['dog-dog'] < row['dog-bird']   and row['dog-dog'] < row['dog-automobile'] ):
		return 1
	else:
		return 0


dog_distances['prediction'] = dog_distances.apply(is_dog_correct_output)
#row will be 1 if correct prediction is made

#check how many correct predcitions made
a = dog_distances['prediction'].sum() + 0.0

b = len(dog_distances)

accuracy =  a/b;




