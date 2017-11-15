import graphlab

image_train = graphlab.SFrame('image_train_data/')

#train a nearest neighbor model for retrieving images using deep_features

knn_model = graphlab.nearest_neighbors.create(image_train,features=['deep_features'],label='id')

#use image retrieval model with deep_features

cat = image_train[18:19]
cat['image'].show()

knn_model.query(cat)



def get_images_ids(query_result):
	return image_train.filter_by(query_result['reference_label'],'id')


cat_neighbors = get_images_ids(knn_model.query(cat))

cat_neighbors['image'].show()


car = image_train[8:9]
car['image'].show()


car_neighbors= get_images_ids(knn_model.query(car))


show_neighbors = lambda i:get_images_ids(knn_model.query(image_train[i:i+1]))