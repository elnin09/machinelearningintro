import graphlab


#loading train and test data

image_train = graphlab.SFrame('image_train_data/')

image_test = graphlab.SFrame('image_test_data/')

#train a classifier using raw image pixels

raw_pixel_model = graphlab.logistic_classifier.create(image_train,target='label',features=['image_array'])

#make a prediction 

image_test[0:3]['image'].show()

image_test[0:3]['label']

raw_pixel_model.predict(image_test[0:3])

raw_pixel_model.evaluate(image_test)


#improving model using deep features
len(image_train)

#deep features already there no need to execute following two commands 
deep_learning_model = graphlab.load_model('imagenet_model')

image_train['deep_features'] = deep_learning_model.extract_features(image_train)