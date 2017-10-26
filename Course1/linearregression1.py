import graphlab;
import matplotlib.pyplot as plt;
#for visualising predictions made by our model

#load data
sales = graphlab.SFrame('home_data.gl/');


sales.show(view='Scatter Plot',x="sqft_living",y="price")
sales.show(view='BoxWhisker Plot', x='zipcode', y='price')

#create a simple regression model of sqft_living to price

#split data into training data and testing data
#setting a seed will split data in the same way
train_data,test_data = sales.random_split(.8,seed=0)


#apply linear regression algorithm
#argument 1= input to data ,argument 2= target ='col name' we are trying to predict
sqft_model = graphlab.linear_regression.create(train_data,target='price',features=['sqft_living'])

print test_data['price'].mean()


#finding rmse (root mean square error in our prediction)
print sqft_model.evaluate(test_data)

pyplot.plot(test_data['sqft_living'],test_data['price'],'.',test_data['sqft_living'],sqft_model.predict(test_data),'-')


sqft_model.get('coefficients') #get wo,w1 and w2


#adding more features for our my_feature
my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','zipcode']
my_features_model = sqft_model = graphlab.linear_regression.create(train_data,target='price',features=my_features);
sqft_model = graphlab.linear_regression.create(train_data,target='price',features=['sqft_living'])

print my_features_model.evaluate(test_data)


#apply learned model to predict three houses

house1 = sales[sales['id']=='5309101200'] 

#actual price 62,000

print sqft_model.predict(house1)
#[627590.9545460119]

 print my_features_model.predict(house1)
#[725809.8720253848]



house2 = sales[sales['id']=='1925069082']
print house2['price']
#[2200000]

print sqft_model.predict(house2)
#[1253998.3935409342]

print my_features_model.predict(house2)

#[1420710.7903599555]




