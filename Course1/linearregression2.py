import graphlab
sf= graphlab.SFrame('home_data.csv');

sf.show(view='BoxWhisker Plot', x='zipcode', y='price')
houses = sf[sf['zipcode']==98039]
print houses['price'].mean()

#output 2160606.6


houses_modified = houses[(houses['sqft_living']>2000) & (houses['sqft_living']<=4000)]

houses.show() #49
houses_modified.show() #24

ratio = 24.0/49

 advanced_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode','condition','grade','waterfront','view','sqft_above','sqft_basement','yr_built','yr_renovated','lat','long','sqft_living15','sqft_lot15']

advanced_features=['bedrooms','bathrooms','sqft_living','sqft_lot']

sales = graphlab.SFrame('home_data.gl/');

train_data,test_data = sales.random_split(.8,seed=0)

#train_data,test_data = sales.random_split(.8,seed=0)



advanced_model = graphlab.linear_regression.create(train_data,target='price',features=advanced_features,validation_set=None);

my_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']

basic_model  = graphlab.linear_regression.create(train_data,target='price',features=my_features,validation_set=None);

print basic_model.evaluate(test_data)
#{'max_error': 4008069.470364224, 'rmse': 252581.90249380007}

print advanced_model.evaluate(test_data)
#{'max_error': 4111411.383242067, 'rmse': 199962.79757368553}

answer = basic_model.evaluate(test_data)['rmse']-advanced_model.evaluate(test_data)['rmse']

#22711