import graphlab;

sales = graphlab.SFrame('kc_house_data.gl/')
train_data,test_data = sales.random_split(.8,seed=0)


#define your linear regression function which will return slope and intercept
#works on the close form method where partial derivate is set to 0
def simple_linear_regression(input_feature, output):
	prodxy = input_feature*output;
    sumx = input_feature.sum(); 
    sumy = output.sum();
    sumx2 = (input_feature*input_feature).sum();
    sumprodxy = prodxy.sum();
    n = input_feature.size()+0.0;
    slope = (sumprodxy- (sumx*sumy)/n)/(sumx2-(sumx*sumx)/n);
    intercept = sumy/n - slope*(sumx/n);
    return (intercept,slope) 

#define input
my_input=train_data['sqft_living'];
output =train_data['price']

params = simple_linear_regression(my_input,output);

#return predicted colum when input column and parameters od the model are known
def get_regression_predictions(input_feature, intercept, slope):
     predicted_output = input_feature*slope+intercept;
     return predicted_output;

train_data['predicted_output'] = get_regression_predictions(my_input,params[0],params[1]); 

#predicting price for house having size sqft living = 2650
print train_data[train_data['sqft_living']==2650]['predicted_output']
#700074.8456294581


#calculate RSS for model where input_feature,output and model parameters are known
def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    errorperrow=(input_feature*slope+intercept-output+0.0)**2     
    return errorperrow.sum()

print get_residual_sum_of_squares(train_data['sqft_living'],train_data['price'],params[0],params[1]);    


#calulate sqft_living from the  output and model parametars
def inverse_regression_predictions(output, intercept, slope):
    input = (output-intercept)/(slope+0.0)
    return input

print inverse_regression_predictions(800000,params[0],params[1]);
#3004.39624762

#Train second model
params2 = simple_linear_regression(train_data['bedrooms'],train_data['price'])


#calculate RSS of two models on test data
RSS1 =  get_residual_sum_of_squares(test_data['sqft_living'],test_data['price'],params[0],params[1]);
RSS2 =  get_residual_sum_of_squares(test_data['bedrooms'],test_data['price'],params2[0],params2[1]);  








