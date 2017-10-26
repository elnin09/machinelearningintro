import graphlab

#Loading from a file 
sf= graphlab.SFrame('/Users/darmora/Desktop/Random/EventRegistration.csv');

sf #we can view first few lines
sf.head() #this does same thing as sf
sf.tail() 
#graphlabCCANVAS 
#Take any data structure

sf.show()  #will publish it on a webpage in localhost

graphlab.canvas.set_target('ipynb')

sf[''].show(view='Categorical')


#inspect some column of dataset
sf['column name'].mean()
sf['column name'].age()

#creating a column with other column
sf['full name'] = sf['first name'] + sf['last name']

#modifying an existing column
sf['col'] = sf['col']+2;

sf['Country'] #may have different values of country


def transformcountry(country):
	if country == 'USA':
		return 'UnitedStates'
	else:
		return country


#apply someopretaion to column
sf['Country'].apply(transformcountry);





