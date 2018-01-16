import graphlab

sales = graphlab.SFrame.read_csv('Philadelphia_Crime_Rate_noNA.csv')
sales.show(view="Scatter Plot",x="CrimeRate",y="HousePrice")
crime_model = graphlab.linear_regression.create(sales,target="HousePrice",features=['CrimeRate'],validation_set=None)

sales_nocc = sales[sales['MilesPhila']!=0.0]
sales_nocc.show(view="Scatter Plot",x="CrimeRate",y="HousePrice")
crime_model_nocc = graphlab.linear_regression.create(sales_nocc,target="HousePrice",features=['CrimeRate'],validation_set=None)



#high leverage and influential points 


sales_nohighend = sales[sales['HousePrice']<=350000]
crime_model_nohighend = graphlab.linear_regression.create(sales_nohighend,target="HousePrice",features=['CrimeRate'],validation_set=None)


crime_model.get('coefficients')
crime_model_nocc.get('coefficients')
crime_model_nohighend.get('coefficients')