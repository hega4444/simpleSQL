from simpleSQL import SimpleSQL

# Test for simpleSQL +simpleMiner 
if __name__ == '__main__':
    #Open database
    with SimpleSQL("salaries_plus", drop_after = True) as db:
        # Load style for printing functions 
        db.printer.load_style(options={'align':'l','color':'multi1'})
        
        # Read and create new table in DB
        db.create_table_from_csv("SalaryData2.csv", 
                        table_name= "salaries" , 
                        create_pri_key = True
                        )
        
        # Show some sample data to understand the table data structure
        db.print("SELECT * FROM salaries LIMIT 20;", title='\nData sample of table Salaries')

        # Create a miner instance to test predictive model
        miner = db.new_model(model_type = 'LinearRegression', table_name = 'salaries')

        # Choose which features (inputs) and target (field to be predicted) the model should consider
        # It will extract the data from the tables and manage the neccesary data conversions 
        miner.define_model_features(features = ['Education', 'Gender', 'Experience'], target= ['Salary'])

        # Show model metrics with current settings
        miner.show_model_performance()
        
        # Make predictions based on current model
        print("\nPrediction:", round(   miner.predict({'Age': 35, 
                                                       'Experience': 6, 
                                                       'Gender': 'Female'
                                                        })))
        
        # Set preferences to determine model quality (or quality of the predictions)
        miner.set_custom_r2_limits(high=0.888, moderate=0.8, low=0.2, ranking=4)
        
        # Find which fields weight the most on determining the target value 
        miner.analize_field_relations(target='Salary', hide_unfit=True, show_details=False)

        #db.init_interface()
