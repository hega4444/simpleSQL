from simpleSQL import SimpleSQL

# Test for simpleSQL + simpleMiner 
if __name__ == '__main__':
    #Open database
    with SimpleSQL("simpledb", drop_after = True) as db:
        # Load style for printing functions 
        db.printer.load_style(options={'align':'l','color':'multi1'})
        
        # Read and create new table in DB
        db.create_table_from_csv("data/SalaryData2.csv", 
                            table_name= "salaries" ,
                            create_pri_key = True
                            )
        # Open GUI for friendlier db administrationmodel 
        db.simple_interface(load_script = "load__miners")         