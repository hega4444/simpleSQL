from simpleSQL import SimpleSQL

# Test for simpleSQL + simpleMiner 
if __name__ == '__main__':
    #Open database
    with SimpleSQL("test_salaries", drop_after = True) as db:
        # Load style for printing functions 
        db.printer.load_style(options={'align':'l','color':'multi1'})
        
        # Read and create new table in DB
        db.create_table_from_csv("SalaryData2.csv", 
                        table_name= "salaries" , 
                        create_pri_key = True
                        )
        # Open GUI for friendlier db administration
        db.simple_interface(load_script = "load_miner")        