from simpleSQL import SimpleSQL

with SimpleSQL("join_demo_db", load_file="load_db.sql", drop_after=True) as db:

    join_type = ['JOIN orders ON customers.id = orders.customer_id', 
                 'LEFT JOIN orders ON customers.id = orders.customer_id', 
                 'RIGHT JOIN orders ON customers.id = orders.customer_id', 
                 'FULL JOIN orders ON customers.id = orders.customer_id',
                 'CROSS JOIN orders']

    for join_statement in join_type:

        db.print(sql_block = f"""SELECT customers.name, orders.order_id 
                            FROM customers
                            {join_statement} ;
                            """, 
                title=f"Results of query using {join_statement}",
                color="Multi2")
    

    


