# SimpleSQL Library

## Overview
The **SimpleSQL** library provides a simplified interface for connecting to a database, executing queries, and easily printing the results in a formatted table using the **printSql** class. This README provides a simple guide on how to use the key components: **printSql** class, **print** method, and **SimpleSQL** class.

## printSql Class

The **printSql** class facilitates printing SELECT queries and associated tables. It offers functionality to manage headings for the lists and apply colored formatting to the tables.

### How to Use

1. Create a connection to the database and create a cursor object.
2. Initialize a **printSql** instance with the cursor.
3. Print queries directly using the `print` method:

```python
# Example Usage

# Assuming you have a cursor instance named 'cursor'
sql = printSql(cursor)

# Print a simple SELECT query
sql.print("SELECT id, title, description FROM movies;", "Movie Details", color="blue")

# Print a more complex SELECT query with additional options
sql.print(sql_block="""SELECT * FROM people
                       WHERE nationality = 'American';
                    """,
          title="American actors",
          color="Magenta5",
          options={'align':'l'}
         )
```

## SimpleSQL Class

The **SimpleSQL** class simplifies the process of connecting to a database, executing queries, and printing results using the **printSql** class.

### How to Use

1. Initialize an instance of **SimpleSQL** with the desired database name.
2. Utilize the `print` method to print queries and their results in formatted tables.

```python
# Example Usage

# Connect to an existing database named "music"
with SimpleSQL("music") as db:
    db.print("SELECT * FROM musician;", color='Multi1')

# Create a new database "new_db", execute a query, and print the results
with SimpleSQL("new_db", drop_after=True) as db:
    db.execute("""
        -- Create customers table
        CREATE TABLE customers (
        customer_id INT PRIMARY KEY,
        customer_name VARCHAR(50),
        customer_email VARCHAR(50));
                   
        INSERT INTO customers (customer_id, customer_name, customer_email)
        VALUES 
        (1, 'John Doe', 'john@example.com');
        """)

    db.print("SELECT * FROM customers")
```

Ensure you have the necessary libraries (`prettytable`, `psycopg2`, and `stringcolor`) installed to use the provided features.

For further details and advanced usage, refer to the code in the library.