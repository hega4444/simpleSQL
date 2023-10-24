-- Create customers table
CREATE TABLE customers (
    id INT PRIMARY KEY,
    name VARCHAR(50),
    email VARCHAR(50));
-- Insert 5 entries into customers table
INSERT INTO customers (id, name, email)
VALUES 
    (1, 'John Doe', 'john@example.com'),
    (2, 'Jane Smith', 'jane@example.com'),
    (3, 'Michael Johnson', 'michael@example.com'),
    (4, 'Sarah Davis', 'sarah@example.com'),
    (5, 'David Brown', 'david@example.com');
-- Create orders table with a foreign key referencing customers table
CREATE TABLE orders (
    order_id INT PRIMARY KEY,
    amount DECIMAL(10, 2),
    customer_id INT,
    FOREIGN KEY (customer_id) REFERENCES customers(id));
-- Insert 5 entries into orders table
INSERT INTO orders (order_id, amount, customer_id)
VALUES
    (101, 100.00, 1),
    (102, 150.50, 2),
    (103, 75.20, 3),
    (104, 200.00, 3),
    (105, 300.00, 2);
