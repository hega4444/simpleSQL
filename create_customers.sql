-- Step 1: Create the customer table
CREATE TABLE customer (
    customer_id serial PRIMARY KEY,
    first_name varchar(50),
    last_name varchar(50),
    email varchar(100),
    phone_number varchar(20),
    address varchar(100),
    city varchar(50),
    state varchar(2),
    zip_code varchar(10),
    registration_date date
);

-- Step 2: Insert 100 dummy entries
INSERT INTO customer (first_name, last_name, email, phone_number, address, city, state, zip_code, registration_date)
SELECT
    'First Name ' || i,
    'Last Name ' || i,
    'email' || i || '@example.com',
    '(123) 456-789' || i,
    '1234 Elm Street',
    'City ' || i,
    'CA',
    '9' || i || '0000',
    CURRENT_DATE - (i || ' days')::interval
FROM generate_series(1, 100) AS i;

-- Step 3: Check the inserted data
SELECT * FROM customer;
