import gzip
import sqlite3

# Path to the PostgreSQL dump file
dump_file = 'tezaurs_current-public.pgsql.gz'

# Connect to the SQLite database (creates a new database if it doesn't exist)
connection = sqlite3.connect('mydatabase.db')

# Create a cursor object
cursor = connection.cursor()

# Create tables and insert data from the dump file
with gzip.open(dump_file, 'rt') as file:
    sql_statements = file.read()

    # Split the dump file into individual SQL statements
    statements = sql_statements.split(';')

    # Execute each SQL statement
    for statement in statements:
        cursor.execute(statement)

# Commit the changes
connection.commit()

# Close the cursor and connection
cursor.close()
connection.close()