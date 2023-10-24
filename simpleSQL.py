#SimpleSQL library
#
#SimpleSQL Class - Handles connection to the database using a WITH block
#                - Opens or creates a new database if needed
#                - If drop_after=True, at the end of the WITH block will DROP database (useful for demos)
#
#printSql Class - Provides an easy way to print SELECT queries and tables
#               - Manages the headings for the lists
#               - Color tables
#how to use:
#
# 1. Create a connection to the db using simpleSql(),
# 2. Print queries directly: 
#                           with  SimpleSql("database_name") as sql
#
#                               sql.print("SELECT id, title, description FROM movies;", "Some nice title", color="blue")
#
#                               sql.print(sql_block="""SELECT * FROM people
#                                                      WHERE nationality = 'American';
#                                                   """, 
#                                       title="American actors", 
#                                       color="Magenta5", 
#                                       options={'align':'l'}
#                                       )
#
from simpleMiner import SimpleMiner

from typing import Any
from prettytable import PrettyTable
import psycopg2
from stringcolor import *
from stringcolor.ops import _load_colors
import datetime
import csv
import os
import readline
import rlcompleter


class PrettyTablePlus(PrettyTable):

    def __init__(self, field_names=None, **kwargs) -> None:
        super().__init__(field_names, **kwargs)
        if 'color' in kwargs:
            self.color = kwargs['color']
        else:
            self.color = None
    
    def get_string(self, **kwargs) -> str:
        lines = super().get_string(**kwargs)
        splitted_lines = lines.split("\n")

        pallete = {}
        pallete['multi1'] = ['DeepPink3', 'DeepPink2', 'DeepPink', 'HotPink', 'IndianRed', 'IndianRed2', 'DarkOrange', 'Red2', 'Salmon']
        pallete['multi2'] = ['DeepSkyBlue5', 'DodgerBlue3', 'DodgerBlue2', 'Green4', 'SpringGreen5','Turquoise4', 
                            'DeepSkyBlue3', 'DeepSkyBlue4', 'DodgerBlue', 'Green3',   
                            'SpringGreen4', 'DarkCyan', 'LightSeaGreen', 'DeepSkyBlue2', 'DeepSkyBlue']

        new_lines = ""

        if self.color and self.color.lower() in pallete:
            pallete_n = self.color.lower()
            cols = pallete[pallete_n]
            n_colors = len(cols)
            
            if self.color.lower() == 'multi1':
                n = 5
            else:
                n = 3
        
            for line_i, line in enumerate(splitted_lines):
                i = 0
                for c in line:
                    nc = cs(c, cols[((i + line_i ) // n)  % n_colors]) if c != "\n" else c
                    new_lines += nc
                    i += 1
                new_lines += "\n"
                print(new_lines, end="")
                new_lines = ""
            new_lines = ""

        elif self.color:
            new_lines = str(cs(lines, self.color))
        else:
            new_lines = lines

        return new_lines
    
    def __str__(self):
        return super().__str__()

class printSql():

    def __init__(self, cursor) -> None:
        self.cur = cursor
        self.saved_options = None
    
    def execute(self, query, *args):
        try:
            return self.cur.execute(query, args)
        except Exception as e:
            print(cs(f"WARNING - SQL syntax error: {e}", color="Red"))
 
    def load_style(self, options):
        self.saved_options = options
    
    def reset_style(self):
        self.saved_options = {}
        
    def print(self, sql_block="", title="", color = None, options = None):
        #Prints a table with the results of a query

        #Execute the query
        try:
            self.cur.execute(sql_block)
        except Exception:
            self.cur.execute("rollback;")
            try:
                self.cur.execute(sql_block)
            except Exception:
                pass
                
        try:
            #Get data
            data = self.cur.fetchall()
            
            #Get the titles of that specifc query
            column_names = [desc[0].capitalize() for desc in self.cur.description]

        except Exception:
            data = [['No data to show.']]
            column_names = ['--']
        
        if options == None and self.saved_options:
            options = self.saved_options
        
        if color == None and (options==None or 'color' not in options) and \
            self.saved_options and 'color' in self.saved_options:
            color = self.saved_options['color']
        elif color == None and options and 'color' in options:
            color = options['color']

        if color:
            if options:
                options['color'] = color
            else:
                options = {'color': color}

        #if isinstance(color, str):
        #    options['color'] = color

        if options:
            table = PrettyTablePlus(**options)
            
        else:
            table = PrettyTablePlus()
        table.field_names = column_names
        for row in data:
            table.add_row(row)
        print(cs(f"{title}", color if color else f"{title}:"))
        print(table)
    
    def show_color_list(self):

        print('List of colors:')
        colors = _load_colors()
        for i, c in enumerate(colors.values()):
            print(cs(f"{c['name']}  ", color=c['name']), end="", flush=True)
            if (i+1) % 5 == 0:
                print()
        return
    
class SimpleSQL():
    #This is a simple class to open/ connect (or create if does not exists) to a Database and query it without handling connections.
    #It also wraps PrettyTablePlus so you can print your results very easily.

    def __init__(self, db_name = None, force = True, load_file = None, drop_after = False) -> None:

        self.db_name = db_name
        self.force = force
        self.conn = None
        self.cur = None
        self.printer = None
        self.drop_after = drop_after
        self.load_file = load_file
        self.interface = None
    
    def connect_to_postgres(self):
        #Connect to posgress as ADMIN to access and connect with the DB
        return psycopg2.connect(dbname='postgres',
                                user='postgres',
                                password='postgres',
                                host='localhost',
                                port=5432)

    def create_database(self):
        #Method to create new DB (in case it does not exists) 
        conn = self.connect_to_postgres()
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            cursor.execute(f'CREATE DATABASE {self.db_name}')
        except psycopg2.Error as e:
            print(f'Failed to create DB. {e}')
            return None
        
        cursor.close()
        conn.close()
    
    def drop_database(self, msg = True):
        #Method to drop new DB (in case it was just for a demo / test) 
        conn = self.connect_to_postgres()
        conn.autocommit = True
        cursor = conn.cursor()
        try:
            cursor.execute(f'DROP DATABASE {self.db_name}')
        except psycopg2.Error as e:
            if msg is True:
                print(f'Failed at deleting DB {self.db_name}. {e}')
            return None
        
        cursor.close()
        conn.close()
        

    def connect_to_database(self):
        #Connect to database
        try:
            return psycopg2.connect(dbname=self.db_name,
                                    user='postgres',
                                    password='postgres',
                                    host='localhost',
                                    port=5432)
        except Exception:
            if self.force:
                try:
                    self.create_database() #database not found and force is True, then creates a new one
                except Exception:
                    print(cs(f"WARNING-simpleQL : Failing at creating Database {self.db_name}.", color="Red"))
                return psycopg2.connect(dbname=self.db_name,
                        user='postgres',
                        password='postgres',
                        host='localhost',
                        port=5432)
            else:
                print(cs(f"WARNING-simpleQL : Database {self.db_name} does not exist.", color="Red"))
    
    def print(self, sql_block="", title="", color=None, options=None):
        
        if not options:
            res = self.printer.print(sql_block, title, color)
        else:
            res = self.printer.print(sql_block, title, color, options)

        if "SELECT" not in sql_block.upper():   #probably an UPDATE or DELETE
            self.conn.commit()                  #commit changes just in case

        return res
    
    def execute(self, query, *args):
        try:
            res = self.cur.execute(query, args)
            
            if "SELECT" not in query.upper():   #probably an UPDATE or DELETE
                self.conn.commit()                  #commit changes just in case

            return res
        except Exception as e:
            print(cs(f"WARNING-simpleQL : {e}", color="Red"))
            if self.conn:
                self.conn.rollback()
            
    def run_file(self, file_name):
        try:
            with open(file_name, "r") as file:
                lines = file.readlines()
                sql_script_file = ''.join(lines)
        except FileNotFoundError:
            print(cs(f"WARNING-simpleSQL : file {file_name} not found.", color="Red"))
            return

        self.execute(sql_script_file)
    
    def check_table_exists(self, table_name):
        self.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s);", (table_name,))
        return self.cur.fetchone()[0]
    
    def get_table_names(self):
        self.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        try:
            return [row[0] for row in self.cur.fetchall()]
        except psycopg2.ProgrammingError as e:
            self.conn.rollback()
        return

    def get_table_name_from_oid(self, table_oid):
        self.cur.execute("""
            SELECT relname
            FROM pg_class
            WHERE oid = %s;
        """, (table_oid,))
        return self.cur.fetchone()[0]
        
    def get_primary_key_columns(self, table_name):
        self.cur.execute("""
            SELECT
                kcu.column_name
            FROM
                information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
            WHERE
                tc.table_name = %s
                AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY
                kcu.ordinal_position;
        """, (table_name,))
        return [row[0] for row in self.cur.fetchall()]
    

    def print_tables_names(self):
        self.print("""SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name;""", options={'align':'l', 'color':'Multi1'})
        
    def get_table_fields(self, table_name):
        self.execute(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}';    
                     """)
        records = self.cur.fetchall()
        fields = [name[0].capitalize() for name in records]
        return fields

    def print_table_fields(self, table_name):
        try:
            self.print(f"""
                        SELECT c.column_name, c.data_type,
                            CASE WHEN k.column_name IS NOT NULL THEN 'Primary Key' ELSE '' END AS key_type
                        FROM information_schema.columns c
                        LEFT JOIN information_schema.key_column_usage k
                        ON c.column_name = k.column_name
                        AND c.table_name = k.table_name
                        AND c.table_schema = k.table_schema
                        WHERE c.table_name = '{table_name}';
                """, 
                title=f"\nTable {table_name} - Fields",
                options={'align' : 'l'})
        except Exception:
            pass
        
    
    def get_table_size(self, table_name):
        try:
            self.execute(f"""
                SELECT pg_size_pretty(pg_relation_size('{table_name}')) AS table_size;
                """)
            records = self.cur.fetchall()
            return records[0][0]
        except Exception:
            return 0
        
    def get_table_length(self, table_name):
        try:
            self.execute(f"""
                SELECT COUNT(*) FROM {table_name};
                """)
            records = self.cur.fetchall()
            return records[0][0]
        except Exception:
            return 0


    def preview(self):

        print(f'Preview of database {self.db_name}\n')
        tables = self.get_table_names()

        self.print_tables_names()
        
        if tables:
            for table in tables:
                self.print(f"SELECT * from {table} LIMIT 1;", title=f"Preview of table {table}:")
    
    def into_dict(self, sql_block):
        #Executes a SELECT query and saves it into a dictionary
        try:
            self.cur.execute(sql_block)
            records = self.cur.fetchall()
            #if len(records) == 0:
            #    return {}

            self.execute(sql_block)
            rows = self.cur.fetchall()
            #Get the titles of that specifc query
            column_names = [desc[0].capitalize() for desc in self.cur.description] 
            #column_names = [desc[0] for desc in self.cur.description] #capitalize was creating confussion in the pandas DF
            result = {}

            #Check for primary key:
            # Get the table name from the cursor description (assuming single table query)
            table_name = self.get_table_name_from_oid(self.cur.description[0].table_oid)
            pri_key = None
            pri_key = self.get_primary_key_columns(table_name)
            
            use_pri_key = False
            if pri_key[0] == column_names[0].lower():
                use_pri_key = True
                
            for n, row in enumerate(rows):
                new_record = {}
                for i, column_name in enumerate(column_names):
                    new_record[column_name] = row[i]                
                
                key = row[0] if use_pri_key is True else n

                result[key] = new_record #uses the row number for indexing the dictionary

            return result

        except Exception as e:
            return {}
        
    def create_table_from_dict(self, dictionary, table_name, create_pri_key = False):

        def data_type(var):

            if isinstance(var, str):
                return "VARCHAR"
            
            elif isinstance(var, int):
                return "INT"
            
            elif isinstance(var, float):
                return "DECIMAL"
            
            elif isinstance(var, bool):
                return "BOOLEAN"
            
            elif isinstance(var, datetime.datetime):
                return "TIMESTAMP"
            
            elif isinstance(var, datetime.date):
                return "DATE"
            
            elif isinstance(var, datetime.time):
                return "TIME"
                
        pri_key = []
        use_dictionary_values_as_key = False

        records = list(dictionary.values())
        sample = records[0]
        n_fields = len(sample)
        max_lenghts = [0] * n_fields

        create = False

        if self.check_table_exists(table_name) is not True: #only if table does not exist
            
            create = True
            for record in records:
                if len(record) != n_fields:
                    raise ValueError(cs(f"ERROR-simpleSQL : Dictionary records differ in column numbers.", color="Red"))

                #Verify if the first value can be used as primary key
                pri_key.append(list(record.values())[0])

                #Check for appropiate size of field
                for i, field_value in enumerate(record.values()):
                    #only for VARCHAR (for now)
                    if isinstance(field_value, str) and (l:=len(field_value)) > max_lenghts[i]:
                        max_lenghts[i] = l
            
            #Truncate size fields to a multiple of 10
            max_lenghts = [max + (10 - max % 10) if max > 0 else 0 for max in max_lenghts]

            if len(dictionary) == len(list(set(pri_key))):
                use_dictionary_values_as_key = True
            
            use_dictionary_values_as_key = use_dictionary_values_as_key and not create_pri_key

            try:
                column_names = list(sample.keys())
            except Exception:
                raise ValueError(cs(f"ERROR-simpleSQL : Dictionary format is incorrect.", color="Red"))

            #Begin to build the structure of the SQL statement 
            sql_create_block = f"CREATE TABLE {table_name} (\n"

            if use_dictionary_values_as_key:
                field_type = data_type(list(sample.values())[0])
                pri_key_def = f"{column_names[0].lower()} {field_type} PRIMARY KEY,\n"
                start = 1
            else:
                pri_key_def = "id INT PRIMARY KEY,\n"               #here we need to consider change to SERIAL and forget about numbering!!!!!
                start = 0
            
            sql_create_block += pri_key_def

            for i, field in enumerate(list(sample.values())):
                if i >= start:
                    field_type = data_type(field)
                    field_def = f"{column_names[i].lower()} "
                    if field_type == 'VARCHAR':
                        field_type += f'({max_lenghts[i]})'

                    field_def += f'{field_type}{"," if i<len(sample)-1 else ""} \n'

                    sql_create_block += field_def
            
            sql_create_block += ");"
            
            #Table creation ----------------------------------------------------------------

        for record in records:
            if len(record) != n_fields:
                raise ValueError(cs(f"ERROR-simpleSQL : Dictionary records differ in column numbers.", color="Red"))

            #Verify if the first value can be used as primary key
            pri_key.append(list(record.values())[0])

            #Check for appropiate size of field
            for i, field_value in enumerate(record.values()):
                #only for VARCHAR (for now)
                if isinstance(field_value, str) and (l:=len(field_value)) > max_lenghts[i]:
                    max_lenghts[i] = l
        
        #Truncate size fields to a multiple of 10
        max_lenghts = [max + (10 - max % 10) if max > 0 else 0 for max in max_lenghts]

        if len(dictionary) == len(list(set(pri_key))):
            use_dictionary_values_as_key = True

        try:
            column_names = list(sample.keys())
        except Exception:
            raise ValueError(cs(f"ERROR-simpleSQL : Dictionary format is incorrect.", color="Red"))

        #Begin to build the structure of the SQL statement 
        sql_create_block = f"CREATE TABLE {table_name} (\n"

        if use_dictionary_values_as_key:
            field_type = data_type(list(sample.values())[0])
            pri_key_def = f"{column_names[0].lower()} {field_type} PRIMARY KEY,\n"
            start = 1
        else:
            pri_key_def = "id INT PRIMARY KEY,\n"
            start = 0
        
        sql_create_block += pri_key_def

        for i, field in enumerate(list(sample.values())):
            if i >= start:
                field_type = data_type(field)
                field_def = f"{column_names[i].lower()} "
                if field_type == 'VARCHAR':
                    field_type += f'({max_lenghts[i]})'

                field_def += f'{field_type}{"," if i<len(sample)-1 else ""} \n'

                sql_create_block += field_def
        
        sql_create_block += ");"
        
        #Table creation
        if self.check_table_exists(table_name) is not True: #only if table does not exist

            try:  
                self.execute(sql_create_block)

            except Exception as e:
                print(cs(f"WARNING - SQL syntax error: {e}", color="Red"))
        
            
        #Check if the sample record matches number of fields and types
        self.cur.execute(f"""SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = '{table_name}';
                        """)

        table_details = self.cur.fetchall()
        column_names = list(sample.keys())
        sample = list(sample.values())
        #check if table has at least one record to compare

        if len(table_details) == len(sample):  #dictionary has correct number of fields
            start = 1
        elif len(table_details) == len(sample) + 1: #additional index probably
            start = 0
        else:
            print(cs(f"WARNING-simpleSQL : Dictionary fields do not comply with table.", color="Red"))
            return

        next_index = 0
        if not create and start == 0: #find last record to update index:
            self.cur.execute(f"SELECT max(id) FROM {table_name} LIMIT 1;")
            try:
                max_recovered = self.cur.fetchall()[0]
                max_recovered = max_recovered[0]
                next_index = max_recovered +1
            except Exception:
                print(cs(f"WARNING-simpleSQL : Error trying to resolve index for new entries", color="Red"))

        #Load records 
        sql_insert_block = f"INSERT INTO {table_name} ("
        if start == 0: # Needs an additional field for index
            sql_insert_block += "id, "

        for i, field_name in enumerate(column_names):
            sql_insert_block += field_name.lower() + (", " if i<len(column_names)-1 else ")\n")
        
        sql_insert_block += "VALUES\n"
        #values
        for n, record in enumerate(records):
            record_def = "("

            if start == 0:
                record_def += str(next_index + n) + ", "

            for i, field_value in enumerate(list(record.values())):

                if isinstance(field_value, (int, float)):
                    record_def += str(field_value)
                else:
                    field_value = field_value.replace("'", "''")
                    record_def += f"'{field_value}'"

                record_def += f'{", " if i<len(sample)-1 else ")"}'

            record_def += f'{"," if n<len(records)-1 else ";"}\n' #end of each record
            
            sql_insert_block += record_def

        try:
            self.execute(sql_insert_block)

        except Exception as e:
            print(cs(f"WARNING! - SQL syntax error: {e}", color="Red"))
            return

    def create_table_from_csv(self, file_name, table_name = None, create_pri_key = None):

        def determine_numeric_type(s):
            try:
                int_value = int(s)
                float_value = float(s)
                return "integer"  # The string can be stored as both an integer and a float
            except ValueError:
                try:
                    float_value = float(s)
                    return "float"  # The string can be stored as a float
                except ValueError:
                    return False  # The string is not numeric

        table_data = {}
        try:
            with open(file_name, "r") as file:
                csv_records = csv.DictReader(file)

                table_data = {}

                for n, record in enumerate(csv_records):

                    key = n
                    cleaned_record = {}

                    for i, field in enumerate(record.keys()):

                        if determine_numeric_type(record[field]) == "integer":
                            try:
                                cleaned_record[field] = int(record[field])
                            except Exception:
                                try:
                                    cleaned_record[field] = float(record[field])
                                except Exception:
                                    print(cs(f"WARNING! - SQL Error converting CSV values.", color="Red"))

                        elif determine_numeric_type(record[field]) == "float":
                            try:
                                cleaned_record[field] = float(record[field])
                            except Exception:
                                    print(cs(f"WARNING! - SQL Error converting CSV values.", color="Red"))
                        
                        else:
                            cleaned_record[field] = record[field]

                    table_data[key]=(dict(cleaned_record))

        except FileNotFoundError:
            print(cs(f"WARNING! - SQL CSV File {file_name} not found.", color="Red"))
            return

        self.create_table_from_dict(table_data, table_name = table_name, create_pri_key = create_pri_key)

    def new_model(self, model_type, table_name):
        return SimpleMiner(self, model_type, table_name)
    
    def new_table(self, options):
        return PrettyTablePlus(**options)
        
    def simple_interface(self, load_script = None):
        interface = SimpleInterface(self)
        interface.interface_init(load_script = load_script)
        self.interface = interface
        return interface

    def __enter__(self):
        #Handler for openning a new connection to DB
        if self.drop_after is True:
            try:
                self.drop_database(msg = False)
            except Exception:
                pass

        try:
            self.conn = self.connect_to_database() 
            self.cur = self.conn.cursor()

            if self.load_file:
                self.run_file(self.load_file)
        except Exception:
            pass

        
    def __enter__(self):
        #Handler for openning a new connection to DB
        try:
            if self.drop_after is True:
                self.drop_database()
                self.create_database()
                
            self.conn = self.connect_to_database() 
            self.cur = self.conn.cursor()

        except Exception as e:
            print(cs(e, color='yellow'))

        if self.load_file:
            self.run_file(self.load_file)

        self.printer = printSql(cursor=self.cur)

        return self #returns a printSql object that can execute any query and print results in tables
    
    def __exit__(self, exc_type, exc_value, traceback):
        #Handler to close connection to DB
        try:
            self.conn.commit()
            self.cur.close()
            self.conn.close()

            if self.drop_after is True:
                self.drop_database()
        except Exception:
            pass

    def close(self):
        self.__exit__()

class SimpleInterface():

    def __init__(self, db_object) -> None:
        self.db = SimpleSQL()
        self.db = db_object
        self.miner = None
        self.recording = False
        self.executing_recording = False
        self.recording_commands = []
        self.execute_buffer = []
        self.recording_file_name = ""
        self.hide_unfit = True
        self.show_r2_details = False
        self.cur_color = 'Multi1'

        self.keywords = ["preview", "show", "select", "update", "table", "create", "from",
                                 "run", "exit", "close", "default", "color", "red", "multi1",
                                 "multi2", "insert", "join", "left", "right", "cross", "where",
                                 "rollback;", "database", "view", "and", "or", "clear", "mine",
                                 "target", "keywords", "help", "len", "size", "start recording",
                                 "end recording", "play recording", "pause", "varchar", "int", 
                                 "decimal", "serial", "into", "foreign key", "primary key", 
                                 "references", "create table", "miner settings", "limits", 
                                 "hide_unfit", "detailed", "reset", "true", "false", "settings", 
                                "list recordings", "execute", "fields", "with", "distinct",
                                 "make prediction", "home", "date", "time", "align", "set", "features", 
                                 "set features", "miner features", "recording", "prediction"]
        
        
        self.rec_file_names = self.get_recording_files_list()

        # Enable command history and set the history file
        readline.parse_and_bind("tab: complete")
        readline.set_history_length(100)  # Set the desired history length
        # Set the custom completer
        readline.set_completer(self.custom_completer)
        self.histfile = "data/cmd_history"  # You can specify your history file   

        try:
            readline.read_history_file(self.histfile)  # Read the history if it exists
        except FileNotFoundError:
            pass

    def custom_completer(self, text, state):

        
        if "play recording" in readline.get_line_buffer():
            keywords = self.rec_file_names
            
        else:
            #capitalize options 
            keywords= [key.upper() for key in self.keywords]
            keywords.extend(self.keywords)

        options = [c for c in keywords if c.startswith(text)]
        
        if state < len(options):            
            return options[state]
        else:
            return None
    
    def add_key_to_history(self, key):
        self.keywords.append(key)

    def clear_screen(self):
        os.system('clear')  # Use 'clear' command to clear the screen

    def save_history(self):
        readline.write_history_file(self.histfile)
    
    def get_input(self):
        if self.recording is True:
            value = input("rec*")
            if value.lower() != "end recording":
                self.recording_commands.append(value)
            return value
        else:
            if self.executing_recording is True:
                if self.execute_buffer:
                    next_command = self.execute_buffer.pop(0).replace("\n", "")
                    print(cs("-->" + next_command, color='white'))
                    return next_command
                else:
                    self.executing_recording = False
                    print(cs(f"\nEnd of recording '{self.recording_file_name}'.", color= "LightSteelBlue2"))
            return input(">>>")
    
    def get_args(self, input_command, values):
        try:
            args = values.split(f"{input_command} ")[1].split(" ")
            args = [arg.replace(" ", "") for arg in args if arg!= ""]
            return  args
        
        except Exception:
            return []
        
    def convert_input_to_data_type(self, value):
        try:
            # First, try to convert the string to a float
            float_value = float(value)
            
            # If it's an integer (e.g., "5.0"), it will still be considered a float, so check if it's also an integer
            if float_value.is_integer():
                return int(value)
            else:
                return float(value)
            
        except ValueError:
            # If converting to a float raises a ValueError, it's not a number
            return str(value)
        
    def get_recording_files_list(self):
        try:
            folder_path = "recordings/"  # Replace with the path to your folder

            # List all files in the folder
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
            files = [f for f in files if "__.rec" in f]
            return files
        
        except Exception:
            return []
    
    
    def home(self):
            self.clear_screen()
            print(cs(f"simpleInterface --- Logged into '{self.db.db_name}' database ---", color= "LightSteelBlue2"))
            print()

        
    def interface_init(self, load_script = None):
        self.clear_screen()

        #add table names to the keyboard help
        table_names = self.db.get_table_names()
        for table in table_names:
            self.add_key_to_history(table)

            #now add fields of the table
            table_fields = self.db.get_table_fields(table_name = table)
            for field in table_fields:
                self.add_key_to_history(field)
        

        self.db.printer.load_style(options = {'color': self.cur_color})
        
        # Start the Interface and run an initial script if provided 
        if load_script is not None:
            self.execute_recording(load_script)
        
        self.home()
        command = self.get_input()
        command_start = None
        command_executed = False
        command_buffer = ""
        arg = ""

        while command.lower() not in ['exit', 'close']:


            if "select" in command.lower() or "with" in command.lower():
                command_start = 'print'
            
            elif command.lower() in ["clear", "home"]:
                self.home()
                command_executed = True
                
            elif command.lower() == "preview":
                self.db.preview()
                command_executed = True
            
            elif "execute" in command.lower():
                command_start = 'execute'
            
            elif "color default" in command.lower():
                try:
                    self.cur_color = 'Multi1'
                    options = {'color': self.cur_color}
                    self.db.printer.load_style(options)
                    self.miner.theme_color = self.cur_color
                except Exception as e:
                       print(cs(e, color='yellow'))
                command_executed = True
            
            elif "color" in command.lower():
                try:
                    arg = self.get_args("color", command)[0]
                    colors = _load_colors()
                    colors = [c['name'].lower() for c in colors.values()]
                    
                    if arg.lower() in colors or arg in ['multi1', 'multi2']:
                        options = {'color':arg}
                        self.db.printer.load_style(options)
                        self.miner.theme_color = arg
                        self.cur_color = arg
                except IndexError:
                    print(cs("Indicate a color.", color='yellow'))
                except Exception as e:
                    print(cs("Check your entry.", color='yellow'))
                command_executed = True
            
            elif "tables" == command.lower():
                try:
                    self.db.print_tables_names()
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "run" in command.lower():
                try:
                    arg = self.get_args("run", command)[0]
                    self.db.run_file(file_name= arg)
                except IndexError:
                    print(cs("Indicate a file.", color='yellow'))
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "align" in command.lower():
                try:
                    arg = self.get_args("align", command)[0][0]
                    if arg == None or arg.lower() not in ['l', 'c', "r"]:
                        raise ValueError("Indicate 'l', 'c' or 'r'.")
                    self.db.printer.load_style({'align':arg, 'color':self.cur_color})
                except IndexError:
                    print(cs("Indicate 'l', 'c' or 'r'.", color='yellow'))
                except Exception as e:
                    print(cs(e, color='yellow'))

                
                command_executed = True
            
            elif "show" in command.lower():
                try:
                    arg = self.get_args("show", command)[0]
                    if not self.db.check_table_exists(table_name = arg):
                        raise ValueError(f"Table '{arg}' does not exist.")

                    self.db.printer.print(f"SELECT * FROM {arg};")
                    self.add_key_to_history(arg)
                except IndexError:
                    print(cs("Indicate a table to show.", color='yellow'))
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
             
            elif "size" in command.lower():
                try:
                    arg = self.get_args("size", command)[0]
                    if not self.db.check_table_exists(table_name = arg):
                        raise ValueError(f"Table '{arg}' does not exist.")
                    print(cs(f"{self.db.get_table_size(table_name= arg)}", color= "LightSteelBlue2"))
                except IndexError:
                    print(cs("Indicate a table.", color='yellow'))
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True   

            elif "len" in command.lower():
                try:
                    arg = self.get_args("len", command)[0]
                    if not self.db.check_table_exists(table_name = arg):
                        raise ValueError(f"Table '{arg}' does not exist.")
                    print(cs(f"{self.db.get_table_length(table_name= arg)}", color= "LightSteelBlue2"))
                except IndexError:
                    print(cs("Indicate a table.", color='yellow'))   
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True             
            
            elif "keywords" in command.lower():
                try:
                    print(cs(f"{self.keywords}", color='yellow'))
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "pause" == command.lower():
                try:
                    if self.executing_recording is True:
                        print(cs("<<<Press Enter to continue...",  color = 'LightSteelBlue2'))
                        input()
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "start recording" in command.lower():
                try:
                    print(cs("Recording started.", color='yellow'))
                    self.recording = True
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "end recording" in command.lower():
                try:
                    if self.recording is True:
                        self.recording = False
                        print(cs("Recording finished. Save? [y/n]:", color='LightSteelBlue2'), end="", flush=True)
                        save =self.get_input()
                        if save.lower() == "y":
                            print(cs("Please indicate record name:", color='LightSteelBlue2'), end="", flush=True)
                            record_name = self.get_input()
                            self.save_recording(record_name = record_name)
                    else:
                        print(cs("There is no active recording.", color='yellow'))
                except Exception as e:
                        print(cs(e, color='yellow'))
                command_executed = True
            
            elif "play recording" in command.lower():
                try:
                    if self.recording == True:
                        print(cs("Cannot play recordings while recording.", color='LightSteelBlue2'))
                    else:
                        arg = command.replace(" ","").split("playrecording")[1]
                        self.execute_recording(record_name = arg)
                        print()
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True


            elif "mine table" in command.lower():
                try:
                    arg = self.get_args("mine table", command)[0]

                    if not self.db.check_table_exists(table_name=arg):
                        raise ValueError(f"Table '{arg}' does not exist.")

                    self.miner = self.db.new_model(model_type = 'LinearRegression', table_name=arg)
                    self.miner.theme_color = self.cur_color

                    print(cs(f"Indicate target field", color= "LightSteelBlue2"))
                    target = self.get_input()
                    fields = self.db.get_table_fields(table_name = arg)

                    if target.capitalize() not in self.db.get_table_fields(table_name = arg):
                        raise ValueError(f"Field '{target}' not present in table '{arg}'.")

                    r2_data = self.miner.analize_field_relations(target = target,
                                                                show_details= self.show_r2_details, 
                                                                hide_unfit= self.hide_unfit)
                                       
                    print(cs(f"Set up model features automatically?[y/n]:", color= "LightSteelBlue2"), end="", flush=True)
                    auto_set_up = True if self.get_input().lower() in ["y", "yes"] else False

                    if auto_set_up:
                        features = [f.replace(" ", "") for f in r2_data[0][0].split(",")]
                        print(cs(f"Setting model features to {r2_data[0][0]}.", color= "yellow"))

                        self.miner.define_model_features(features = features, target= [target.capitalize()])

                    print(cs(f"Plot results?[y/n]:", color= "LightSteelBlue2"), end="", flush=True)
                    show_plot = True if self.get_input().lower() in ["y", "yes"] else False

                    if show_plot:
                        self.miner.show_plot()
                except IndexError:
                    print(cs("Missing argument <table>.", color='yellow'))
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "miner settings" in command.lower():
                try:
                    args = self.get_args("miner settings", command)
                    args = [arg for arg in args if arg != ',']

                    if args == []:
                        if not self.miner:
                            raise AttributeError("Miner not active. Type 'miner settings help'")
                        else:
                            self.miner.show_model_performance()
                            r2_1, r2_2, r2_3 = self.miner.r2_limits
                            print(cs(f"""R² high threshole: {r2_1}
R² moderate threshold:{ r2_2} 
R² low threshold: {r2_3}
                                """, color='LightSteelBlue2'))
                            print(cs(f"Hide_unfit: {self.hide_unfit}", color='LightSteelBlue2'))
                            print(cs(f"Detailed: {self.show_r2_details}", color='LightSteelBlue2'))
                            print(cs("\ntype 'miner settings help' to learn how to adjust settings", color='LightSteelBlue2'))
                    
                    elif args[0].lower() == 'help':
                        print(cs("type 'miner settings' ...", color='LightSteelBlue2'))
                        print(cs("limits <low, mod, high>     : Define r2 threshols ", color='LightSteelBlue2'))  
                        print(cs("hide_unfit: <True, False>   : Hide unfit combinations ", color='LightSteelBlue2'))
                        print(cs("detailed:   <True, False>   : Show detailed R² classifiation ", color='LightSteelBlue2'))
                        print(cs("reset:                      : Set default values ", color='LightSteelBlue2'))
                    
                    elif args[0].lower() == 'limits':
                        if len(args) != 4:
                            print(cs("Format must be 'limits low, mod, high'.", color='yellow'))
                        elif self.miner:
                            low, mod, high = 1, 2, 3
                            self.miner.set_custom_r2_limits(high = float(args[high]), 
                                                            moderate = float(args[mod]), 
                                                            low = float(args[low]), 
                                                            )     
                    elif args[0].lower() == 'reset':
                        if self.miner:
                            self.miner.reset_limits()
                            self.show_r2_details = False
                            self.hide_unfit = True
   
                    elif args[0].lower() == 'hide_unfit':
                        if args[1].lower() == 'true':
                            self.hide_unfit = True
                        elif args[1].lower() == 'false':
                            self.hide_unfit = False
                        else:
                            print(cs("hide_unfit <True / False>", color='yellow'))
                    
                    elif args[0].lower() == 'detailed':
                        if args[1].lower() == 'true':
                            self.show_r2_details = True
                        elif args[1].lower() == 'false':
                            self.show_r2_details = False
                        else:
                            print(cs("detailed <True / False>", color='yellow'))
                
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "list recordings" in command.lower():
                try:
                    files = self.get_recording_files_list()
                    for file in files:
                        print(cs(file, color='LightSteelBlue2'))
                    print()  

                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "fields" in command.lower():
                try:
                    arg = self.get_args("fields", command)
                    self.db.print_table_fields(table_name = arg[0])

                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
            elif "make prediction" in command.lower():
                try:
                    if not self.miner:
                        print(cs("No miner has been set up yet. Check 'miner settings help'...", color='yellow'))
                    else:
                        try:
                            table = self.get_args("make prediction", command)[0]
                            if self.miner.table_name != table:
                                print(cs(f"Miner has not been set up for table {table}", color='yellow'))
                            else:
                                args = {}
                                print(cs("Specify value for field...", color='LightSteelBlue2'))  
                                for feature in self.miner.current_features:
                                    print(cs(f"* '{feature}'", color='LightSteelBlue2'), end="", flush=True)  

                                    value_input = self.get_input()
                                    args[feature] = self.convert_input_to_data_type(value_input)
                                
                                result = self.miner.predict(args)
                                print(cs(f"\nPredicted output: '{result}'\n", color='Green')) 
                            
                        except IndexError:
                            print(cs("Missing argument <table>.", color='yellow'))
                        except Exception:
                            print(cs("Input values presented an issue, try again.", color='yellow'))
                except Exception:
                    print(cs(e, color='yellow'))
                command_executed = True

            elif "miner set features" in command.lower():
                try:
                    if self.miner is None:
                        raise ValueError("Model not initiated. Type 'mine table <table> first.'")
                    fields = self.get_args("miner set features", command)
                    features = [f.replace(" ", "").replace(",", "") for f in fields if f != ""]
                    if len(features) == 0:
                        raise ValueError("Missing arguments <features>.")

                    self.miner.define_model_features(features=features)
                    
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True

            elif "miner features" in command.lower():
                try:
                    if self.miner is None:
                        raise ValueError("Model not initiated. Type 'mine table <table> first.'")
                    
                    print(cs("Current model features:", color='LightSteelBlue2'))
                    print(cs(f"{', '.join(self.miner.current_features)}", color='white'))
                    print()     
                   
                except Exception as e:
                    print(cs(e, color='yellow'))
                         
                command_executed = True

            
            elif "help" in command.lower():
                try:
                    self.help()
                except Exception as e:
                    print(cs(e, color='yellow'))
                command_executed = True
            
                 
            if not command_executed:
                command_buffer += " " + command

            if ";" in command:
                if not command_start:
                    print(cs("<<<Command not recognized, check your entry.", color='LightSteelBlue2'))

                elif command_start == 'print':
                        try:
                            self.db.print(command_buffer)
                        except Exception as e:
                            print(cs(e, color='yellow'))   
                        command_start = None
                        command_buffer = ""
                
                elif command_start == 'execute':
                        command_buffer = command_buffer.split("execute")[1]
                        self.db.execute(command_buffer)
                        command_start = None
                        command_buffer = ""
            elif not command_start  and not command_executed and not ";" in command:
                print(cs(f"<<<Command {command} not recognized, check your entry.", color='LightSteelBlue2'))

            readline.redisplay() 
            command = self.get_input()
            command_executed = False
        print("Closing interface...")
        
        self.save_history()

    def save_recording(self, record_name):
            
        if self.recording_commands is not None:   
            try:
                file_name = "./recordings/" + record_name + "__.rec"
                with open(file_name, "w") as file:
                        # Write each string to the file, one per line
                    for string in self.recording_commands:
                        file.write(string + "\n")
                
                self.recording_commands = []
            except FileExistsError:
                print(cs(f"Record '{record_name}' already exists. Try again or just hit enter to skip.", color='yellow'))
                value = self.get_input()
                self.save_recording(value)
    
    def execute_recording(self, record_name):

            try:
                file_name = "recordings/" + record_name + "__.rec"
                with open(file_name, "r") as file:
                    #Read records into biffer
                    self.execute_buffer = file.readlines()
                    self.recording_commands = []
                    self.executing_recording = True
                    self.recording_file_name = record_name
            except FileNotFoundError:
                try:
                    file_name = "recordings/" + record_name
                    with open(file_name, "r") as file:
                        #Read records into biffer
                        self.execute_buffer = file.readlines()
                        self.recording_commands = []
                        self.executing_recording = True
                        self.recording_file_name = record_name
                except FileNotFoundError:
                        print(cs(f"File '{file_name}' does not exist. Check your entry.", color='yellow'))
                        return
                
    
    def help(self):

        commands_list = [
                    "select: Execute SQL queries.",
                    "exit / close: Close simpleInterface",
                    "home / clear: Clear the terminal screen.",
                    "preview: Preview the database content.",
                    "execute: Execute SQL commands.",
                    "color default: Set default text color.",
                    "color <color>: Set text color.",
                    "tables: Print table names.",
                    "fields: Show table fields"
                    "run <filename>: Run a SQL file.",
                    "align <alignment>: Set text alignment.",
                    "show <table>: Show table data.",
                    "size <table>: Get table size.",
                    "len <table>: Get table length.",
                    "keywords: Show keywords.",
                    "pause: Pause recording playback.",
                    "start recording: Start recording commands.",
                    "end recording: End recording commands.",
                    "play recording <name>: Play recorded commands.",
                    "list recordings: Show recording files available",
                    "mine table <table>: Analyze table relations and plot.",
                    "miner settings help: Custom settings for mining component.",
                    "miner features: Display the current features the model was trained with.",
                    "miner set features <fields>: Update features (fields from table).",
                    "make prediction <table>: Based on a running miner, predict an output value."
                    ]
        
        max_command_length = max(len(command.split(':')[0]) for command in commands_list)

        
        print(cs("<<<need some help? Commands list:\n", color='LightSteelBlue2'))

        for command in commands_list:
            command_name, command_description = command.split(':', 1)
            formatted_description = command_name.ljust(max_command_length) + ':' + command_description
            print(cs(formatted_description, color='LightSteelBlue2'))
                

        
if __name__ == "__main__":

    with SimpleSQL("music") as db:
        db.printer.load_style(options={'color':'Multi2'})
        db.print("SELECT * FROM musician WHERE instrument='Guitar';", color='Multi1')



    with SimpleSQL("salaries", drop_after = True) as db:

        db.create_table_from_csv("data/Experience-Salary.csv", table_name= "salaries" , create_pri_key = True)

        db.print("SELECT * FROM salaries", color='Multi2')


    with SimpleSQL("population",load_file='data/population_db.sql' , drop_after=True ) as db:
        db.printer.load_style(options={'color':'Multi2'})

        
        db.simple_interface()

        #db.printer.show_color_list()

        




     
    

