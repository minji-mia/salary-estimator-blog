# impor DB packge
import sqlite3

def create_connection():
    """ 
    create a database connection to the SQLite database specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    db_file = "data/data.db"
    conn = None
    
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Exception as e:
        pass

def create_prediction_table():
    """ create a prediction table """
    try:
        conn = create_connection()
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS predtable
        (age NUMERIC,workclass TEXT,fnlwgt NUMERIC,education TEXT,education_num NUMERIC,marital_status TEXT,occupation TEXT,relationship TEXT,race TEXT,sex TEXT,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country TEXT,predicted_class NUMERIC,model_class TEXT, time_of_prediction TEXT)""")

    except Exception as e:
        pass

def add_pred_data(age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,predicted_class,model_class, time_of_prediction):
    conn = create_connection()
    c = conn.cursor() 
    c.execute("""INSERT INTO predtable
    (age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,predicted_class,model_class, time_of_prediction) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (age ,workclass ,fnlwgt ,education ,education_num ,marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country,predicted_class,model_class, time_of_prediction))
    conn.commit()

def view_all_pred_data():
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM predtable')
    data = c.fetchall()
    return data

