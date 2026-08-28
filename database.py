import sqlite3
from datetime import datetime
import uuid

def get_db_connection():
    """
    Create and return a SQLite database connection.
    The database file will be created in the current directory as 'options_pricing.db'.
    """
    conn = sqlite3.connect('options_pricing.db')
    conn.row_factory = sqlite3.Row
    return conn

def initialize_database():
    """
    Initialize the database with the required tables if they don't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create inputs_table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inputs_table (
            calculation_id TEXT PRIMARY KEY,
            timestamp TEXT,
            spot_price REAL,
            strike_price REAL,
            time_to_maturity REAL,
            volatility REAL,
            risk_free_rate REAL,
            call_purchase_price REAL,
            put_purchase_price REAL
        )
    ''')
    
    # Create outputs_table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS outputs_table (
            output_id INTEGER PRIMARY KEY AUTOINCREMENT,
            calculation_id TEXT,
            shocked_spot REAL,
            shocked_vol REAL,
            call_pnl REAL,
            put_pnl REAL,
            FOREIGN KEY (calculation_id) REFERENCES inputs_table (calculation_id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_calculation(spot_price, strike_price, time_to_maturity, volatility, 
                     risk_free_rate, call_purchase_price, put_purchase_price,
                     outputs_data):
    """
    Save a calculation and its outputs to the database.
    
    Parameters:
    spot_price, strike_price, time_to_maturity, volatility, risk_free_rate : Input parameters
    call_purchase_price, put_purchase_price : Purchase prices
    outputs_data : List of tuples containing (shocked_spot, shocked_vol, call_pnl, put_pnl)
    
    Returns:
    str - The calculation_id of the saved record
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Generate unique calculation_id
    calculation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    # Insert into inputs_table
    cursor.execute('''
        INSERT INTO inputs_table 
        (calculation_id, timestamp, spot_price, strike_price, time_to_maturity, 
         volatility, risk_free_rate, call_purchase_price, put_purchase_price)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (calculation_id, timestamp, spot_price, strike_price, time_to_maturity,
          volatility, risk_free_rate, call_purchase_price, put_purchase_price))
    
    # Bulk insert into outputs_table
    for shocked_spot, shocked_vol, call_pnl, put_pnl in outputs_data:
        cursor.execute('''
            INSERT INTO outputs_table 
            (calculation_id, shocked_spot, shocked_vol, call_pnl, put_pnl)
            VALUES (?, ?, ?, ?, ?)
        ''', (calculation_id, shocked_spot, shocked_vol, call_pnl, put_pnl))
    
    conn.commit()
    conn.close()
    
    return calculation_id

def get_calculation_history(limit=10):
    """
    Retrieve recent calculation history from the database.
    
    Parameters:
    limit : int - Maximum number of records to return
    
    Returns:
    list - List of calculation records
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM inputs_table 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in results]