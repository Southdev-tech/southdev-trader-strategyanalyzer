import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_database_config():
    """
    Obtain the database configuration from environment variables
    with default values for local development
    """
    config = {
        "database": os.getenv("DATABASE_NAME", "postgres"),
        "user": os.getenv("DATABASE_USER", "postgres"), 
        "password": os.getenv("DATABASE_PASS", "password"),
        "host": os.getenv("DATABASE_HOST", "localhost"),
        "port": os.getenv("DATABASE_PORT", "5432")
    }
    return config

def test_database_connection():
    """
    Prueba la conexión a la base de datos
    """
    try:
        config = get_database_config()
        
        print("🔗 Attempting to connect with configuration:")
        print(f"   Host: {config['host']}")
        print(f"   Port: {config['port']}")
        print(f"   Database: {config['database']}")
        print(f"   User: {config['user']}")
        
        connection = psycopg2.connect(**config)
        print("✅ Database connection successful")
        return connection
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return None
    
if __name__ == "__main__":
    print("="*60)
    print("🗄️  CLIENTE DE BASE DE DATOS")
    print("="*60)
    
    test_database_connection()