import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def get_database_config():
    """
    Obtiene la configuración de la base de datos desde variables de entorno
    con valores por defecto para desarrollo local
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
        
        print("🔗 Intentando conectar con la configuración:")

        connection = psycopg2.connect(**config)
        cursor = connection.cursor()
        
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        
        print("✅ Conexión exitosa!")
        print(f"   Versión PostgreSQL: {db_version[0]}")
        
        cursor.close()
        connection.close()
        
        return True
        
    except psycopg2.OperationalError as e:
        print("❌ Error de conexión:")
        return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False
    
if __name__ == "__main__":
    print("="*60)
    print("🗄️  CLIENTE DE BASE DE DATOS")
    print("="*60)
    
    test_database_connection()