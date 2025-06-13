# test_db_client.py
import os
import json
import unittest
from datetime import datetime, timezone
import uuid
# from dotenv import load_dotenv


# Importa las funciones de tu módulo (ajusta el nombre si no es db_client.py)
from data.database_client import (
    test_database_connection,
    get_database_config,
)

from database_functions import (
    insert_backtest_result,
    get_backtest_results,
    delete_backtest_result,
    insert_selected_strategy,
    get_selected_strategies,
)

class TestDBClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Verifica primero la conexión
        conn = test_database_connection()
        assert conn is not None, "No se pudo conectar a la DB"
        conn.close()

        # Generamos datos únicos para evitar colisiones
        cls.unique_strategy = f"test-strat-{uuid.uuid4()}"
        cls.unique_selected = f"test-selected-{uuid.uuid4()}"
        cls.unique_symbol = f"TEST{uuid.uuid4().hex[:4].upper()}"
        cls.timestamp = datetime.now(timezone.utc)

    def test_insert_and_fetch_backtest_result(self):
        # 1) Inserta un resultado de backtest con métricas
        sample_params = {"param1": 0.123, "param2": 42}
        sample_metrics = {
            "total_return": 0.15,
            "annual_return": 0.12,
            "sharpe_ratio": 1.5,
            "max_drawdown": -0.05,
            "num_trades": 25
        }
        
        result_id = insert_backtest_result(
            self.unique_strategy,
            json.dumps(sample_params),
            ticker_symbol=self.unique_symbol,
            **sample_metrics
        )
        
        self.assertIsNotNone(result_id, "No se pudo insertar el resultado de backtest")

        # 2) Recupera los últimos resultados
        rows = get_backtest_results(limit=5)
        
        # Busca nuestro registro por strategy_name
        matches = [r for r in rows if r[1] == self.unique_strategy]
        self.assertTrue(matches, "No se encontró el resultado insertado")

        # Comprueba contenido
        fetched = matches[0]
        self.assertEqual(fetched[0], result_id)  # result_id
        self.assertEqual(fetched[1], self.unique_strategy)  # strategy_name
        self.assertEqual(fetched[2], sample_params)  # params (already parsed from JSONB)
        self.assertEqual(fetched[4], self.unique_symbol)  # symbol
        self.assertEqual(fetched[5], sample_metrics["total_return"])  # total_return
        self.assertEqual(fetched[6], sample_metrics["num_trades"])  # num_trades

        # Limpieza: borra el registro insertado
        delete_backtest_result(result_id)

    def test_insert_selected_strategy(self):
        # Inserta una estrategia seleccionada
        params = {"param1": 10, "param2": "foo"}
        metrics = {
            "total_return": 0.25,
            "annual_return": 0.20,
            "sharpe_ratio": 2.0,
            "win_rate": 0.6
        }
        
        ticker_id = insert_selected_strategy(
            self.unique_selected,
            json.dumps(params),
            ticker_symbol=self.unique_symbol,
            **metrics
        )
        
        self.assertIsNotNone(ticker_id, "No se pudo insertar la estrategia seleccionada")

        # Verifica usando la nueva función get_selected_strategies
        rows = get_selected_strategies(limit=5)
        matches = [r for r in rows if r[1] == self.unique_selected]
        
        self.assertTrue(matches, "No se encontró la estrategia seleccionada insertada")
        rec = matches[0]
        self.assertEqual(rec[1], self.unique_selected)  # strategy_name
        self.assertEqual(rec[2], params)  # params (already parsed from JSONB)
        self.assertEqual(rec[4], self.unique_symbol)  # symbol
        self.assertEqual(rec[5], metrics["total_return"])  # total_return

        # Limpieza: elimina la estrategia seleccionada
        import psycopg2
        cfg = get_database_config()
        conn = psycopg2.connect(**cfg)
        cur = conn.cursor()
        cur.execute("DELETE FROM selected_strategies WHERE ticker_id = %s", (rec[0],))
        conn.commit()
        cur.close()
        conn.close()

    def test_strategy_and_symbol_creation(self):
        """Test que las estrategias y símbolos se crean automáticamente"""
        unique_strategy = f"auto-created-{uuid.uuid4()}"
        unique_symbol = f"AUTO{uuid.uuid4().hex[:4].upper()}"
        
        # Inserta con estrategia y símbolo nuevos
        result_id = insert_backtest_result(
            unique_strategy,
            json.dumps({"test": True}),
            ticker_symbol=unique_symbol,
            total_return=0.1
        )
        
        self.assertIsNotNone(result_id, "No se pudo crear estrategia/símbolo automáticamente")
        
        # Verifica que se crearon
        rows = get_backtest_results(limit=10)
        matches = [r for r in rows if r[1] == unique_strategy and r[4] == unique_symbol]
        self.assertTrue(matches, "No se encontró el resultado con estrategia/símbolo auto-creados")
        
        # Limpieza
        delete_backtest_result(result_id)

if __name__ == "__main__":
    unittest.main()
