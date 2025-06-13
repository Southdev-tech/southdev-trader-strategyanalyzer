from data.database_client import database_connect

def get_or_create_strategy(strategy_name):
    """Get strategy_id by name, create if doesn't exist"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        
        cur.execute("SELECT strategy_id FROM strategies WHERE name = %s", (strategy_name,))
        result = cur.fetchone()
        
        if result:
            strategy_id = result[0]
        else:
            cur.execute("""
                INSERT INTO strategies (name, description, version) 
                VALUES (%s, %s, %s) RETURNING strategy_id
            """, (strategy_name, f"Auto-created strategy: {strategy_name}", "1.0"))
            strategy_id = cur.fetchone()[0]
            conn.commit()
        
        cur.close()
        conn.close()
        return strategy_id
    except Exception as e:
        print(f"❌ Error getting/creating strategy: {e}")
        return None

def get_or_create_symbol(symbol_name):
    """Get symbol_id by name, create if doesn't exist"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        
        cur.execute("SELECT symbol_id FROM symbols WHERE ticker = %s AND active = TRUE", (symbol_name,))
        result = cur.fetchone()
        
        if result:
            symbol_id = result[0]
        else:
            cur.execute("""
                INSERT INTO symbols (ticker, name, active) 
                VALUES (%s, %s, %s) RETURNING symbol_id
            """, (symbol_name, f"Auto-created symbol: {symbol_name}", True))
            symbol_id = cur.fetchone()[0]
            conn.commit()
        
        cur.close()
        conn.close()
        return symbol_id
    except Exception as e:
        print(f"❌ Error getting/creating symbol: {e}")
        return None

def create_backtest_run(notes=None):
    """Create a new backtest run and return its ID"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO backtest_runs (notes) 
            VALUES (%s) RETURNING run_id
        """, (notes,))
        run_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return run_id
    except Exception as e:
        print(f"❌ Error creating backtest run: {e}")
        return None

def insert_backtest_result(strategy_name, params_json, ticker_symbol="TEST", run_id=None, **metrics):
    """Insert backtest result with proper foreign keys"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        
        strategy_id = get_or_create_strategy(strategy_name)
        symbol_id = get_or_create_symbol(ticker_symbol)
        
        if not strategy_id or not symbol_id:
            raise Exception("Could not get strategy_id or symbol_id")
        
        if run_id is None:
            run_id = create_backtest_run("Test run")
        
        cur.execute("""
            INSERT INTO backtest_results 
            (run_id, ticker_id, strategy_id, params, total_return, annual_return, 
             sharpe_ratio, sortino_ratio, max_drawdown, win_rate, profit_factor, num_trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING result_id
        """, (
            run_id, symbol_id, strategy_id, params_json,
            metrics.get('total_return'), metrics.get('annual_return'),
            metrics.get('sharpe_ratio'), metrics.get('sortino_ratio'),
            metrics.get('max_drawdown'), metrics.get('win_rate'),
            metrics.get('profit_factor'), metrics.get('num_trades')
        ))
        
        result_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Backtest result inserted")
        return result_id
    except Exception as e:
        print(f"❌ Error inserting backtest result: {e}")
        return None

def insert_selected_strategy(strategy_name, params_json, ticker_symbol="TEST", run_id=None, **metrics):
    """Insert selected strategy with proper foreign keys"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        
        strategy_id = get_or_create_strategy(strategy_name)
        symbol_id = get_or_create_symbol(ticker_symbol)
        
        if not strategy_id or not symbol_id:
            raise Exception("Could not get strategy_id or symbol_id")
        
        if run_id is None:
            run_id = create_backtest_run("Test selected strategy")
        
        cur.execute("""
            INSERT INTO selected_strategies 
            (ticker_id, run_id, strategy_id, params, total_return, annual_return,
             sharpe_ratio, sortino_ratio, max_drawdown, win_rate, profit_factor, num_trades)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker_id) DO UPDATE SET
                run_id = EXCLUDED.run_id,
                strategy_id = EXCLUDED.strategy_id,
                params = EXCLUDED.params,
                total_return = EXCLUDED.total_return,
                annual_return = EXCLUDED.annual_return,
                sharpe_ratio = EXCLUDED.sharpe_ratio,
                sortino_ratio = EXCLUDED.sortino_ratio,
                max_drawdown = EXCLUDED.max_drawdown,
                win_rate = EXCLUDED.win_rate,
                profit_factor = EXCLUDED.profit_factor,
                num_trades = EXCLUDED.num_trades,
                selected_at = CURRENT_TIMESTAMP
        """, (
            symbol_id, run_id, strategy_id, params_json,
            metrics.get('total_return'), metrics.get('annual_return'),
            metrics.get('sharpe_ratio'), metrics.get('sortino_ratio'),
            metrics.get('max_drawdown'), metrics.get('win_rate'),
            metrics.get('profit_factor'), metrics.get('num_trades')
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Selected strategy inserted")
        return symbol_id
    except Exception as e:
        print(f"❌ Error inserting selected strategy: {e}")
        return None

def get_backtest_results(limit=10):
    """Get backtest results with strategy and symbol names"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT br.result_id, s.name as strategy_name, br.params, br.created_at,
                   sym.ticker, br.total_return, br.num_trades
            FROM backtest_results br
            JOIN strategies s ON br.strategy_id = s.strategy_id
            JOIN symbols sym ON br.ticker_id = sym.symbol_id
            ORDER BY br.created_at DESC 
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Error fetching backtest results: {e}")
        return []

def get_selected_strategies(limit=10):
    """Get selected strategies with strategy and symbol names"""
    try:
        conn = database_connect()
        cur = conn.cursor()
        cur.execute("""
            SELECT ss.ticker_id, s.name as strategy_name, ss.params, ss.selected_at,
                   sym.ticker, ss.total_return
            FROM selected_strategies ss
            JOIN strategies s ON ss.strategy_id = s.strategy_id
            JOIN symbols sym ON ss.ticker_id = sym.symbol_id
            ORDER BY ss.selected_at DESC 
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Error fetching selected strategies: {e}")
        return []

def delete_backtest_result(result_id):
    try:
        conn = database_connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM backtest_results WHERE result_id = %s", (result_id,))
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Backtest result deleted")
    except Exception as e:
        print(f"❌ Error deleting backtest result: {e}")