import sqlite3

db = sqlite3.connect('/home/sathya/resco-for-malaysia/resco_benchmark/mlflow.db')
cur = db.cursor()

# Find the run by name
cur.execute("SELECT run_uuid, status, start_time, end_time, artifact_uri FROM runs WHERE run_uuid IN (SELECT run_uuid FROM tags WHERE key='mlflow.runName' AND value LIKE '%IDQN-BB5B_net-default_act-relu_rw-queue_maxwait_seed-42_09_03_22_13%')")
runs = cur.fetchall()
print('=== RUNS FOUND ===')
for r in runs:
    print(f'  run_id: {r[0]}')
    print(f'  status: {r[1]}')
    print(f'  start_time: {r[2]}')
    print(f'  end_time: {r[3]}')
    print(f'  artifact_uri: {r[4]}')

if runs:
    run_id = runs[0][0]
    
    # Get tags
    cur.execute('SELECT key, value FROM tags WHERE run_uuid=?', (run_id,))
    tags = cur.fetchall()
    print(f'\n=== TAGS ({len(tags)}) ===')
    for t in tags:
        print(f'  {t[0]}: {t[1]}')
    
    # Get params
    cur.execute('SELECT key, value FROM params WHERE run_uuid=?', (run_id,))
    params = cur.fetchall()
    print(f'\n=== PARAMS ({len(params)}) ===')
    for p in params:
        print(f'  {p[0]}: {p[1]}')
    
    # Get all unique metric keys
    cur.execute('SELECT DISTINCT key FROM metrics WHERE run_uuid=?', (run_id,))
    metric_keys = [m[0] for m in cur.fetchall()]
    print(f'\n=== METRIC KEYS ({len(metric_keys)}) ===')
    for m in metric_keys:
        print(f'  {m}')
    
    # Count data points per metric
    print(f'\n=== METRIC COUNTS ===')
    for m in metric_keys:
        cur.execute('SELECT COUNT(*) FROM metrics WHERE run_uuid=? AND key=?', (run_id, m))
        count = cur.fetchone()[0]
        print(f'  {m}: {count} pts')

db.close()
