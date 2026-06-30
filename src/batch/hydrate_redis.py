import json
import psycopg2
import psycopg2.extras
import redis
from src.config import DB_CONFIG, KAFKA_BOOTSTRAP

class ProfileHydrator:
    def __init__(self, db_config, redis_host='localhost', redis_port=6379):
        self.conn = psycopg2.connect(**db_config)
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)
    
    def hydrate_client_profiles(self):
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Get the latest snapshot for each client
        cursor.execute("""
            SELECT DISTINCT ON (client_id) 
                client_id, profile_data
            FROM profile_snapshots
            ORDER BY client_id, computed_at DESC
        """)
        
        rows = cursor.fetchall()
        pipe = self.redis.pipeline()
        
        for row in rows:
            profile = row['profile_data']
            if isinstance(profile, str):
                profile = json.loads(profile)
            pipe.set(
                f"profile:client:{row['client_id']}", 
                json.dumps(profile)
            )
        
        pipe.execute()
        cursor.close()
        print(f"Hydrated {len(rows)} client profiles into Redis")
        return len(rows)
    
    def hydrate_employee_profiles(self):
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT ON (employee_id)
                employee_id, profile_data
            FROM employee_profile_snapshots
            ORDER BY employee_id, computed_at DESC
        """)
        
        rows = cursor.fetchall()
        pipe = self.redis.pipeline()
        
        for row in rows:
            profile = row['profile_data']
            if isinstance(profile, str):
                profile = json.loads(profile)
            pipe.set(
                f"profile:employee:{row['employee_id']}",
                json.dumps(profile)
            )
        
        pipe.execute()
        cursor.close()
        print(f"Hydrated {len(rows)} employee profiles into Redis")
        return len(rows)
    
    def hydrate_archetype_baselines(self):
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("SELECT archetype, baseline_data FROM archetype_baselines")
        
        rows = cursor.fetchall()
        pipe = self.redis.pipeline()
        
        for row in rows:
            baseline = row['baseline_data']
            if isinstance(baseline, str):
                baseline = json.loads(baseline)
            pipe.set(
                f"baseline:{row['archetype']}",
                json.dumps(baseline)
            )
        
        pipe.execute()
        cursor.close()
        print(f"Hydrated {len(rows)} archetype baselines into Redis")
        return len(rows)
    

    def hydrate_client_buffers(self):
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cursor.execute("""
            SELECT DISTINCT ON (client_id)
                client_id, profile_data
            FROM profile_snapshots
            ORDER BY client_id, computed_at DESC
        """)
        
        rows = cursor.fetchall()
        pipe = self.redis.pipeline()
        count = 0
        
        for row in rows:
            profile = row['profile_data']
            if isinstance(profile, str):
                profile = json.loads(profile)
            
            buffer = profile.get('recent_events_buffer', [])
            if not buffer:
                continue
            
            # Map to enrichment service buffer format
            cbuf = []
            for evt in buffer:
                cbuf.append({
                    'ts': evt['timestamp'],
                    'amount': evt['amount'],
                    'op': evt['operation_type'],
                    'eid': evt.get('employee_id', ''),
                    'payload': '{}'
                })
            
            pipe.set(
                f"buffer:client:{row['client_id']}",
                json.dumps(cbuf)
            )
            count += 1
        
        pipe.execute()
        cursor.close()
        print(f"Hydrated {count} client buffers into Redis")
        return count
    def run(self):
        print("Starting profile hydration...")
        self.hydrate_client_profiles()
        self.hydrate_employee_profiles()
        self.hydrate_archetype_baselines()
        self.hydrate_client_buffers()
        print("Hydration complete")
    
    def close(self):
        self.conn.close()
        self.redis.close()


if __name__ == '__main__':
    
    hydrator = ProfileHydrator(db_config=DB_CONFIG)
    try:
        hydrator.run()
    finally:
        hydrator.close()