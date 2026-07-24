import json
import time
import psycopg2
import psycopg2.extras
from kafka import KafkaProducer
from datetime import datetime
from src.config import DB_CONFIG, KAFKA_BOOTSTRAP

class EventSimulator:
    def __init__(self, 
                 db_config,
                 kafka_bootstrap='localhost:9092',
                 topic='raw-events',
                 speed_factor=100,
                 burst_mode=False,
                 burst_delay_ms=10):
        
        self.topic = topic
        self.speed_factor = speed_factor
        self.burst_mode = burst_mode
        self.burst_delay = burst_delay_ms / 1000  # convert to seconds
        
        # Kafka producer — partition by client_id
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap,
            key_serializer=lambda k: k.encode('utf-8'),
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # PostgreSQL connection
        self.conn = psycopg2.connect(**db_config)
    
    def load_events(self, limit=None, offset=None, start_date=None):
        cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        query = "SELECT * FROM transactions"
        if start_date:
            query += f" WHERE timestamp >= '{start_date}'"
        query += " ORDER BY timestamp ASC"
        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"
        cursor.execute(query)
        events = cursor.fetchall()
        cursor.close()
        return events
    
    def run(self, limit=None, offset=None, start_date=None):
        events = self.load_events(limit, offset, start_date)
        print(f"Loaded {len(events)} events from PostgreSQL")
        
        prev_ts = None
        sent = 0
        
        for event in events:
            # Calculate delay from previous event
            current_ts = event['timestamp']
            if isinstance(current_ts, str):
                current_ts = datetime.fromisoformat(current_ts)
            
            if prev_ts and not self.burst_mode:
                gap = (current_ts - prev_ts).total_seconds()
                sleep_time = max(0, gap / self.speed_factor)
                # Cap sleep to avoid huge gaps (e.g. weekends)
                sleep_time = min(sleep_time, 2.0)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            elif self.burst_mode:
                time.sleep(self.burst_delay)
            
            prev_ts = current_ts
            
            # Build the message
            message = {
                'event_id': event['event_id'],
                'client_id': event['client_id'],
                'account_id': event['account_id'],
                'employee_id': event['employee_id'],
                'branch_id': event['branch_id'],
                'timestamp': current_ts.isoformat(),
                'amount': float(event['amount']),
                'currency': event['currency'],
                'channel': event['channel'],
                'operation_type': event['operation_type'],
                'payload': event['payload'] if isinstance(event['payload'], str) 
                          else json.dumps(event['payload'])
            }
            
            # Produce with client_id as partition key
            self.producer.send(
                self.topic,
                key=event['client_id'],
                value=message
            )
            
            sent += 1
            if sent % 1000 == 0:
                self.producer.flush()
                print(f"Sent {sent}/{len(events)} events")
        
        self.producer.flush()
        print(f"Simulation complete: {sent} events published to '{self.topic}'")
    
    def close(self):
        self.producer.close()
        self.conn.close()


if __name__ == '__main__':
    sim = EventSimulator(db_config=DB_CONFIG, kafka_bootstrap=KAFKA_BOOTSTRAP, burst_mode=True, burst_delay_ms=10)
    try:
        
        sim.run(limit=2000, start_date='2025-06-20')
    finally:
        sim.close()


        