from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'raw-events',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    consumer_timeout_ms=5000
)

count = 0
for msg in consumer:
    if count < 3:
        print(f"Key: {msg.key.decode()}")
        print(f"Event: {msg.value['operation_type']} | {msg.value['amount']} TND")
        print(f"Client: {msg.value['client_id'][:12]}...")
        print("---")
    count += 1

print(f"\nTotal events on topic: {count}")
consumer.close()