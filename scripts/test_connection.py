from kafka import KafkaProducer, KafkaConsumer
import json

# Test producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send a test event
producer.send('test-topic', {'message': 'hello from python'})
producer.flush()
print("Produced message")

# Test consumer
consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    consumer_timeout_ms=5000
)

for msg in consumer:
    print(f"Consumed: {msg.value}")
    break

consumer.close()
print("Connection works!")