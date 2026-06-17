import psycopg2
import json

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="amen_anomaly",
    user="postgres", password="pwd"
)
cur = conn.cursor()

# Row count
cur.execute("SELECT COUNT(*) FROM profile_snapshots")
print(f"Total profiles: {cur.fetchone()[0]}")

# Sample one profile to inspect structure
cur.execute("SELECT client_id, profile_data FROM profile_snapshots LIMIT 1")
row = cur.fetchone()
profile = row[1] if isinstance(row[1], dict) else json.loads(row[1])
print(f"\nClient: {row[0]}")
print(f"Fields in profile: {len(profile)}")
print(f"\nField names:\n{sorted(profile.keys())}")
print(f"\nSample values:")
for key in sorted(profile.keys())[:10]:
    print(f"  {key}: {profile[key]}")

cur.close()
conn.close()