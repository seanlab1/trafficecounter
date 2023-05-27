import sqlite3
conn = sqlite3.connect("traffic.db")

cur = conn.cursor()

cur.execute("SELECT * FROM traffic_data")

rows = cur.fetchall()

for row in rows:
    print(row)

conn.close()

