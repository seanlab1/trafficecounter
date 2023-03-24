import csv
import random

# 시간대별 교통량 데이터 생성
hours = [i for i in range(24)]
days=[i for i in range(30)]
counts = [random.randint(0, 100) for _ in range(24)]

# CSV 파일에 저장
with open('traffic_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['hour', 'count','day'])
    for hour, count,day in zip(hours, counts,days):
        writer.writerow([hour, count,day])