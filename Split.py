import pandas as pd
from sklearn.model_selection import train_test_split

with open("trail2.log", "r") as file:
    logs = file.readlines()

data =[]
for log in logs:
    timestamp = log.split(",")[0]
    cpu_usage = int(log.split("CPU USAGE:")[1].split("%")[0])
    data.append([timestamp, cpu_usage])

df = pd.DataFrame(data, columns=["Timestamp", "CPU Usage"])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(by ="timestamp")

df = df.sort_values(by = "timestamp")
df["cpu_usage"] = df["cpu_usage"]/100.0
print(df.head())

train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False)
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)