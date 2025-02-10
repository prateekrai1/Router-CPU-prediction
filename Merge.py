import pandas as pd

# File paths
replay_log_path = "E:/Router CPU prediction/Data_2/replay_log.txt"
resource_usage_path = "E:/Router CPU prediction/Data_2/resource_usage.log"
output_file = "merged_log.csv"

replay_data = []
with open(replay_log_path, "r") as file:
    for line in file:
        if line.startswith("File:"):
            parts = line.split("|")
            filename = parts[0].split("File: ")[1].strip()
            start_time = pd.to_datetime(parts[1].split("Start: ")[1].strip()).tz_localize(None)
            end_time = pd.to_datetime(parts[2].split("End: ")[1].strip()).tz_localize(None)
            bandwidth = float(parts[3].split("Bandwidth: ")[1].split(" bps")[0].strip())
            tcp_count = int(parts[4].split("TCP: ")[1].strip())
            udp_count = int(parts[5].split("UDP: ")[1].strip())
            replay_data.append((start_time, end_time, bandwidth, tcp_count, udp_count))

# Load resource usage data
resource_usage = []
with open(resource_usage_path, "r") as file:
    for line in file:
        parts = line.strip().split(", ")
        timestamp = pd.to_datetime(parts[0]).tz_localize(None)
        cpu_usage = int(parts[1].split(": ")[1].replace("%", ""))
        total_mem = int(parts[2].split(": ")[1].replace(" KB", ""))
        used_mem = int(parts[3].split(": ")[1].replace(" KB", ""))
        free_mem = int(parts[4].split(": ")[1].replace(" KB", ""))
        rx_rate = int(parts[5].split(": ")[1].replace(" pps", ""))
        tx_rate = int(parts[6].split(": ")[1].replace(" pps", ""))
        resource_usage.append((timestamp, cpu_usage, total_mem, used_mem, free_mem, rx_rate, tx_rate))

# Convert to DataFrame
df_resource = pd.DataFrame(resource_usage, columns=["Timestamp", "CPU_Usage", "Total_Memory_KB", "Used_Memory_KB", "Free_Memory_KB", "RX_Rate", "TX_Rate"])

# Assign bandwidth, TCP, and UDP count based on replay periods
bandwidths = []
tcp_counts = []
udp_counts = []
for ts in df_resource["Timestamp"]:
    bw = None  # Default empty value
    tcp = None
    udp = None
    for start, end, bw_value, tcp_value, udp_value in replay_data:
        if start <= ts <= end:
            bw = bw_value
            tcp = tcp_value
            udp = udp_value
            break
    bandwidths.append(bw)
    tcp_counts.append(tcp)
    udp_counts.append(udp)

df_resource["Bandwidth_bps"] = bandwidths
df_resource["TCP_Count"] = tcp_counts
df_resource["UDP_Count"] = udp_counts

# Save merged data
df_resource.to_csv(output_file, index=False)
print(f"Merged file saved as {output_file}")
