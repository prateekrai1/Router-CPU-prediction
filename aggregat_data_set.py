import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the data from a CSV file
data = pd.read_csv('619_620/output.csv', parse_dates=['timestamp'])



print(len(data))



# Step 3: Drop duplicate timestamps
data.drop_duplicates(subset='timestamp', inplace=True)

# Step 7: Drop any rows with NaN values
data.dropna(inplace=True)

print(len(data))




# Set the timestamp column as the index
data.set_index('timestamp', inplace=True)






# minute_aggregated_data = data.resample('50S').mean().astype(int)
# minute_aggregated_data = data.resample('40S').mean().astype(int)
# minute_aggregated_data = data.resample('90S').mean().astype(int)
minute_aggregated_data = data.resample('30S').mean().astype(int)


# minute_aggregated_data = data.resample('10S').mean().astype(int)
# minute_aggregated_data = data.resample('5S').mean().astype(int)
# minute_aggregated_data = data.resample('2S').mean().astype(int)


# minute_aggregated_data = data.resample('20S').mean().astype(int)
# minute_aggregated_data = data.resample('100S').mean().astype(int)
# minute_aggregated_data = data.resample('120S').mean().astype(int)
# minute_aggregated_data = data.resample('180S').mean().astype(int)
# minute_aggregated_data = data.resample('240S').mean().astype(int)
# minute_aggregated_data = data.resample('300S').mean().astype(int)
# minute_aggregated_data = data.resample('600S').mean().astype(int)


# Reset the index to make the timestamp a column again
minute_aggregated_data.reset_index(inplace=True)

# Optionally, save the aggregated data to a new CSV file
minute_aggregated_data.to_csv('619_620/aggregated_data_minute.csv', index=False)

print(type(minute_aggregated_data['cpu_usage'][:100]))
print(minute_aggregated_data['cpu_usage'][:100])


# Convert Series to list
data_list = minute_aggregated_data['cpu_usage'][:100].tolist()
data_list_full = minute_aggregated_data['cpu_usage'].tolist()

# Print the list
print(data_list)

# Open a file in write mode (this will create the file if it doesn't exist)
with open('cpu_usage.txt', 'w') as file:
    # Iterate over the list and write each value to a new line
    for value in data_list_full:
        file.write(f"{value}\n")


# assert 0 == 1
# Plotting the CPU usage
# plt.figure(figsize=(10, 6))


start = 0
end = 100



# Calculate variance
variance = np.var(minute_aggregated_data.index[start:end])
print("Variance:", variance)



plt.plot(minute_aggregated_data.index[start:end], minute_aggregated_data['cpu_usage'][start:end], linestyle='-', color='red', linewidth=3)  # Increase linewidth as needed
# plt.plot(minute_aggregated_data.index[start:end], minute_aggregated_data['cpu_usage'][start:end], linestyle='-', color='red', linewidth=3)  # Increase linewidth as needed
# plt.plot(minute_aggregated_data.index, minute_aggregated_data['cpu_usage'], linestyle='-', color='red', linewidth=3)  # Increase linewidth as needed
plt.xlabel('Timestamp', fontsize=16)
plt.ylabel('CPU Usage (%)', fontsize=16)



# Increasing the size of the ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)


plt.grid(True)
plt.tight_layout()  # Adjusts plot parameters to give some padding

plt.savefig('cpu_usage.png', dpi=300, bbox_inches='tight')

plt.show()





# Plotting the CPU usage
# plt.figure(figsize=(10, 6))
plt.plot(minute_aggregated_data.index, minute_aggregated_data['packet_rate'],  linestyle='-', linewidth=3)
# plt.title('CPU Usage Aggregated by Minute')
plt.xlabel('Timestamp', fontsize=16)
plt.ylabel('Packet Rate', fontsize=16)


# Increasing the size of the ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Adding a legend with increased font size
# plt.legend(fontsize=16)

plt.grid(True)
# plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjusts plot parameters to give some padding
plt.savefig('packet_rate.png', dpi=300, bbox_inches='tight')
plt.show()


