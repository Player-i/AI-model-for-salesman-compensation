import csv

# Input CSV file (assuming it contains the full table data)
input_file = 'data/raw_data.csv'
output_file = 'data/raw_data_ruta_01.csv'

# Read the input CSV and filter for Ruta 01
ruta_01_data = []
with open(input_file, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Read the header row
    ruta_01_data.append(header)  # Add header to the output
    for row in reader:
        if row[1] == 'RUTA 01':  # Check if the second column (RUTA VENTA) is "RUTA 01"
            ruta_01_data.append(row)

# Write the filtered data to a new CSV file
with open(output_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for row in ruta_01_data:
        writer.writerow(row)

print(f"CSV file '{output_file}' has been created with Ruta 01 data.")