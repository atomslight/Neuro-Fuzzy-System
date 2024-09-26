import openpyxl

# Load your Excel file
file_path = 'Precipitation.xlsx'  # Replace with your file path
wb = openpyxl.load_workbook(file_path)

# Select the active worksheet (you can also specify a sheet by name)
sheet = wb.active

# Define the Precipitation column (adjust the column as necessary)
Precipitation_column = 'A'  # Replace 'A' with the actual column of Precipitation
calc_column = 'B'  # Column where formula will be stored

# Find the last row with data in the Precipitation column
last_row = sheet.max_row

# Loop through all rows in the Precipitation column, starting from the first row
for row in range(1, last_row + 1):
    # Get the cell reference for the current row
    current_cell = f'{Precipitation_column}{row}'
    
    # Create the formula for each row using the current row as the base, and saving the formula in the adjacent column
    formula = f"=({current_cell}+MIN({Precipitation_column}1,{Precipitation_column}{last_row}))/({Precipitation_column}1+MAX({Precipitation_column}1,{Precipitation_column}{last_row}))"
    
    # Apply the formula in the calculation column (next column)
    sheet[f'{calc_column}{row}'].value = formula

# Save the workbook
wb.save('updated_excel_file.xlsx')

print("Formulas applied successfully to all rows.")
