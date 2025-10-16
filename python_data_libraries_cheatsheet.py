# =================================================================================================
# COMPREHENSIVE PYTHON DATA SCIENCE CHEAT SHEET
# =================================================================================================
# This file contains detailed, runnable examples for the core Python data science libraries.
# It is designed for beginners and can be executed directly in environments like Google Colab.
#
# Sections:
# 1. NumPy: For numerical operations and working with n-dimensional arrays.
# 2. Pandas: For data manipulation and analysis using DataFrames.
# 3. Matplotlib: For creating static, animated, and interactive visualizations.
# 4. Seaborn: A high-level interface for drawing attractive and informative statistical graphics.
# =================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =================================================================================================
# 1. NUMPY (Numerical Python)
# =================================================================================================
# NumPy is the fundamental package for scientific computing in Python. It provides a powerful
# N-dimensional array object, sophisticated (broadcasting) functions, tools for integrating
# C/C++ and Fortran code, and useful linear algebra, Fourier transform, and random number capabilities.

print("="*20 + " 1. NUMPY CHEAT SHEET " + "="*20 + "\n")

# ---------------------------------
# Section 1.1: Array Creation
# ---------------------------------
print("# --- 1.1: Array Creation --- #\n")

# np.array(): Create an array from a Python list or tuple.
# This is the most common way to create an array.
print("# np.array(): Create an array from a list")
list_data = [1, 2, 3, 4, 5]
arr_from_list = np.array(list_data)
print(f"Array from list {list_data}:\n{arr_from_list}\n")

# np.zeros(): Create an array filled with zeros. Useful for initialization.
# The argument is a shape tuple (rows, columns).
print("# np.zeros(): Create a 2x3 array of zeros")
zeros_arr = np.zeros((2, 3))
print(f"A 2x3 array of zeros:\n{zeros_arr}\n")

# np.ones(): Create an array filled with ones.
print("# np.ones(): Create a 2x3 array of ones with integer type")
ones_arr = np.ones((2, 3), dtype=np.int16)
print(f"A 2x3 array of ones:\n{ones_arr}\n")

# np.arange(): Create an array with a range of numbers, similar to Python's range().
# Arguments: arange(start, stop, step). 'stop' is exclusive.
print("# np.arange(): Create an array from 10 to 19 with a step of 2")
arange_arr = np.arange(10, 20, 2)
print(f"Array from 10 to 19 (step 2): {arange_arr}\n")

# np.linspace(): Create an array with evenly spaced numbers over a specified interval.
# Arguments: linspace(start, stop, num_of_points). 'stop' is inclusive.
print("# np.linspace(): Create 5 points from 0 to 10")
linspace_arr = np.linspace(0, 10, num=5)
print(f"5 points from 0 to 10: {linspace_arr}\n")

# np.full(): Create a constant array of a given shape and fill value.
print("# np.full(): Create a 2x2 array filled with the number 7")
full_arr = np.full((2, 2), 7)
print(f"A 2x2 array filled with 7:\n{full_arr}\n")

# np.eye(): Create a 2-D identity matrix (1s on the diagonal, 0s elsewhere).
print("# np.eye(): Create a 3x3 identity matrix")
eye_arr = np.eye(3)
print(f"A 3x3 identity matrix:\n{eye_arr}\n")

# np.random.rand(): Create an array of a given shape with random values from [0, 1).
print("# np.random.rand(): Create a 2x2 array with random floats")
rand_arr = np.random.rand(2, 2)
print(f"A 2x2 random array:\n{rand_arr}\n")


# ---------------------------------
# Section 1.2: Array Attributes and Inspection
# ---------------------------------
print("\n# --- 1.2: Array Attributes and Inspection --- #\n")
arr_inspect = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Our inspection array:\n{arr_inspect}\n")

# .shape: Returns a tuple with the dimensions of the array (rows, columns).
print(f"arr.shape: {arr_inspect.shape}")
# .ndim: Returns the number of array dimensions.
print(f"arr.ndim: {arr_inspect.ndim}")
# .size: Returns the total number of elements in the array.
print(f"arr.size: {arr_inspect.size}")
# .dtype: Returns the data type of the array's elements.
print(f"arr.dtype: {arr_inspect.dtype}")


# ---------------------------------
# Section 1.3: Indexing and Slicing
# ---------------------------------
print("\n# --- 1.3: Indexing and Slicing --- #\n")
arr_slice = np.arange(10, 20)
print(f"Our slicing array: {arr_slice}\n")

# Basic indexing: Access a single element.
print(f"Element at index 3: {arr_slice[3]}") # Result: 13

# Slicing: Access a range of elements. array[start:stop:step].
print(f"Elements from index 2 to 5: {arr_slice[2:5]}") # Result: [12 13 14]

# Multidimensional indexing:
arr_2d = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]])
print(f"Our 2D array:\n{arr_2d}\n")
print(f"Element at row 1, column 2: {arr_2d[1, 2]}") # Result: 15

# Multidimensional slicing:
# Get a sub-array of the first two rows and columns 1 and 2.
sub_arr = arr_2d[:2, 1:]
print(f"Sub-array (first 2 rows, cols 1-2):\n{sub_arr}\n")

# Boolean indexing: Use boolean arrays to filter elements.
bool_idx = arr_2d > 14
print(f"Boolean mask (elements > 14):\n{bool_idx}\n")
print(f"Elements where value > 14: {arr_2d[bool_idx]}\n")


# ---------------------------------
# Section 1.4: Array Math and Operations
# ---------------------------------
print("\n# --- 1.4: Array Math and Operations --- #\n")
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

# Element-wise operations:
print(f"Element-wise addition (x + y):\n{x + y}\n")
print(f"Element-wise subtraction (y - x):\n{y - x}\n")
print(f"Element-wise multiplication (x * y):\n{x * y}\n")
print(f"Element-wise division (y / x):\n{y / x}\n")
print(f"Element-wise square root (np.sqrt(x)):\n{np.sqrt(x)}\n")

# Matrix multiplication (Dot product):
# np.dot() or the @ operator can be used.
print(f"Matrix multiplication (np.dot(x, y)):\n{np.dot(x, y)}\n")
print(f"Matrix multiplication (x @ y):\n{x @ y}\n")

# Aggregation functions:
stats_arr = np.array([1, 2, 3, 4, 5])
print(f"Sum of elements: {stats_arr.sum()}")
print(f"Mean of elements: {stats_arr.mean()}")
print(f"Max value: {stats_arr.max()}")
print(f"Min value: {stats_arr.min()}\n")


# ---------------------------------
# Section 1.5: Reshaping and Transposing
# ---------------------------------
print("\n# --- 1.5: Reshaping and Transposing --- #\n")
arr_reshape = np.arange(1, 13) # 12 elements
print(f"Original array: {arr_reshape}\n")

# .reshape(): Reshapes the array without changing its data.
reshaped = arr_reshape.reshape(3, 4)
print(f"Array reshaped to 3x4:\n{reshaped}\n")

# .T or .transpose(): Transposes the array (swaps rows and columns).
transposed = reshaped.T
print(f"Transposed 3x4 array (now 4x3):\n{transposed}\n")

# .flatten(): Returns a 1D copy of the array.
flattened = transposed.flatten()
print(f"Flattened array: {flattened}\n")


# This concludes the detailed NumPy section.
# The following sections will cover Pandas, Matplotlib, and Seaborn.


# =================================================================================================
# 2. PANDAS
# =================================================================================================
# Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and
# manipulation tool, built on top of the Python programming language. It is the go-to library
# for handling structured data.

print("\n" + "="*20 + " 2. PANDAS CHEAT SHEET " + "="*20 + "\n")

# ---------------------------------
# Section 2.1: Pandas Data Structures (Series and DataFrame)
# ---------------------------------
print("# --- 2.1: Pandas Data Structures --- #\n")

# pd.Series: A one-dimensional labeled array capable of holding any data type.
# The labels are collectively referred to as the index.
print("# pd.Series: Create a Series from a list")
s = pd.Series([10, 20, 30, 40, 50], name='MyNumbers')
print(f"A Pandas Series:\n{s}\n")

# pd.DataFrame: A two-dimensional, size-mutable, and potentially heterogeneous
# tabular data structure with labeled axes (rows and columns).
print("# pd.DataFrame: Create a DataFrame from a dictionary")
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']}
df = pd.DataFrame(data)
print(f"A Pandas DataFrame:\n{df}\n")


# ---------------------------------
# Section 2.2: Reading and Writing Data
# ---------------------------------
print("\n# --- 2.2: Reading and Writing Data --- #\n")

# Pandas supports reading from and writing to a wide variety of file formats.
# Let's create a dummy CSV file to demonstrate.
df.to_csv('sample_data.csv', index=False) # index=False prevents writing row numbers

# pd.read_csv(): Read a comma-separated values (csv) file into a DataFrame.
print("# pd.read_csv(): Reading data from 'sample_data.csv'")
df_from_csv = pd.read_csv('sample_data.csv')
print(f"DataFrame read from CSV:\n{df_from_csv}\n")

# Clean up the dummy file
os.remove('sample_data.csv')


# ---------------------------------
# Section 2.3: Viewing and Inspecting Data
# ---------------------------------
print("\n# --- 2.3: Viewing and Inspecting Data --- #\n")
print(f"Our DataFrame:\n{df}\n")

# .head(): View the first n rows (default is 5).
print(f"First 2 rows (.head(2)):\n{df.head(2)}\n")

# .tail(): View the last n rows (default is 5).
print(f"Last 2 rows (.tail(2)):\n{df.tail(2)}\n")

# .info(): Get a concise summary of the DataFrame, including data types and non-null values.
print("# .info(): Summary of the DataFrame")
df.info()
print("") # for spacing

# .describe(): Generate descriptive statistics (count, mean, std, etc.).
print(f"# .describe(): Descriptive statistics:\n{df.describe()}\n")


# ---------------------------------
# Section 2.4: Selection and Indexing (loc, iloc)
# ---------------------------------
print("\n# --- 2.4: Selection and Indexing --- #\n")

# Selecting a single column returns a Series.
print(f"Selecting the 'Name' column:\n{df['Name']}\n")

# Selecting multiple columns returns a DataFrame.
print(f"Selecting 'Name' and 'City' columns:\n{df[['Name', 'City']]}\n")

# .loc: Selection by label (row and column names).
print(f"# .loc[1]: Select row with index label 1\n{df.loc[1]}\n")
print(f"# .loc[0:2, 'Age':'City']: Slice rows and columns by label\n{df.loc[0:2, 'Age':'City']}\n")

# .iloc: Selection by integer position.
print(f"# .iloc[2]: Select row at integer position 2\n{df.iloc[2]}\n")
print(f"# .iloc[0:2, 0:2]: Slice rows and columns by position\n{df.iloc[0:2, 0:2]}\n")


# ---------------------------------
# Section 2.5: Conditional Filtering
# ---------------------------------
print("\n# --- 2.5: Conditional Filtering --- #\n")

# Filter rows based on a condition.
print(f"Rows where Age is greater than 30:\n{df[df['Age'] > 30]}\n")

# Filter with multiple conditions (& for AND, | for OR).
print(f"Rows where Age > 25 AND City is 'Chicago':\n{df[(df['Age'] > 25) & (df['City'] == 'Chicago')]}\n")


# ---------------------------------
# Section 2.6: Handling Missing Data
# ---------------------------------
print("\n# --- 2.6: Handling Missing Data --- #\n")
data_missing = {'A': [1, 2, np.nan, 4], 'B': [5, np.nan, np.nan, 8], 'C': [10, 20, 30, 40]}
df_missing = pd.DataFrame(data_missing)
print(f"DataFrame with missing values:\n{df_missing}\n")

# .dropna(): Remove missing values.
# `how='any'` drops a row/column if any of its values are NaN.
print(f"Dropping rows with any missing values:\n{df_missing.dropna()}\n")

# .fillna(): Fill missing values with a specified value.
print(f"Filling missing values with the mean of column 'B':\n{df_missing.fillna({'B': df_missing['B'].mean()})}\n")


# ---------------------------------
# Section 2.7: Grouping and Aggregation (Group By)
# ---------------------------------
print("\n# --- 2.7: Grouping and Aggregation --- #\n")
data_group = {'Team': ['A', 'B', 'A', 'B', 'A', 'B'],
              'Player': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'],
              'Points': [10, 12, 15, 18, 13, 9]}
df_group = pd.DataFrame(data_group)
print(f"DataFrame for grouping:\n{df_group}\n")

# The "group by" process involves splitting the data into groups based on some criteria,
# applying a function to each group independently, and combining the results into a data structure.
# Here, we group by 'Team' and calculate the sum of 'Points' for each team.
print(f"Total points per team (grouped by 'Team' and summed):\n{df_group.groupby('Team')['Points'].sum()}\n")

# We can apply multiple aggregation functions at once.
print(f"Mean and standard deviation of points per team:\n{df_group.groupby('Team')['Points'].agg(['mean', 'std'])}\n")


# ---------------------------------
# Section 2.8: Merging and Joining DataFrames
# ---------------------------------
print("\n# --- 2.8: Merging and Joining DataFrames --- #\n")
left_df = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'A': ['A0', 'A1', 'A2']})
right_df = pd.DataFrame({'key': ['K0', 'K1', 'K3'], 'B': ['B0', 'B1', 'B3']})

print(f"Left DataFrame:\n{left_df}\n")
print(f"Right DataFrame:\n{right_df}\n")

# pd.merge(): Connects rows in DataFrames based on one or more keys.
# `how='inner'` (default) keeps only rows where the key exists in BOTH DataFrames.
inner_merge = pd.merge(left_df, right_df, on='key', how='inner')
print(f"Inner merge on 'key':\n{inner_merge}\n")

# `how='outer'` keeps all rows from both DataFrames.
outer_merge = pd.merge(left_df, right_df, on='key', how='outer')
print(f"Outer merge on 'key':\n{outer_merge}\n")


# This concludes the detailed Pandas section.


# =================================================================================================
# 3. MATPLOTLIB
# =================================================================================================
# Matplotlib is a comprehensive library for creating static, animated, and interactive
# visualizations in Python. It's the foundation for many other plotting libraries.

print("\n" + "="*20 + " 3. MATPLOTLIB CHEAT SHEET " + "="*20 + "\n")

# ---------------------------------
# Section 3.1: Basic Plotting
# ---------------------------------
print("# --- 3.1: Basic Plotting --- #\n")

# Data for plotting
x_data = np.linspace(0, 10, 100)
y_data = np.sin(x_data)

# plt.figure(): Creates a new figure for the plot.
# plt.plot(): Plots y versus x as lines and/or markers.
# plt.title(), plt.xlabel(), plt.ylabel(): Add labels to the plot.
# plt.legend(): Adds a legend.
# plt.grid(): Adds a grid.
# plt.savefig(): Saves the figure to a file.
# plt.show(): Displays the figure. (Use in scripts; often not needed in notebooks).
# plt.close(): Closes the figure window to free up memory.

print("# Example: A simple line plot")
plt.figure(figsize=(8, 5)) # Set figure size
plt.plot(x_data, y_data, color='blue', linestyle='--', marker='o', markersize=4, label='sin(x)')
plt.title("Simple Matplotlib Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.grid(True)
plt.savefig("matplotlib_line_plot.png")
print("  -> Saved 'matplotlib_line_plot.png'")
plt.close() # Close the figure


# ---------------------------------
# Section 3.2: Different Plot Types
# ---------------------------------
print("\n# --- 3.2: Different Plot Types --- #\n")

# --- Scatter Plot ---
# Useful for visualizing the relationship between two numerical variables.
print("# Example: A scatter plot")
x_scatter = np.random.rand(50) * 10
y_scatter = np.random.rand(50) * 10
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

plt.figure(figsize=(8, 5))
plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(label='Color Intensity') # Add a color bar
plt.title("Matplotlib Scatter Plot")
plt.savefig("matplotlib_scatter_plot.png")
print("  -> Saved 'matplotlib_scatter_plot.png'")
plt.close()


# --- Bar Chart ---
# Ideal for comparing categorical data.
print("# Example: A bar chart")
categories = ['Category A', 'Category B', 'Category C']
values = [7, 11, 5]

plt.figure(figsize=(8, 5))
plt.bar(categories, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title("Matplotlib Bar Chart")
plt.ylabel("Count")
plt.savefig("matplotlib_bar_chart.png")
print("  -> Saved 'matplotlib_bar_chart.png'")
plt.close()


# --- Histogram ---
# Shows the distribution of a single numerical variable.
print("# Example: A histogram")
hist_data = np.random.randn(1000) # Sample data from a normal distribution

plt.figure(figsize=(8, 5))
plt.hist(hist_data, bins=30, color='skyblue', edgecolor='black')
plt.title("Matplotlib Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("matplotlib_histogram.png")
print("  -> Saved 'matplotlib_histogram.png'")
plt.close()


# ---------------------------------
# Section 3.3: Subplots
# ---------------------------------
print("\n# --- 3.3: Subplots --- #\n")

# You can display multiple plots in one figure using subplots.
# plt.subplots() is a convenient function that creates a figure and a grid of subplots.
print("# Example: Creating a figure with 1 row and 2 columns of subplots")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot on the first subplot (ax1)
ax1.plot(x_data, y_data, color='red')
ax1.set_title('First Subplot (sin(x))')
ax1.grid(True)

# Plot on the second subplot (ax2)
ax2.scatter(x_scatter, y_scatter, color='green')
ax2.set_title('Second Subplot (Scatter)')

plt.suptitle("Matplotlib Subplots Example") # Add a centered title to the figure
plt.savefig("matplotlib_subplots.png")
print("  -> Saved 'matplotlib_subplots.png'")
plt.close()


# =================================================================================================
# 4. SEABORN
# =================================================================================================
# Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level
# interface for drawing attractive and informative statistical graphics. It often requires less
# code to create complex, aesthetically pleasing plots.

print("\n" + "="*20 + " 4. SEABORN CHEAT SHEET " + "="*20 + "\n")

# Seaborn works very well with Pandas DataFrames.
# We'll use the built-in 'tips' dataset.
tips = sns.load_dataset("tips")
print("Loaded 'tips' dataset for Seaborn examples:\n")
print(tips.head())


# ---------------------------------
# Section 4.1: Distribution Plots
# ---------------------------------
print("\n# --- 4.1: Distribution Plots --- #\n")

# --- Histogram (histplot) ---
# Similar to Matplotlib's hist, but with added features like a KDE curve.
print("# Example: A histogram of total bill amounts")
plt.figure(figsize=(8, 5))
sns.histplot(data=tips, x="total_bill", kde=True, bins=20)
plt.title("Distribution of Total Bill Amounts")
plt.savefig("seaborn_histogram.png")
print("  -> Saved 'seaborn_histogram.png'")
plt.close()

# --- Box Plot (boxplot) ---
# Shows the distribution of quantitative data in a way that facilitates comparisons between variables.
# It displays the five-number summary: min, first quartile, median, third quartile, and max.
print("# Example: A box plot showing total bill by day")
plt.figure(figsize=(8, 5))
sns.boxplot(data=tips, x="day", y="total_bill", palette="pastel")
plt.title("Total Bill Distribution by Day")
plt.savefig("seaborn_boxplot.png")
print("  -> Saved 'seaborn_boxplot.png'")
plt.close()


# ---------------------------------
# Section 4.2: Relational Plots
# ---------------------------------
print("\n# --- 4.2: Relational Plots --- #\n")

# --- Scatter Plot (scatterplot) ---
# The primary way to visualize the relationship between two numeric variables.
# Seaborn makes it easy to add a third variable via color (hue), size, or style.
print("# Example: A scatter plot of tip vs. total bill, colored by time of day")
plt.figure(figsize=(8, 5))
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", style="smoker", size="size")
plt.title("Tip Amount vs. Total Bill")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Move legend outside
plt.tight_layout()
plt.savefig("seaborn_scatterplot.png")
print("  -> Saved 'seaborn_scatterplot.png'")
plt.close()


# ---------------------------------
# Section 4.3: Categorical Plots
# ---------------------------------
print("\n# --- 4.3: Categorical Plots --- #\n")

# --- Count Plot (countplot) ---
# A special kind of bar plot that shows the count of observations in each category.
print("# Example: A count plot of smokers vs. non-smokers")
plt.figure(figsize=(8, 5))
sns.countplot(data=tips, x="smoker", palette="viridis")
plt.title("Count of Smokers vs. Non-Smokers")
plt.savefig("seaborn_countplot.png")
print("  -> Saved 'seaborn_countplot.png'")
plt.close()

# --- Violin Plot (violinplot) ---
# A combination of a box plot and a kernel density estimate (KDE). It shows the distribution
# of the data and its probability density.
print("# Example: A violin plot showing total bill by day")
plt.figure(figsize=(8, 5))
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True, palette="muted")
plt.title("Distribution of Total Bill by Day and Sex")
plt.savefig("seaborn_violinplot.png")
print("  -> Saved 'seaborn_violinplot.png'")
plt.close()


# ---------------------------------
# Section 4.4: Matrix and Grid Plots
# ---------------------------------
print("\n# --- 4.4: Matrix and Grid Plots --- #\n")

# --- Heatmap ---
# A graphical representation of data where values are depicted by color.
print("# Example: A heatmap of flight passenger data")
flights = sns.load_dataset("flights").pivot("month", "year", "passengers")
plt.figure(figsize=(10, 8))
sns.heatmap(flights, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
plt.title("Heatmap of Monthly Flight Passengers")
plt.savefig("seaborn_heatmap.png")
print("  -> Saved 'seaborn_heatmap.png'")
plt.close()

# --- Pair Plot (pairplot) ---
# Creates a grid of scatter plots for each pair of numerical variables in a dataset.
# The diagonal shows the distribution of each individual variable.
print("# Example: A pair plot of the 'iris' dataset")
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species", palette="husl", markers=["o", "s", "D"])
plt.savefig("seaborn_pairplot.png")
print("  -> Saved 'seaborn_pairplot.png'")
plt.close()