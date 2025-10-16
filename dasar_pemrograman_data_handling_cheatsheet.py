# =================================================================================================
# COMPREHENSIVE CHEAT SHEET: DASAR PEMROGRAMAN & DATA HANDLING
# =================================================================================================
# This file contains detailed, runnable examples for fundamental Python programming concepts
# and basic data (tensor) handling with PyTorch. It is designed for beginners.
#
# Sections:
# 1. Dasar Python: Core Python programming concepts.
# 2. Dasar PyTorch: Introduction to Tensors and basic operations.
# =================================================================================================

import torch

# =================================================================================================
# 1. DASAR PYTHON (PYTHON FUNDAMENTALS)
# =================================================================================================
print("="*20 + " 1. PYTHON FUNDAMENTALS CHEAT SHEET " + "="*20 + "\n")

# ---------------------------------
# Section 1.1: Variables and Data Types
# ---------------------------------
print("# --- 1.1: Variables and Data Types --- #\n")
# Variables are containers for storing data values.

# String: for text data.
my_string = "Hello, Python!"
print(f"String: '{my_string}' (type: {type(my_string)})")

# Integer: for whole numbers.
my_integer = 101
print(f"Integer: {my_integer} (type: {type(my_integer)})")

# Float: for numbers with a decimal point.
my_float = 3.14
print(f"Float: {my_float} (type: {type(my_float)})")

# Boolean: for True/False values.
my_boolean = False
print(f"Boolean: {my_boolean} (type: {type(my_boolean)})\n")


# ---------------------------------
# Section 1.2: Basic Data Structures
# ---------------------------------
print("# --- 1.2: Basic Data Structures --- #\n")

# --- List ---
# An ordered and mutable (changeable) collection. Allows duplicate members.
print("# List: Ordered and changeable")
my_list = ["apple", "banana", "cherry"]
print(f"Original list: {my_list}")
# Accessing items by index
print(f"  First item: {my_list[0]}")
# Changing an item
my_list[1] = "blueberry"
print(f"  List after changing an item: {my_list}")
# Adding an item
my_list.append("orange")
print(f"  List after appending an item: {my_list}\n")

# --- Tuple ---
# An ordered and immutable (unchangeable) collection. Allows duplicate members.
print("# Tuple: Ordered and unchangeable")
my_tuple = ("red", "green", "blue")
print(f"Tuple: {my_tuple}")
# You can access items, but you cannot change them.
# my_tuple[0] = "yellow"  # This would raise a TypeError\n")

# --- Dictionary ---
# An unordered, mutable, and indexed collection of key-value pairs. No duplicate keys.
print("# Dictionary: Unordered and changeable key-value pairs")
my_dict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(f"Original dictionary: {my_dict}")
# Accessing items by key
print(f"  Model: {my_dict['model']}")
# Changing a value
my_dict["year"] = 2020
print(f"  Dictionary after changing a value: {my_dict}")
# Adding a new key-value pair
my_dict["color"] = "red"
print(f"  Dictionary after adding an item: {my_dict}\n")


# ---------------------------------
# Section 1.3: Control Flow (Loops and Conditionals)
# ---------------------------------
print("# --- 1.3: Control Flow --- #\n")

# --- If-Elif-Else Statement ---
# Used for conditional execution.
print("# Conditional Statements (if/elif/else)")
x = 20
if x > 50:
  print("x is greater than 50")
elif x == 20:
  print("x is exactly 20")
else:
  print("x is less than 50 and not 20\n")

# --- For Loop ---
# Used for iterating over a sequence.
print("# For Loop: Iterating through a list")
for fruit in ["apple", "banana", "cherry"]:
  print(f"  Current fruit: {fruit}")
print("")

# --- While Loop ---
# Used for executing a set of statements as long as a condition is true.
print("# While Loop: Counting from 1 to 3")
i = 1
while i <= 3:
  print(f"  Current count: {i}")
  i += 1 # Increment i, or the loop will continue forever
print("")


# ---------------------------------
# Section 1.4: Functions
# ---------------------------------
print("# --- 1.4: Functions --- #\n")
# A function is a block of code which only runs when it is called.
# You can pass data, known as parameters, into a function.

# Defining a function
def calculate_area(width, height):
  """This function calculates the area of a rectangle."""
  area = width * height
  return area

# Calling the function
w = 5
h = 10
rectangle_area = calculate_area(w, h)
print(f"The area of a rectangle with width {w} and height {h} is {rectangle_area}.\n")


# =================================================================================================
# 2. DASAR PYTORCH (PYTORCH FUNDAMENTALS)
# =================================================================================================
print("\n" + "="*20 + " 2. PYTORCH FUNDAMENTALS CHEAT SHEET " + "="*20 + "\n")
# PyTorch is an open source machine learning library used for developing and training neural networks.
# Its core data structure is the Tensor.

# ---------------------------------
# Section 2.1: Tensor Creation and Attributes
# ---------------------------------
print("# --- 2.1: Tensor Creation and Attributes --- #\n")

# Create a tensor from a Python list.
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(f"Tensor created from list:\n{x_data}\n")

# Get tensor attributes.
print(f"Tensor Shape: {x_data.shape}") # Dimensions of the tensor
print(f"Tensor DataType: {x_data.dtype}") # Data type of elements
print(f"Device tensor is stored on: {x_data.device}\n") # CPU or GPU

# Create tensors with specific values.
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n {rand_tensor} \n")
print(f"Ones Tensor:\n {ones_tensor} \n")
print(f"Zeros Tensor:\n {zeros_tensor} \n")


# ---------------------------------
# Section 2.2: Tensor Operations
# ---------------------------------
print("# --- 2.2: Tensor Operations --- #\n")
tensor = torch.ones(2, 2)
print(f"Original Tensor:\n{tensor}\n")

# --- Indexing and Slicing ---
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}\n")

# --- Joining Tensors ---
# torch.cat stacks tensors along an existing dimension.
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"Concatenated Tensor (dim=1):\n{t1}\n")

# --- Arithmetic operations ---
# Matrix multiplication
matrix_mul = tensor.matmul(tensor.T) # .T is the transpose
print(f"Matrix multiplication (tensor @ tensor.T):\n{matrix_mul}\n")

# Element-wise product
element_mul = tensor.mul(tensor)
print(f"Element-wise multiplication (tensor * tensor):\n{element_mul}\n")


# ---------------------------------
# Section 2.3: CPU vs GPU
# ---------------------------------
print("# --- 2.3: CPU vs GPU --- #\n")
# PyTorch can move computations to a GPU to accelerate them.
if torch.cuda.is_available():
    device = "cuda"
    print("GPU is available. Moving tensor to GPU.")
    tensor = tensor.to(device)
    print(f"Tensor is now on device: {tensor.device}")
else:
    print("GPU not available. Tensor remains on CPU.")

# ---------------------------------
# Section 2.4: Reproducibility
# ---------------------------------
print("\n# --- 2.4: Reproducibility --- #\n")
# Setting a random seed is crucial for getting the same results across runs.
torch.manual_seed(42)
rand1 = torch.rand(2,2)
print(f"First random tensor with seed 42:\n{rand1}\n")

torch.manual_seed(42)
rand2 = torch.rand(2,2)
print(f"Second random tensor with seed 42 (should be identical):\n{rand2}\n")