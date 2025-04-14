

# \title{Password Analysis}
# \author{Ugochi Akoh}
# \date{April 2025}


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset from CSV file
file_path = './common_passwords.csv'

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!!!")
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("The file is empty.")
    exit()
except pd.errors.ParserError:
    print("The file could not be parsed.")
    exit()

# Ensure the column 'length' exists
if 'length' not in data.columns:
    raise ValueError("The column 'length' does not exist in the dataset.")

values = data['length']

# Calculate Mean
mean = np.mean(values)
print(f"Mean: {mean:.2f}")

# Calculate Median
median = np.median(values)
print(f"Median: {median:.2f}")

# Calculate Mode
mode_result = stats.mode(values, keepdims=True)
mode = mode_result.mode[0] if mode_result.mode.size > 0 else None
mode_count = mode_result.count[0] if mode_result.count.size > 0 else None
print(f"Mode: {mode}, Frequency: {mode_count}")

# Calculate Variance & Standard Deviation
variance = np.var(values, ddof=1)
std_dev = np.std(values, ddof=1)

print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# Visualization
plt.figure(figsize=(15, 6))

# Plotting histogram (bar chart)
plt.subplot(1, 2, 1)
value_counts = data['length'].value_counts().sort_index()  
# Using correct column
plt.bar(value_counts.index, value_counts.values, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
if mode is not None:
    plt.axvline(mode, color='orange', linestyle='dashed', linewidth=1, label=f'Mode: {mode:.2f}')
plt.legend()
plt.title('BAr Chart of Length Frequencies')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

###########################################################################################################
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset from CSV file
file_path = './common_passwords.csv'

try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!!!")
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print("The file is empty.")
    exit()
except pd.errors.ParserError:
    print("The file could not be parsed.")
    exit()

# Ensure column 'length' exists
if 'length' not in data.columns:
    raise ValueError("The column 'length' does not exist in the dataset.")

values = data['length']

# Calculate Mean
mean = np.mean(values)
print(f"Mean: {mean:.2f}")

# Calculate Median
median = np.median(values)
print(f"Median: {median:.2f}")

# Calculate Mode
mode_result = stats.mode(values, keepdims=True)
mode = mode_result.mode[0] if mode_result.mode.size > 0 else None
mode_count = mode_result.count[0] if mode_result.count.size > 0 else None
print(f"Mode: {mode}, Frequency: {mode_count}")

# Calculate Variance & Standard Deviation
variance = np.var(values, ddof=1)  # Sample variance (unbiased)
std_dev = np.std(values, ddof=1)  # Sample standard deviation (unbiased)

print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# Visualization
plt.figure(figsize=(15, 6), dpi=300)

# Plot histogram
plt.subplot(1, 2, 1)
value_counts = values.value_counts().sort_index()
plt.bar(value_counts.index, value_counts.values, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
if mode is not None:
    plt.axvline(mode, color='orange', linestyle='dashed', linewidth=1, label=f'Mode: {mode}')
plt.legend()
plt.title('Bar Chart of Length Frequencies')
plt.xlabel('Password Length')
plt.ylabel('Frequency')
# Set the x-axis to start from 0
plt.xlim(0, None)

# To Create box plot
fig, ax = plt.subplots(figsize=(2.5, 1.25), dpi=300, facecolor='#ADD8E6', edgecolor='k', alpha=0.75)
ax.set_facecolor('#E6F0FFBB')
ax.boxplot(values, patch_artist=True, vert=False)  
# Ensure vert=false for a horizontal box plot

# Ensure the tick labels are visible
ax.set_yticklabels(['Password \nLength'], fontsize=6.5, weight='bold')
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=7)

# Set title and labels
ax.set_title(f'Box Plot of \nPassword Length', font='DejaVu Sans', fontsize=6.5, weight='bold')
ax.set_xlabel('Count', fontsize=6.5, color='black', alpha=0.95, font='DejaVu Sans', weight='bold')

# Adjust the position of the tick labels
plt.xticks(rotation=90)
plt.yticks(rotation=0)

import seaborn as sns
# Creating a boxplot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x=values, color='skyblue', fliersize=5, linewidth=2, whis=1.5,
            boxprops=dict(facecolor='skyblue', edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='red'))
plt.title('Boxplot of Length using Seaborn')
plt.xlabel('Length')
plt.show()

#Character composition
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Load the dataset from the location
file_path = './common_passwords.csv'
df = pd.read_csv(file_path)

# Assuming the dataset has columns 'num_lower', 'num_upper', 'num_digits', 'num_special', and 'length'
char_types = ['num_lower', 'num_upper', 'num_digits', 'num_special']

# Check if the columns exist in the DataFrame
missing_columns = [col for col in char_types if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
else:
    # Calculate character type composition
    char_ratios = df[char_types].div(df['length'], axis=0) * 100

    # Presence of character types
    has_types = {
        'Lowercase': (df['num_lower'] > 0).mean() * 100,
        'Uppercase': (df['num_upper'] > 0).mean() * 100,
        'Digits': (df['num_digits'] > 0).mean() * 100,
        'Special': (df['num_special'] > 0).mean() * 100
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(has_types.keys()), y=list(has_types.values()), ax=ax)
    ax.set_title('Percentage of Passwords Containing Different Character Types')
    ax.set_ylabel('Percentage (%)')
    ax.yaxis.set_major_formatter(PercentFormatter())
    
    # Save figure
    plt.savefig('char_type_presence.png')
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.pie(has_types.values(),
        labels=has_types.keys(),
        autopct='%1.1f%%',
        startangle=140,
        colors=['#66b3ff','#99ff99','#ffcc99','#ff9999'])

ax2.set_title('Character Type Presence in Passwords')
plt.tight_layout()
plt.savefig('char_type_pie.png')
plt.show()

plt.show()

##############################################################################
#import tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset from the location
file_path = './common_passwords.csv'
df = pd.read_csv(file_path)


# Compute basic statistics
print(f"Mean Password Length: {df['length'].mean():.2f}")
print(f"Median Password Length: {df['length'].median():.2f}")
print(f"Mode Password Length(s): \n{df['length'].mode()}")

# Histogram to visualize password length distribution
plt.figure(figsize=(10, 5))
df['length'].hist(bins=5, color="blue", edgecolor="black")
plt.title("Distribution of Password Lengths")
plt.xlabel("Password Length")
plt.ylabel("Frequency")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the plot
plt.show()

# Set figure size
plt.figure(figsize=(10, 10))

# Set seaborn font scale
sns.set(font_scale=1)

# Ensure your dataframe is properly referenced
corr_matrix = df[['num_chars', 'num_digits', 'num_upper',
                  'num_lower', 'num_special', 'num_vowels', 'num_syllables']].corr()

# Create heatmap
sns.heatmap(data=corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Show plot
plt.show()

####################################################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import string
import matplotlib.patches as mpatches 

# Load the dataset
file_path = './common_passwords.csv'
df = pd.read_csv(file_path)

# Function to calculate entropy of a password
def calculate_entropy(password):
    R = 0
    if isinstance(password, str):
        if any(c.islower() for c in password):
            R += 26
        if any(c.isupper() for c in password):
            R += 26
        if any(c.isdigit() for c in password):
            R += 10
        if any(c in string.punctuation for c in password):
            R += len(string.punctuation)

        if R == 0:
            return 0
        L = len(password)
        entropy = L * math.log2(R)
        return round(entropy, 2)
    else:
        return 0

# Function to classify passwords by entropy
def classify_entropy_strength(entropy):
    if entropy <= 35:
        return "Very Weak"
    elif entropy <= 59:
        return "Weak"
    elif entropy <= 119:
        return "Strong"
    else:
        return "Very Strong"

# Apply entropy and classification
df['entropy'] = df['password'].apply(calculate_entropy)
df['strength'] = df['entropy'].apply(classify_entropy_strength)

# Order and color mapping
category_order = ['Very Weak', 'Weak', 'Strong', 'Very Strong']
colors = {
    "Very Weak": "#FF9999",
    "Weak": "#FFCC99",
    "Strong": "#99CC99",
    "Very Strong": "#66B2FF"
}

# Count how many fall in each category
strength_counts = df['strength'].value_counts().reindex(category_order).fillna(0).astype(int)

#For the bar chart
plt.figure(figsize=(8, 4), dpi=300)
bars = plt.bar(category_order, strength_counts[category_order], color=[colors[c] for c in category_order])
plt.title("Password Strength Classification (Bar Chart)")
plt.ylabel("Number of Passwords")
plt.xlabel("Strength Category")

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 10, int(yval), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

#for the pie chart
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
wedges, texts, autotexts = ax.pie(
    strength_counts[category_order],
    colors=[colors[c] for c in category_order],
    labels=None,
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 9}
)

# Create legend
legend_labels = [f"{cat} ({strength_counts[cat]})" for cat in category_order]
ax.legend(wedges, legend_labels, title="Strength Category", loc="center left", bbox_to_anchor=(1, 0.5))
ax.set_title("Password Strength Distribution (Entropy-based)")
plt.tight_layout()
plt.show()

# Total count for percentage calc
total = strength_counts.sum()

# Create figure
plt.figure(figsize=(8, 5))
bars = plt.bar(
    category_order,
    strength_counts[category_order],
    color=[colors[c] for c in category_order]
)

# Add count labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 5,
        f"{int(yval)}",
        ha='center',
        va='bottom',
        fontsize=9
    )

plt.title("Password Strength Classification (Bar Chart)")
plt.ylabel("Number of Passwords")
plt.xlabel("Strength Category")

# Create legend with percentages
legend_labels = [
    f"{cat} - {strength_counts[cat]/total:.1%}"
    for cat in category_order
]
patches = [mpatches.Patch(color=colors[cat], label=legend_labels[i])
           for i, cat in enumerate(category_order)]

plt.legend(handles=patches, title="Strength %", loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.show()


