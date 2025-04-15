

# \title{Password Analysis}
# \author{Ugochi Akoh}
# \date{April 2025}


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset from a CSV file
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
variance = np.var(values, ddof=1)
std_dev = np.std(values, ddof=1)

print(f"Variance: {variance:.2f}")
print(f"Standard Deviation: {std_dev:.2f}")

# Visualization
plt.figure(figsize=(15, 6))

# Plot histogram (bar chart)
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
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset from CSV file
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
# Set x-axis to start from 0
plt.xlim(0, None)

# Create box plot
fig, ax = plt.subplots(figsize=(2.5, 1.25), dpi=300, facecolor='#ADD8E6', edgecolor='k', alpha=0.75)
ax.set_facecolor('#E6F0FFBB')
ax.boxplot(values, patch_artist=True, vert=False)  

# Ensure tick labels are visible
ax.set_yticklabels(['Password \nLength'], fontsize=6.5, weight='bold')
ax.tick_params(axis='x', labelsize=7)
ax.tick_params(axis='y', labelsize=7)

# Set title and labels
ax.set_title(f'Box Plot of \nPassword Length', font='DejaVu Sans', fontsize=6.5, weight='bold')
ax.set_xlabel('Count', fontsize=6.5, color='black', alpha=0.95, font='DejaVu Sans', weight='bold')

# Adjust position of tick labels
plt.xticks(rotation=90)
plt.yticks(rotation=0)

import seaborn as sns
# Create boxplot using Seaborn
plt.figure(figsize=(10, 6))
sns.boxplot(x=values, color='skyblue', fliersize=5, linewidth=2, whis=1.5,
            boxprops=dict(facecolor='skyblue', edgecolor='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            medianprops=dict(color='red'))
plt.title('Boxplot of Length using Seaborn')
plt.xlabel('Length')
plt.show()

######################################################
# Box Plot with Central Tendencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Load dataset
data = pd.read_csv('./common_passwords.csv')

# Ensure the column of interest is named 'length'
if 'length' not in data.columns:
    raise ValueError("The column 'value' does not exist in the dataset.")
values = data['length']

# Calculate central tendency measures
mean = np.mean(values)
median = np.median(values)

# Create boxplot using Matplotlib
plt.figure(figsize=(12, 6), dpi=300)
plt.boxplot(values, vert=False, patch_artist=True,
boxprops=dict(facecolor='skyblue', color='black'),
whiskerprops=dict(color='black'),
capprops=dict(color='black'),
medianprops=dict(color='red'))

# Annotate central tendency measures
plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
plt.legend()
plt.title('Boxplot of Values using Matplotlib')
plt.xlabel('Length')
plt.show()

###########################################################################################
# Character composition
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Load dataset
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = './common_passwords.csv'
df = pd.read_csv(file_path)

# Correlation Matrix Heatmap
plt.figure(figsize=(10, 10))
sns.set(font_scale=1)

# Create correlation matrix for selected features
corr_matrix = df[['num_chars', 'num_digits', 'num_upper',
                  'num_lower', 'num_special', 'num_vowels', 'num_syllables']].corr()

sns.heatmap(data=corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Password Features")
plt.show()


# Check and compute entropy if not already present
if 'entropy' not in df.columns:
    def calculate_entropy(password):
        charset = 0
        if any(c.islower() for c in password): charset += 26
        if any(c.isupper() for c in password): charset += 26
        if any(c.isdigit() for c in password): charset += 10
        if any(c in "!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`~" for c in password): charset += 32
        return len(password) * np.log2(charset) if charset > 0 else 0
    df['entropy'] = df['password'].apply(calculate_entropy)

# Calculate Pearson correlation
correlation = df['length'].corr(df['entropy'], method='pearson')
print(f"Pearson Correlation between Password Length and Entropy: {correlation:.3f}")

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

# Count number that falls in each category
strength_counts = df['strength'].value_counts().reindex(category_order).fillna(0).astype(int)

# For the bar chart
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

# For the pie chart
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

# Adding count labels on top of bars
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

#####################################################################
# Estimated crack time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Load dataset from csv file
file_path = './common_passwords.csv'
df = pd.read_csv(file_path)

# Entropy function
def calculate_entropy(password):
    charset = 0
    if any(c.islower() for c in password): charset += 26
    if any(c.isupper() for c in password): charset += 26
    if any(c.isdigit() for c in password): charset += 10
    if any(c in "!@#$%^&*()-_=+[{]}\|;:'\",<.>/?`~" for c in password): charset += 32
    return len(password) * np.log2(charset) if charset > 0 else 0

# crack time in seconds function
def crack_time_seconds(entropy, guesses_per_second=1e10):
    return 2 ** entropy / guesses_per_second

# convert time to human readable format function
def human_readable_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}hr"
    elif seconds < 31536000:
        return f"{seconds/86400:.1f}days"
    else:
        return f"{seconds/31536000:.1f}yrs"

# categorize crack time function
def crack_time_category(seconds):
    if seconds < 60:
        return "Instant"
    elif seconds < 3600:
        return "Minutes"
    elif seconds < 86400:
        return "Days"
    elif seconds < 604800:  # 7 days
        return "Weeks"
    else:
        return "Years"


df['entropy'] = df['password'].apply(calculate_entropy)
df['length'] = df['password'].apply(len)
df['crack_time_sec'] = df['entropy'].apply(crack_time_seconds)
df['crack_time_human'] = df['crack_time_sec'].apply(human_readable_time)
df['crack_category'] = df['crack_time_sec'].apply(crack_time_category)


# Qualitative category summary
category_order = ['Instant', 'Minutes', 'Days', 'Weeks', 'Years']
summary = df['crack_category'].value_counts().reindex(category_order, fill_value=0)

# Bar chart: Crack Time Categories (Qualitative)
colors = ['#6baed6', '#a6c8f0', '#7cb0e7', '#5298de', '#297fd5']  

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
bars = ax.bar(summary.index, summary.values, color=colors)

# For count and percentage labels
total = summary.sum()
for bar in bars:
    height = bar.get_height()
    percentage = (height / total) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{height}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=10)


ax.set_title('Estimated Crack Time by Category (Qualitative)', fontsize=14)
ax.set_xlabel('Crack Time Category', fontsize=12)
ax.set_ylabel('Number of Passwords', fontsize=12)
plt.tight_layout()
plt.savefig('crack_time_category_chart.png')
plt.show()

# Bar chart: Log-scale Crack Time (x-axis) vs Count
df['log_crack_time'] = np.log10(df['crack_time_sec'].replace(0, np.nan)).fillna(0).round().astype(int)
log_summary = df['log_crack_time'].value_counts().sort_index()

fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
bar2 = ax2.bar(log_summary.index.astype(str), log_summary.values, color='#297fd5')

# Add labels
for bar in bar2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height + 5, f'{height}',
             ha='center', va='bottom', fontsize=9)

ax2.set_title('Estimated Crack Time (Log Scale)', fontsize=14)
ax2.set_xlabel('Log10(Estimated Crack Time in Seconds)', fontsize=12)
ax2.set_ylabel('Number of Passwords', fontsize=12)
plt.tight_layout()
plt.savefig('crack_time_log_chart.png')
plt.show()

