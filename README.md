# Statistical Analysis of Password Strength

This project provides two different approaches (using VS Code with Jupyter Notebooks and Spyder IDE with a Python script) to perform statistical analysis on a dataset of common passwords (`common_passwords.csv`). The analysis focuses on password length, character composition, entropy, and estimated crack times.

## Dataset

Both options utilize the `common_passwords.csv` dataset located within their respective directories. This dataset contains a list of common passwords and potentially pre-calculated features like length and character counts (depending on the specific CSV version used by each option).

## General Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/Statistical-analysis-of-Password-strength.git 
    cd Statistical-analysis-of-Password-strength
    ```
    *(Replace the URL with the actual repository URL if available)*

2.  **Ensure Python:** Make sure you have Python 3.x installed on your system.

3.  **(Recommended) Create a Virtual Environment:** It's good practice to use a virtual environment to manage dependencies.
    ```bash
    # Create a virtual environment (e.g., named 'venv')
    python -m venv venv

    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate 
    ```

## Option 1: Using Visual Studio Code (Jupyter Notebook)

This option uses a Jupyter Notebook (`Passwords_analysis-Option 1.ipynb`) for interactive analysis within the Visual Studio Code environment.

### Prerequisites

*   Visual Studio Code installed.
*   Python extension for VS Code installed.
*   Jupyter extension for VS Code installed.

### Setup

Install the required Python libraries:
```bash
pip install pandas matplotlib seaborn numpy zxcvbn notebook
```

### Analysis

The notebook performs the following analysis:
*   Loads the `common_passwords.csv` dataset.
*   Calculates password length.
*   Determines the size of the character set used (numbers, lowercase, uppercase, special characters).
*   Calculates password entropy using a custom formula based on character set size and length.
*   Calculates password entropy, strength score (0-4), and estimated online crack time using the `zxcvbn` library.
*   Generates various visualizations:
    *   Histograms and Box Plots for Length, Character Set Size, and Entropy.
    *   Scatter plots showing relationships between Length, Entropy, and Character Set Size.
    *   Bar chart of Zxcvbn strength scores.

### How to Run

1.  Open the `Statistical-analysis-of-Password-strength` folder in Visual Studio Code.
2.  Navigate to the `Option 1 Using Visual Studio Code/` directory in the VS Code explorer.
3.  Ensure the `common_passwords.csv` file is present in this directory.
4.  Open the `Passwords_analysis-Option 1.ipynb` file.
5.  Run the notebook cells sequentially to perform the analysis and view the results.

## Option 2: Using Spyder IDE (Python Script)

This option uses a standard Python script (`Script.py`) designed to be run in an IDE like Spyder.

### Prerequisites

*   Spyder IDE installed (or another Python IDE/environment).

### Setup

Install the required Python libraries:
```bash
pip install pandas numpy matplotlib scipy seaborn
```

### Analysis

The script performs the following analysis:
*   Loads the `common_passwords.csv` dataset.
*   Calculates basic statistics (mean, median, mode, variance, standard deviation) for password length.
*   Analyzes character composition (percentage of passwords containing lowercase, uppercase, digits, special characters). *Note: This section depends on specific columns like `num_lower`, `num_upper`, etc., being present in the CSV.*
*   Calculates password entropy using a formula based on character types present and length.
*   Classifies password strength based on calculated entropy ("Very Weak", "Weak", "Strong", "Very Strong").
*   Generates various visualizations:
    *   Bar chart and Box Plot for password length distribution.
    *   Bar chart and Pie chart for character type presence.
    *   Correlation heatmap for character counts (if columns exist).
    *   Bar chart and Pie chart for entropy-based strength classification.

### How to Run

1.  Open Spyder IDE.
2.  Navigate to the `Option 2 - Using Spyder IDE/` directory.
3.  Ensure the `common_passwords.csv` file is present in this directory.
4.  Open the `Script.py` file.
5.  Run the script (e.g., by pressing F5 in Spyder or using `python Script.py` in the terminal from within the directory).

## Results

Both options provide statistical insights into the characteristics of the passwords in the dataset, visualized through various plots. Option 1 leverages the `zxcvbn` library for a more sophisticated strength estimation, while Option 2 includes analysis of character composition and a different entropy calculation.

## License

This project is licensed under the terms of the LICENSE file.
