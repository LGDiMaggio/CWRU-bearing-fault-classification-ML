# CWRU-bearing-fault-classification-ML
A machine learning project for classifying bearing faults using the CWRU dataset, with models built using Python and various ML techniques.

## Project Description
This repository hosts a machine learning project focused on fault classification in bearings using the widely recognized Case Western Reserve University (CWRU) dataset. It employs various machine learning models to predict bearing failures, showcasing implementations in Python that are often cited but seldom demonstrated in practical applications.

### Motivation
This project is crafted to bridge the gap between theoretical machine learning applications in fault diagnosis and practical, executable examples. It leverages the CWRU dataset, a cornerstone in academic research on bearing faults, to provide a robust foundation for scholars and practitioners alike. By openly sharing these implementations, this project aims to foster a deeper understanding and broader application of machine learning in industrial diagnostics.

## Features
- Comprehensive data preprocessing and exploration.
- Application of dimensionality reduction techniques like PCA.
- Utilization of machine learning models such as SVM, kNN, and XGBoost.
- Detailed evaluation of model performance using metrics like accuracy, precision, recall, and F1-score.
- Visualization of results through histograms, violin plots, and t-SNE plots.

## Getting Started

1. **Clone the Repository**:
```
git clone https://github.com/LGDiMaggio/CWRU-bearing-fault-classification-ML.git
```
### Set Up Your Environment

#### Windows
1. **Create a Virtual Environment**:
   - Open Command Prompt (or PowerShell).
   - Navigate to the project directory:
   ```
   cd CWRU-bearing-fault-classification-ML
   ```
   - Create a virtual environment:
   ```
   python -m venv venv
   ```

2. **Activate the Virtual Environment**:
   - In Command Prompt:
   ```
   venv\Scripts\activate
   ```
   - In PowerShell:
   ```
   .\venv\Scripts\Activate
   ```

3. **Install Dependencies**:
   - Ensure your virtual environment is activated, then install the required packages:
   ```
   pip install -r requirements.txt
   ```

#### Linux and macOS
1. **Create a Virtual Environment**:
   - Open a terminal and navigate to the project directory:
   ```
   cd CWRU-bearing-fault-classification-ML
   ```
   - Create a virtual environment:
   ```
   python3 -m venv venv
   ```

2. **Activate the Virtual Environment**:
   ```
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   - With the virtual environment activated, install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Explore the Notebooks:
- Navigate through the Jupyter notebooks provided in the repository to understand the workflows and methodologies applied.

## Contributing
Contributions to this project are warmly welcomed. Here are some ways you can contribute:
- Extend existing methodologies or introduce new analysis techniques.
- Enhance visualizations or contribute additional datasets.
- Review or improve documentation.

Please fork the repository, make your changes, and submit a pull request for review.

## License
This project is licensed under the GNU General Public License v3.0. It is free software, and you are welcome to redistribute it under certain conditions. There is **NO WARRANTY**, to the extent permitted by law. For more details, see the LICENSE file in this repository.

## Acknowledgments
- Case Western Reserve University for providing the dataset used in this project.
- Researchers and developers whose work has contributed to the methods applied here.

## Why Share This Project?
Many of the techniques used in this project are frequently mentioned in scientific articles concerning machine learning for system diagnostics. However, practical implementations are not commonly available, thus this project aims to serve as a resource for scholars entering this field. It is envisioned as a living educational tool, open to further development and contributions, particularly through the addition of new notebooks or methods.
