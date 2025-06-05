# 📊 Telco Customer Churn Prediction Model

This project uses the **Telco Customer Churn** dataset to build a machine learning model that predicts whether a customer is likely to leave a telecom company. It includes data preprocessing, exploratory data analysis, feature engineering, and model training using Python.

---

## 📂 Dataset

- **Source**: [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Format**: CSV
- **Columns**: Customer demographics, service subscriptions, billing info, and churn status.

---

## 📦 Installation

To run this project locally:

```bash
git clone https://github.com/9mithun9/Telco-Customer-Churn-Prediction-Model.git
cd Telco-Customer-Churn-Prediction-Model
pip install -r requirements.txt
```

Or install manually:

```bash
pip install notebook opendatasets pandas scikit-learn matplotlib seaborn
```

---

## 🚀 How to Run

Run the notebook locally:

```bash
jupyter notebook Telco-Customer-Churn-Prediction-Model.ipynb
```

Or open in Google Colab (recommended):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 🧠 ML Workflow

- 📥 Dataset download via Kaggle API  
- 🧹 Data cleaning & exploration  
- 📊 Feature engineering  
- 🧪 Model training (e.g., Logistic Regression)  
- ✅ Evaluation with accuracy, confusion matrix, and classification report

---

## 📈 Example Code

```python
import opendatasets as od

dataset_url = 'https://www.kaggle.com/datasets/blastchar/telco-customer-churn'
od.download(dataset_url)
```

---

## 🛠 Technologies Used

- Python 🐍
- Jupyter Notebook
- Pandas
- scikit-learn
- Matplotlib & Seaborn
- OpenDatasets

---

## 📁 Folder Structure

```
├── Telco-Customer-Churn-Prediction-Model.ipynb  # Main analysis notebook
├── telco-customer-churn/                        # Dataset folder (downloaded)
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset CSV file
├── README.md                                     # This documentation file
└── requirements.txt                              # Python dependencies

```

---

## ✍️ Author

- [Mithun Marshal](https://github.com/9mithun9)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
