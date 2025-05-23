# Anomaly Detection Project

This project applies machine learning to detect anomalous patterns in user behavior on an e-commerce platform. The goal is to support cybersecurity applications such as fraud detection, intrusion prevention, and secure system monitoring.

# Use Case

Anomalies in user behavior may indicate:
- Fraudulent transactions
- Unauthorized access attempts
- Compromised user accounts

# Models Used
- Isolation Forest
- One-Class SVM
- Local Outlier Factor

These models were trained and evaluated on behavioral datasets to flag unusual activity.

# Tools & Technologies
- Python (3.8+)
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebooks
- [Wireshark](https://www.wireshark.org/) – for basic packet inspection (testing phase)
- [OWASP ZAP](https://owasp.org/www-project-zap/) – for internal web vulnerability scanning

## 🔐 Security Highlights
- Secure login flow with password hashing and HTTPS implementation
- Encrypted storage of sensitive user/transaction data (simulated environment)
- Integration of anomaly detection into backend workflow for real-time monitoring

# Getting Started

# Installation

```bash
git clone https://github.com/joetech-rgb/anomaly-detection-project.git
cd anomaly-detection-project
pip install -r requirements.txt
