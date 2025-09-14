# Automated-Data-Analyzer-Web-Application-
An interactive Flask-based web application that allows users to upload CSV files and instantly generate automated data analysis reports with summary statistics and visualizations.

🚀 Features

📂 Upload any CSV file through the web interface

📝 Dataset summary (rows, columns, dtypes, missing values, statistics)

📊 Visualizations including:

Histograms

Boxplots

Pie charts

Correlation heatmaps

Bar plots

👀 Sample rows preview

⚡ Easy-to-use browser interface

🖼️ Screenshots
Upload CSV

Dataset Summary

Sample Rows

Visualizations

🛠️ Tech Stack

Backend: Flask (Python)

Frontend: HTML, CSS, Bootstrap

Data Analysis: Pandas, NumPy

Visualizations: Matplotlib, Seaborn

Deployment: Localhost (Flask server)

📂 Project Structure
Data-Analyzer/
│── app.py              # Main Flask app  
│── templates/          # HTML templates (Jinja2)  
│── static/             # CSS, JS, and generated plots  
│── uploads/            # Uploaded CSV files  
│── requirements.txt    # Python dependencies  
│── README.md           # Project documentation  

⚡ Installation & Usage

Clone the repository

git clone https://github.com/your-username/Data-Analyzer.git
cd Data-Analyzer


Create a virtual environment & activate it

python -m venv venv
venv\Scripts\activate   # For Windows
source venv/bin/activate # For Mac/Linux


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


Open in browser

http://127.0.0.1:5000/

👨‍💻 Author

Developed by Pandey Sunny
💼 Aspiring Data Analyst | Python | SQL | Power BI | Machine Learning
