# 📊Automated-Data-Analyzer-Web-Application-
An interactive **Flask-based web application** that allows users to upload CSV files and automatically generate **statistical summaries and visualizations** for exploratory data analysis (EDA).  

---

## 🚀 Features  

- 📂 **CSV Upload Support** (up to 500 MB) with secure validation  
- 📝 **Dataset Summary**: Data types, non-null counts, unique values, mean & std deviation  
- 📊 **Visualizations**:  
  - Histograms  
  - Boxplots  
  - Correlation heatmaps  
  - Bar charts (top categories)  
  - Pie charts (categorical distributions)  
- 👀 **Dataset Preview**: View top 10 rows and column-level metadata  
- ⚡ **Large Dataset Handling**: Efficient sampling for datasets >100k rows  
- 🔒 **Error Handling**: User-friendly messages for oversized or invalid files  

---

## 🖼️ Screenshots  

### Upload CSV  
![Upload Screenshot](./screenshots/upload.png)  

### Dataset Summary  
![Summary Screenshot](./screenshots/summary.png)  

### Visualizations  
![Visualizations Screenshot](./screenshots/visuals.png)  

---

## 🛠️ Tech Stack  

- **Backend:** Flask (Python)  
- **Frontend:** HTML, CSS, Jinja2 Templates  
- **Data Analysis:** Pandas, NumPy  
- **Visualizations:** Matplotlib, Seaborn  
- **Deployment:** Localhost (Flask server)  

---

## 📂 Project Structure  

Data-Analyzer/
│── app.py # Main Flask application
│── templates/ # HTML templates (upload, results)
│── static/ # CSS, JS, and generated plots
│ └── plots/ # Auto-generated visualizations
│── uploads/ # Uploaded CSV files
│── requirements.txt # Python dependencies
│── README.md # Project documentation


---

## Create a virtual environment & activate it

python -m venv venv
venv\Scripts\activate      # For Windows  
source venv/bin/activate   # For Mac/Linux  

---

## Install dependencies

pip install -r requirements.txt

---

## Run the Flask app

python app.py


Open in browser
---

http://127.0.0.1:5000/

👨‍💻 Author

Developed by Pandey Sunny
💼 Aspiring Data Scientist | Data Analyst | Python | SQL | Power BI | Machine Learning Engineer
