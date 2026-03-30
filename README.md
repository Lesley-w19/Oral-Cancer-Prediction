# Oral Cancer Analysis Project

**Understanding the Progression of Oral Cancer Using Data Analytics**



## Project Overview

This project explores how **demographic, clinical, and lifestyle factors** influence oral cancer outcomes such as tumor size, cancer stage, survival rates, and economic burden.

The goal is to uncover actionable insights that can support **early diagnosis, better treatment decisions, and reduced healthcare costs**.

---

## Project Objectives

* Analyze the impact of **age, gender, and country** on oral cancer prevalence
* Evaluate how **lifestyle factors** (tobacco, alcohol, HPV) affect diagnosis rates
* Assess the influence of **treatment types and early diagnosis** on survival outcomes
* Investigate **economic burden** (treatment cost and lost workdays)
* Build predictive models to identify **key risk factors and outcomes** 

---

## Research Questions

1. **Demographic Influence**

   * How do age, gender, and country affect oral cancer likelihood?

2. **Risk Factor Prediction**

   * What are the most significant predictors of oral cancer diagnosis?

3. **Treatment & Survival**

   * How do early diagnosis and treatment types impact survival rates and cost?

4. **Lifestyle Impact**

   * How do tobacco, alcohol, diet, and sun exposure affect tumor size and cancer stage?

5. **Economic Burden**

   * What factors influence treatment costs across patient groups?

---

## 🧹 Data Cleaning & Preprocessing

* Handled **missing values** and visualized them using heatmaps
* Removed **duplicate records**
* Standardized **column names and formats**
* Fixed **data types** for consistency
* Treated **outliers** using statistical techniques
* Performed **exploratory data analysis (EDA)**

---

## Key Findings

### 1. Demographics

* Oral cancer prevalence increases slightly with age
* Minimal differences across gender
* Country-specific variations observed (e.g., higher rates in Kenya and Russia)

### 2. Lifestyle Risk Factors

* **Tobacco, alcohol, and HPV** strongly increase diagnosis rates
* Tobacco users showed **over 50% diagnosis rate**

### 3. Predictive Modeling

* Models used:

  * Linear Regression
  * Random Forest Regression
  * Gradient Boosting Regression

* **Best Model:** Gradient Boosting

  * R² ≈ 0.98
  * Lowest MSE (10.74)
  * Most accurate predictions

### 4. Treatment & Survival

* Early diagnosis leads to:

  * Smaller tumor size
  * Higher survival rates
  * Lower treatment costs

### 5. Economic Impact

* Costs vary significantly based on:

  * Stage at diagnosis
  * Treatment type
  * Patient demographics

---

## Insights & Recommendations

* Oral cancer outcomes are influenced by a **combination of factors**, not just one
* **Early detection is critical** for improving survival and reducing costs
* Promote:

  * Early screening programs
  * Affordable treatment access
  * Integration of **HPV testing** into routine checkups

---

## 🛠️ Tools & Technologies

* **Python** (Data Analysis & Modeling)
* **Pandas, NumPy** (Data manipulation)
* **Matplotlib, Seaborn** (Visualization)
* **Scikit-learn** (Machine Learning Models)
* **Excel / Spreadsheets** (Exploratory Analysis)

---


## ▶️ How to Run the Project

1. Clone the repository

```bash
git clone https://github.com/your-username/oral-cancer-analysis.git
```

2. Navigate to the project folder

```bash
cd oral-cancer-analysis
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the notebooks or scripts

```bash
jupyter notebook
```

---

## Conclusion

This project highlights that oral cancer is driven by an **ecosystem of factors**, including demographics, lifestyle, and treatment decisions.

The strongest takeaway:

`Early diagnosis significantly improves outcomes and reduces economic burden.`

---

## 👩‍💻 Author

**Lesley Wanjiku Kamamo**

* Data Analyst | Data Science Enthusiast
* Passionate about solving real-world problems using data

---

## License

This project is for academic and research purposes.
