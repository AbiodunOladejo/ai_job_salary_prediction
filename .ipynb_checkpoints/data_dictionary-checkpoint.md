# 📖 Data Dictionary — Global AI Job Salary Dataset

**Dataset:** Global AI Job Postings  
**Rows:** 15,000 (14,517 after outlier removal)  
**Columns:** 19  
**Missing values:** None

---

## Column Definitions

| Column | Data Type | Description | Example Values |
|---|---|---|---|
| `job_id` | String | Unique identifier for each job posting | AI00001, AI00002 |
| `job_title` | Categorical | Role name | ML Engineer, AI Specialist, Data Analyst |
| `salary_usd` | Numeric | **Target variable.** Annual salary in US Dollars | 63,000 — 399,095 |
| `salary_currency` | Categorical | Original currency before USD conversion | USD, EUR, GBP |
| `experience_level` | Categorical (Ordinal) | Required experience level | EN, MI, SE, EX |
| `employment_type` | Categorical | Nature of employment contract | FT, PT, CT, FL |
| `company_location` | Categorical | Country where the hiring company is based | United States, Germany, India |
| `company_size` | Categorical (Ordinal) | Number of employees at the company | S, M, L |
| `employee_residence` | Categorical | Country where the employee lives | United States, Canada, France |
| `remote_ratio` | Numeric | Degree of remote work allowed | 0, 50, 100 |
| `required_skills` | Text | Comma-separated list of required skills | Python, SQL, TensorFlow |
| `education_required` | Categorical (Ordinal) | Minimum education qualification required | Associate, Bachelor, Master, PhD |
| `years_experience` | Numeric | Years of experience required for the role | 0 — 20 |
| `industry` | Categorical | Industry sector of the hiring company | Technology, Healthcare, Finance |
| `posting_date` | Date | Date the job was posted | 2024-01-15 |
| `application_deadline` | Date | Last date to apply | 2024-02-15 |
| `job_description_length` | Numeric | Character count of the job description | 500 — 2,499 |
| `benefits_score` | Numeric | Quality rating of employee benefits (1–10) | 5.0 — 10.0 |
| `company_name` | Categorical | Name of the hiring company | TechCorp Inc, Smart Analytics |

---

## Categorical Value Codes

### `experience_level`
| Code | Meaning |
|---|---|
| EN | Entry-Level |
| MI | Mid-Level |
| SE | Senior |
| EX | Executive |

### `employment_type`
| Code | Meaning |
|---|---|
| FT | Full-Time |
| PT | Part-Time |
| CT | Contract |
| FL | Freelance |

### `company_size`
| Code | Meaning | Headcount |
|---|---|---|
| S | Small | Fewer than 50 |
| M | Medium | 50 to 250 |
| L | Large | More than 250 |

### `remote_ratio`
| Value | Meaning |
|---|---|
| 0 | On-site |
| 50 | Hybrid |
| 100 | Fully Remote |

---

## Feature Decisions

### Columns Dropped

| Column | Reason |
|---|---|
| `job_id` | Row identifier only — no predictive value |
| `salary_currency` | Redundant — `salary_usd` is already converted |
| `company_name` | Only 16 unique companies — already captured by `industry` and `company_size` |
| `posting_date` | Requires time-series treatment to be useful |
| `application_deadline` | Same reason as `posting_date` |
| `required_skills` | Comma-separated text — needs NLP treatment (reserved for future iteration) |
| `employee_residence` | Replaced by the engineered `same_country` feature |

### Columns Used as Features

| Column | Encoding | Justification |
|---|---|---|
| `job_title` | OneHotEncoder | Each title needs independent salary representation |
| `experience_level` | OneHotEncoder | Used inside pipeline for consistent preprocessing |
| `employment_type` | OneHotEncoder | No meaningful order between contract types |
| `company_location` | OneHotEncoder | Each country needs its own representation |
| `company_size` | OneHotEncoder | Used inside pipeline for consistent preprocessing |
| `education_required` | OneHotEncoder | Used inside pipeline for consistent preprocessing |
| `industry` | OneHotEncoder | Each sector needs independent salary representation |
| `remote_ratio` | Passthrough (numeric) | Already numeric — no transformation needed |
| `years_experience` | Passthrough (numeric) | Already numeric |
| `job_description_length` | Passthrough (numeric) | Already numeric |
| `benefits_score` | Passthrough (numeric) | Already numeric |
| `same_country` | Engineered binary | 1 if company_location equals employee_residence, else 0 |

### Outlier Handling

Salary outliers were removed using the IQR method before modelling:

```
Lower bound = Q1 − 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR
```

This removed 483 rows (~3.2% of the data).

**Justification:** Although high executive salaries are genuine data points, removing extreme values focuses model accuracy on the typical salary range that most users of this tool — job seekers and HR professionals — actually operate in. This decision is documented transparently so any reader can evaluate the trade-off.

---

## Potential Use Cases

| User | Use Case |
|---|---|
| Job seekers | Estimate expected salary before negotiating an offer |
| Students & career switchers | Compare salary potential across AI roles and locations |
| HR professionals | Benchmark compensation packages against market rates |
| Recruiters | Set competitive salary ranges when writing job postings |
| Companies | Conduct compensation analysis across geographies |

---

*Dataset used with permission for portfolio and educational purposes.*
