# Question Generation Comparison Report

Generated: 2025-12-24 17:33:55

## Summary

| Metric | LLM-Generated | Compositional |
|--------|--------------|---------------|
| Total questions | 28 | 25 |
| Avg n_steps | 8.71 | 8.77 |
| % naming columns explicitly | 82.1% | 15.4% |
| % code in hints | 7.1% | 0.0% |

## Per-Dataset Results

### 1. abcsds_pokemon
- Shape: 0 × 13
- Numeric cols: 0, Categorical: 0

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Implement a permutation test (10,000 permutations) to determ... | 9 | Yes | No |
| Using Monte Carlo simulation (100,000 draws), estimate the p... | 9 | Yes | No |

**Compositional Questions (0):**
*(No compositional questions generated)*

---

### 2. fedesoriano_heart-failure-prediction
- Shape: 918 × 12
- Numeric cols: 7, Categorical: 5

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| What is the adjusted R‑squared of a multiple linear regressi... | 7 | Yes | No |
| Is there a significant difference in average cholesterol lev... | 4 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Using the dataset, identify the numeric column that has the ... | 10 | No | No |
| Identify the numeric column that exhibits the greatest absol... | 9 | No | No |

---

### 3. fedesoriano_stroke-prediction-dataset
- Shape: 5110 × 12
- Numeric cols: 7, Categorical: 5

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Estimate a 95 % confidence interval for the difference in me... | 10 | Yes | No |
| Perform propensity‑score matching on age, hypertension, and ... | 11 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| In the given dataset, first determine the numeric column tha... | 10 | Yes | No |
| Identify the numeric column that exhibits the greatest absol... | 9 | Yes | No |

---

### 4. gregorut_videogamesales
- Shape: 16598 × 11
- Numeric cols: 7, Categorical: 4

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Using a train‑test split (80/20, random_state=42), train a l... | 9 | Yes | No |
| Build a ridge regression model (alpha=1.0) to predict Global... | 9 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Using the dataset, perform a multiple linear regression wher... | 10 | No | No |
| For the numeric column that has the highest absolute skewnes... | 9 | No | No |

---

### 5. imakash3011_customer-personality-analysis
- Shape: 2240 × 1
- Numeric cols: 0, Categorical: 1

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Estimate the average treatment effect of having a child at h... | 11 | No | No |
| Using a logistic regression model with predictors Income, Re... | 7 | No | No |

**Compositional Questions (1):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| How many columns have a missing-value proportion greater tha... | 3 | No | No |

---

### 6. mirichoi0218_insurance
- Shape: 1338 × 7
- Numeric cols: 4, Categorical: 3

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Using 5‑fold cross‑validation, what is the average RMSE of a... | 9 | Yes | No |
| Using bootstrapping (1000 resamples), estimate the 95% confi... | 9 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Identify the numeric column with the greatest variance and t... | 10 | No | No |
| For the numeric column that has the greatest absolute skewne... | 9 | No | No |

---

### 7. neuromusic_avocado-prices
- Shape: 18249 × 14
- Numeric cols: 11, Categorical: 2

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Build a multivariate regression model to predict AveragePric... | 9 | Yes | No |
| Do organic and conventional avocados exhibit different price... | 5 | No | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Among the numeric columns, determine which column has the la... | 10 | No | No |
| Identify the numeric column that has the greatest absolute s... | 9 | No | No |

---

### 8. pavansubhasht_ibm-hr-analytics-attrition-dataset
- Shape: 1470 × 35
- Numeric cols: 26, Categorical: 9

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Create a multiple linear regression model to predict Monthly... | 8 | Yes | No |
| Using ANOVA, test whether the mean TotalWorkingYears differs... | 6 | Yes | Yes |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Using the dataset, perform a multiple linear regression wher... | 10 | No | No |
| Identify the numeric column with the greatest absolute skewn... | 9 | No | No |

---

### 9. russellyates88_suicide-rates-overview-1985-to-2016
- Shape: 27820 × 12
- Numeric cols: 7, Categorical: 5

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| After performing multiple imputation to fill missing HDI val... | 10 | No | No |
| Investigate whether GDP per capita mediates the relationship... | 12 | No | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Identify the numeric column with the highest variance in the... | 10 | No | No |
| For the numeric column that shows the greatest absolute skew... | 9 | No | No |

---

### 10. shivamb_netflix-shows
- Shape: 8807 × 12
- Numeric cols: 1, Categorical: 11

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| What are the most influential predictors of a title being ra... | 10 | Yes | No |
| Construct a decision tree (max_depth=4, random_state=42) to ... | 11 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Identify the numeric column that exhibits the highest varian... | 9 | No | No |
| For the numeric column with the highest variance, determine ... | 5 | No | No |

---

### 11. spscientist_students-performance-in-exams
- Shape: 1000 × 8
- Numeric cols: 3, Categorical: 5

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Calculate the 95% confidence interval for the mean differenc... | 7 | Yes | No |
| Using hierarchical linear modeling, estimate the variance co... | 8 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Identify the numeric column with the highest absolute skewne... | 9 | No | No |
| When you select the numeric column that exhibits the greates... | 9 | No | No |

---

### 12. uciml_breast-cancer-wisconsin-data
- Shape: 569 × 33
- Numeric cols: 31, Categorical: 1

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Investigate whether the standard error features (e.g., radiu... | 9 | Yes | No |
| Implement a nested cross‑validation pipeline (outer 5‑fold, ... | 10 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| In the given data set, first identify the numeric column tha... | 10 | Yes | No |
| Among all numeric columns, identify the column with the grea... | 9 | Yes | No |

---

### 13. uciml_pima-indians-diabetes-database
- Shape: 768 × 9
- Numeric cols: 9, Categorical: 0

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Apply a stepwise forward selection (using p‑values) to ident... | 9 | Yes | No |
| Conduct a bootstrap analysis (1,000 resamples) to estimate t... | 9 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Identify the numeric column that exhibits the greatest varia... | 10 | No | No |
| For the numeric column that has the highest absolute skewnes... | 9 | No | No |

---

### 14. uciml_red-wine-quality-cortez-et-al-2009
- Shape: 1599 × 12
- Numeric cols: 12, Categorical: 0

**LLM Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Apply a Box‑Cox transformation to the 'residual sugar' varia... | 8 | Yes | Yes |
| Using a permutation test (10,000 permutations, random_state=... | 9 | Yes | No |

**Compositional Questions (2):**
| Question | n_steps | Explicit Cols? | Code in Hint? |
|----------|---------|----------------|---------------|
| Using the numeric column that exhibits the highest variance ... | 10 | No | No |
| Identify the numeric column that exhibits the greatest absol... | 9 | No | No |

---
