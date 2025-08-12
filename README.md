# Practical-Assignment-3
Comparing Classifiers on Portuguese Bank Marketing Dataset

# Overview
We compared four classifiers: K-Nearest Neighbors (KNN), Logistic Regression (LR), Decision Tree (DT), and Support Vector Machine with RBF kernel (SVM). The models were applied on the Portuguese Bank Marketing dataset (bank-additional-full.csv).
The goal was to predict if a client will subscribe to a term deposit (target = y).
We chose AUC as the primary metric because the dataset is imbalanced, and AUC reflects the model’s ranking ability independent of threshold.

# Business Understanding
The Portuguese bank’s marketing team runs outbound telemarketing campaigns to promote term deposits.
Every call represents time, cost, and customer goodwill, so prioritizing who to call is critical.

The business challenge is to increase conversion rates (more “yes” responses) while reducing wasted calls (contacting customers who are unlikely to subscribe). This is not just about accuracy, the cost of a false positive (calling a customer who will say “no”) is significantly different from the cost of a false negative (missing a customer who would say “yes”).

# Key considerations:
- **Resource Optimization:** Sales teams have limited capacity. Predicting high-likelihood leads ensures efficient resource allocation.
- **Customer Experience:** Minimizing unnecessary calls improves customer perception and reduces attrition risk.
- **Regulatory Compliance:** Many regions limit unsolicited marketing, so targeting only likely customers is not just profitable but also compliant.
- **Why Pre-Contact Models?**
    We explicitly exclude the duration feature from the model because:
  1. duration is the length of the call after it has happened.
  2. It is strongly correlated with the outcome (y) — longer calls tend to be successful.
  3. Including it would cause data leakage: the model would be “cheating” by using information that isn’t available at the time of decision.

## Choice of Primary Metric: AUC
We chose AUC (Area Under the ROC Curve) as the primary evaluation metric because:
  1. The dataset is imbalanced: most customers say “no”.
  2. Accuracy could be misleading (predicting “no” for everyone would yield high accuracy).
  3. AUC measures the model’s ability to rank customers, which is what matters in lead prioritization — sales teams can call the top-ranked customers first.

This Target Variable Distribution chart shows the severe class imbalance in the dataset.

"No" (Did not subscribe to term deposit): The overwhelming majority (around 36,000 customers).

"Yes" (Subscribed to term deposit): A much smaller group (roughly 4,000 customers).



<img width="480" height="393" alt="image" src="https://github.com/user-attachments/assets/f8bcfbc0-6f45-45fc-aae7-e8d401ac8f7b" />

## Data Manipulation:

Column names were standardized by removing spaces, special characters, and inconsistent casing to ensure smooth integration with Python libraries and ML pipelines.
This data cleaning step improves code readability and keeps feature names consistent across the entire workflow. More importantly, the data was preserved and unfrequently removed during this study.

## Age Distribution by Subscription Outcome

Observations:
- Most customers are between 30–40 years old, regardless of subscription outcome.
- The Yes curve (orange) has a noticeable presence in older age groups (50+), suggesting older clients are slightly more likely to subscribe.
- Younger clients (<25) rarely subscribe.

Implications for modeling/marketing: Age could be a useful predictor, but not in a simple linear way. Older demographics might be more receptive to term deposits.

<img width="471" height="393" alt="image" src="https://github.com/user-attachments/assets/f041aecf-40af-47dd-a1ca-1cfdf1577145" />

## Campaign (Number of Contacts During Current Campaign)

Observations:
- Strong right skew — most customers were contacted 1–3 times.
- Success rates drop after multiple contacts, with few Yes outcomes when campaign > 5.

Implications:
- Repeated calls to the same customer yield diminishing returns.
- Could be a negative predictor for subscription likelihood.

<img width="475" height="393" alt="image" src="https://github.com/user-attachments/assets/f3b46d71-1eb3-42b1-9b83-8070e347f48f" />

## pdays (Days Since Last Contact in a Previous Campaign)

Observations:
- Two main spikes:
    - 0 days — means customer was contacted very recently.
    - 999 days — code for “never contacted before”.
- A small cluster of Yes responses around low pdays (recent contacts).

Implications:
- Recency matters — recently contacted customers are more likely to respond positively.
- The 999 category needs to be treated as a separate feature (categorical, not continuous).
  
<img width="476" height="393" alt="image" src="https://github.com/user-attachments/assets/e508980d-43bc-43a2-ae49-746295d9c8c7" />

## previous (Number of Contacts Before This Campaign)

Observations:
- Most customers had 0 previous contacts.
- Customers with 1–2 prior contacts have slightly higher subscription rates.
- Few observations for higher values (>4).

Implications:
- A small history of prior contact is associated with higher likelihood of conversion.
- Excessive past contacts may indicate a low-probability lead.

<img width="456" height="393" alt="image" src="https://github.com/user-attachments/assets/3bfc8e8d-2649-406b-9d75-9474c266f8bf" />

## General Pattern Across All Plots
1. The features age, campaign, pdays, and previous show non-linear relationships with the target variable.
2. Some features (like pdays and campaign) have extreme skew and special codes (like 999) that need preprocessing.
3. Marketing-wise, these distributions highlight where to focus calls (e.g., mid-age groups, low campaign counts, recent contacts).



# Key Considerations:


## Employment Status Impact:

Occupation is a valuable segmentation variable for marketing, as it reveals clear differences in subscription likelihood across job categories.
Marketing teams should focus more resources on high-conversion groups such as retirees, students, and those in management roles, while considering deprioritizing low-conversion groups like blue-collar and services unless new strategies are implemented to improve outcomes.

Leveraging this feature can refine lead scoring and enable more targeted messaging. For instance, retirees may be more receptive to long-term savings products, making them a prime audience for term deposit campaigns.

<img width="551" height="458" alt="image" src="https://github.com/user-attachments/assets/b21afd57-8d44-46e5-bd31-11317f92804d" />

Observations:
- Top Occupations by Volume: Admin., blue-collar, and technician make up the bulk of the customer base.
- Most of these customers said “No,” but the large base means they still generate many total “Yes” subscriptions.

Higher Conversion Rate Segments:
- Jobs like retired, student, and management have a higher proportion of “Yes” outcomes relative to their group size.
- Retired customers in particular have a visibly higher rate of subscriptions despite smaller total counts.

Lower Conversion Rate Segments:
- Blue-collar and services show low relative “Yes” counts despite high call volumes — indicating potentially poor ROI for campaign targeting.

## Marital Status Impact:

Marital status appears to influence term deposit subscription likelihood, with married customers forming the largest group contacted but also showing a lower relative conversion rate compared to their volume. Single customers have a smaller base but a proportionally higher subscription rate, suggesting they may be more receptive to the offer.

Divorced customers represent the smallest segment and also have the lowest conversion rates, making them a less promising target group. These patterns indicate that marital status can be a useful segmentation variable for refining marketing strategies, allowing the team to prioritize high-potential groups such as singles while reassessing the approach for lower-performing segments like divorced customers.

<img width="558" height="431" alt="image" src="https://github.com/user-attachments/assets/4a4d4277-1a04-44cd-a337-f36c1327bda6" />

## Education Level Impact:

Education level shows a clear relationship with term deposit subscription rates, with customers holding a university degree representing both the largest group and one of the higher-converting segments.

High school graduates also contribute a notable share of subscriptions, though their conversion rate is lower than that of degree holders.
Customers with only basic education (4y, 6y, or 9y) form a moderate portion of the dataset but generally show weaker subscription rates, while those with professional courses fall in between.
The illiterate category is extremely negligeable, making it statistically less relevant. These insights suggest that education level can be a strong predictor for marketing segmentation, with higher-educated groups potentially more responsive to term deposit offers, warranting tailored communication strategies for different educational backgrounds.


<img width="558" height="484" alt="image" src="https://github.com/user-attachments/assets/edb04b73-8e39-4986-8d22-e2a99ac4a0d4" />

## Correlation Heat Map Overview:

The correlation heatmap of numeric features reveals several notable relationships that can impact modeling and feature selection.
Strong positive correlations exist between emp_var_rate, euribor3m, and nr_employed (all above 0.90), indicating that these economic indicators tend to move together and may contribute redundant information if all are included in a model. Similarly, emp_var_rate and cons_price_idx also show a high correlation (0.78). On the negative side, previous is moderately inversely correlated with pdays (-0.59) and emp_var_rate (-0.42), suggesting that customers contacted previously tend to have fewer days since their last contact and lower employment variation rates. Most other feature pairs have low correlations, implying they provide unique information.

<img width="690" height="615" alt="image" src="https://github.com/user-attachments/assets/64ca9629-f0be-4df6-b55d-a28592967117" />


# Modeling:

## K-Nearest Neighbor(KNN): 

KNN is useful here because it makes predictions based on similar past customers, capturing complex, non-linear patterns without strong assumptions about the data. Using StratifiedKFold ensures balanced class ratios across folds, giving a fair AUC-based evaluation despite the dataset’s imbalance.

These results show that the KNN model, tuned via grid search, performed best with 21 neighbors, Manhattan distance (p=1), and uniform weights.
Performance breakdown:
- Accuracy (0.9004) looks high, but in this imbalanced dataset it’s misleading — predicting “No” most of the time can inflate accuracy.
- Precision (0.6578) means that when the model predicts “Yes,” it’s correct about 66% of the time, which is decent.
- Recall (0.2411) is low, meaning it’s missing most of the actual “Yes” customers (high false negatives: 1,174 cases).
- F1 score (0.3529) reflects that imbalance between precision and recall, indicating limited ability to capture the positive class.
- Confusion matrix confirms the low recall (many actual subscribers were predicted as non-subscribers).
- Prediction time (~54 seconds) is quite high for KNN because it must compare each test sample to all training samples, which is costly in large datasets.

In this case, KNN finds some high-quality positive predictions (good precision) but misses the majority of true positives (poor recall) and is computationally expensive for prediction. For a marketing campaign where missing potential customers is costly, this low recall could be a major drawback, making other models (like logistic regression or decision trees) potentially more practical despite KNN’s decent precision.

<img width="490" height="472" alt="image" src="https://github.com/user-attachments/assets/47bd37b3-b54d-494d-aa17-d84dc7ddbcbe" />

This KNN confusion matrix reinforces the earlier takeaway:
- True Negatives (11,989) dominate, showing the model is excellent at predicting “No” correctly.
- False Negatives (1,174) are high, meaning many actual “Yes” cases slip through as “No,” which hurts recall.
- True Positives (373) are relatively few, but the low False Positives (194) contribute to the strong precision.

In a marketing context, this means KNN would send relatively few false leads to sales, but it would also fail to identify a large portion of potential subscribers. This makes it a low-risk, low-reward strategy—safe but not aggressive in finding new customers.


## Logistic Regression (LR): 

Logistic Regression (LR) performed moderately well on this dataset, striking a balance between identifying subscribers (“Yes”) and avoiding excessive false positives.

From the metrics:
- Accuracy (83.20%) is decent, but because the target variable is imbalanced (most customers say “No”), this alone isn’t enough to judge performance.
- Recall (64.45%) is a strong point. LR captures a significant proportion of actual subscribers, which is valuable for marketing because missing a potential subscriber is costly.
- Precision (36.22%) is relatively low, meaning many predicted “Yes” customers actually won’t subscribe, potentially increasing wasted calls.
- F1-score (0.4637) shows the trade-off between precision and recall, indicating balanced but not perfect performance.
- Prediction speed (0.138s) is extremely fast, making LR highly scalable for large datasets.

<img width="475" height="490" alt="image" src="https://github.com/user-attachments/assets/48ca4fcb-8b38-4b20-8011-144fa0085f58" />

From the confusion matrix:
- LR correctly identifies 85.59% of “No” responses, minimizing wasted resources, but 14.41% are false positives.
- For “Yes” responses, it captures 64.45% correctly but misses 35.55%, which are false negatives — lost opportunities.

<img width="523" height="490" alt="image" src="https://github.com/user-attachments/assets/8749df1b-f5d6-47d2-85d8-69463c8096e1" />

Logistic Regression is effective here due to its speed, interpretability, and solid recall, which aligns with the marketing goal of maximizing true subscriber detection. However, the low precision suggests the model should be fine-tuned or combined with other methods to reduce false positives and improve targeting efficiency.

## Decision Tree:

<img width="1880" height="966" alt="image" src="https://github.com/user-attachments/assets/565dfbf3-d5f2-482d-b938-5d4d7cd934ad" />


This Decision Tree visualization shows the top decision-making rules for predicting whether a customer will subscribe (“Yes”) or not (“No”).

Key interpretations:
1. Root Node (nr_employed)
  - The first split is based on nr_employed (number of employees in the economy, a proxy for economic conditions).
  - If nr_employed <= -1.093 (normalized value), customers are more likely to subscribe (“Yes”).
  - If higher, they tend toward “No.”
2. Left Branch (More likely “Yes”)
  - Within this group, pdays (days since last contact) is a major factor.
  - Shorter gaps since last contact (pdays <= -2.349) lean heavily toward “Yes.”
  - Contact type also matters: those contacted via cellular and with lower cons_price_idx are more likely to subscribe.
3. Right Branch (More likely “No”)
  - Here, cons_conf_idx (consumer confidence index) is an important split — higher values lean toward “No.”
  - Further splits include cons_price_idx, month_oct, and euribor3m (interest rates).
      - Higher interest rates and certain months like October correspond more with “No” outcomes.

From a business perspective, the decision tree model uses a mix of economic indicators, contact history, and campaign details to segment customers.
Customers contacted recently, via cellular, during favorable economic conditions, and with lower interest rates have higher chances of subscribing.
This tree structure can guide marketing by prioritizing leads with these characteristics, reducing wasted calls.

Best Params:
  - max_depth=5 → The tree is moderately deep, preventing overfitting while capturing key patterns.
  - min_samples_leaf=20 → Requires at least 20 samples per leaf, improving generalization.
Performance:
  - AUC = 0.7898 → Good discriminatory power, though not as strong as some more complex models might achieve.
  - Accuracy = 84.16% → Overall correct predictions are solid, but accuracy alone can be misleading given the class imbalance.
  - Precision = 37.85% → Of all predicted “Yes,” only about 38% were actual subscribers.
  - Recall = 63.22% → Captures a good portion of actual subscribers — much better than random guessing.
  - F1 Score = 0.4735 → Balanced trade-off between precision and recall, leaning toward recall.
  - Confusion Matrix Insight:
      - TP=978: True subscribers correctly identified.
      - FP=1606: Many “No” customers predicted as “Yes,” which can lead to wasted marketing calls.
      - TN=10577: Strong at identifying non-subscribers.
      - FN=569: Missed subscribers — smaller than FP count, showing recall emphasis.
  - Speed: Prediction Time = 0.118s → Extremely fast, making it viable for real-time scoring

<img width="475" height="490" alt="image" src="https://github.com/user-attachments/assets/378c6292-239c-4be3-b1fe-03fcd243a5e9" /> <img width="499" height="490" alt="image" src="https://github.com/user-attachments/assets/8a570b0d-add2-43c7-a73b-a84f882798ed" />


This Decision Tree strikes a good balance between complexity and generalization, offering high recall (good for marketing outreach) but at the cost of precision (more false positives). It’s interpretable, quick to run, and a practical choice for targeting more potential subscribers, though follow-up strategies to filter false positives would improve efficiency.

## SVM:

The SVM (RBF) model with C = 1 and gamma = 0.01 is showing:
- AUC = 0.7905 → Decent ability to rank positive cases above negatives, slightly better than random but not elite-level for marketing targeting.
- Accuracy = 83.74% → High overall correctness, but this is inflated by the dataset’s class imbalance (majority "no" class).
- Precision = 37.07% → Of those predicted as "yes", only ~37% actually subscribed — meaning many false positives.
- Recall = 63.54% → It catches almost two-thirds of true subscribers, which is good for marketing when missing a potential customer is costly.
- F1 = 0.4682 → Balanced performance between precision and recall, but still leaves room for improvement.
- Confusion Matrix → 983 true positives, 1,669 false positives, 10,514 true negatives, and 564 false negatives — so more than half of "yes" calls are correct, but there’s still a sizable cost in wasted calls.
- Tuning Time = ~160 seconds → Very fast for SVM, given reduced grid search.
- Prediction Time = ~49 seconds → Relatively heavy for predictions compared to Logistic Regression or Decision Trees, due to SVM’s computational complexity.

<img width="475" height="490" alt="image" src="https://github.com/user-attachments/assets/ea915296-430a-4eba-9eba-3433e17f355b" /> <img width="487" height="490" alt="image" src="https://github.com/user-attachments/assets/8949c7f8-9b8d-4a63-a7a3-ee2572391596" />

The SVM (RBF) confusion matrices show that the model is fairly balanced in detecting both classes, though it performs better at predicting "No" than "Yes."
  - True Negatives (TN): 10,514 cases correctly identified as "No" (86.3% of all actual No’s).
  - False Positives (FP): 1,669 cases incorrectly flagged as "Yes."
  - True Positives (TP): 983 correct "Yes" predictions (63.54% recall for Yes class).
  - False Negatives (FN): 564 missed "Yes" cases.

The SVM model is strong at ruling out non-subscribers but misses around 36% of potential subscribers, which could mean lost marketing opportunities. However, it’s more balanced than some other models (like KNN) and keeps a reasonable false positive rate.
This model balances recall and accuracy well, making it useful if the priority is to capture as many potential subscribers as possible, even at the expense of calling some non-interested customers. However, its slower prediction speed could be a drawback in real-time or high-volume environments. 

### PS: It is also worth mentioning that execution speed was boosted by tuning on a smaller dataset with fewer parameter combinations and CV folds, then retraining the best model on the full data once.The initial execution took longer than 32 minutes.


# Model Comparision Table:
| model              | best\_params                                                            | AUC        | Accuracy   | Precision  | Recall     | F1         | TP    | FP    | TN      | FN     | Tuning Time (s) | Prediction Time (s) | Tune n |
| ------------------ | ----------------------------------------------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----- | ----- | ------- | ------ | --------------- | ------------------- | ------ |
| LogisticRegression | {'clf\_\_C': 1.0}                                                       | 🟢0.802562 | 0.832921   | 0.363437   | 🟢0.642534 | 0.464269   | 🟢994 | 1741  | 10442   | 553    | 🟢1.56          | 🟢0.1329            | 15000  |
| SVM\_RBF           | {'clf\_\_C': 1, 'clf\_\_gamma': 0.01}                                   | 0.790457   | 0.837363   | 0.370664   | 0.635423   | 0.468207   | 983   | 1669  | 10514   | 564    | 🔴282.09        | 🔴48.7784           | 15000  |
| DecisionTree       | {'clf\_\_max\_depth': 5, 'clf\_\_min\_samples\_leaf': 20}               | 0.789762   | 0.841588   | 0.378483   | 0.632191   | 🟢0.473493 | 978   | 1606  | 10577   | 569    | 6.60            | 0.0901              | 15000  |
| KNN                | {'clf\_\_n\_neighbors': 21, 'clf\_\_p': 1, 'clf\_\_weights': 'uniform'} | 🔴0.779131 | 🟢0.900364 | 🟢0.657848 | 🔴0.241112 | 🔴0.352886 | 🔴373 | 🟢194 | 🟢11989 | 🔴1174 | 65.37           | 54.5701             | 15000  |

🟢 = best in column
🔴 = worst in column

In the fast mode tuning, models were optimized with a reduced cross-validation strategy (3 folds) to speed up execution while still allowing fair comparison.
- Logistic Regression achieved the highest AUC (0.803), showing the strongest ability to separate classes, with balanced recall (0.64) and low prediction time (~0.13s).
- SVM (RBF) followed closely in AUC (0.790) but had an extremely high prediction time (~48s), making it less practical for real-time use despite solid recall (0.64).
- Decision Tree matched SVM in AUC (0.790) and recall (~0.63) but was far faster (~0.09s), offering a good balance of interpretability and speed.
- KNN, while boasting the highest accuracy (0.900) and precision (0.66), had the lowest recall (0.24) — indicating it missed many positives — and suffered from the slowest prediction time (~54s), limiting scalability.

Based on this study, Logistic Regression stands out as the most practical overall, balancing predictive power, recall, and execution speed, while Decision Tree offers a good interpretable alternative. SVM and KNN provide competitive accuracy but come with heavier computational costs.

## ROC Curves Interpretation:

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/02c772e0-33f0-49bc-8744-a7d200b7556c" />

The ROC curves above compare the performance of K-Nearest Neighbors (KNN), Logistic Regression, and Decision Tree classifiers on the provided dataset. The Area Under the Curve (AUC) values quantify each model’s ability to correctly rank positive cases (subscribers) ahead of negative cases (non-subscribers).
1. Logistic Regression achieves the highest AUC (0.802), indicating the strongest ranking performance. This means it is the most effective model for prioritizing leads, ensuring that the customers most likely to subscribe are placed at the top of the contact list.
2. Decision Tree follows closely with an AUC of 0.790. While slightly less powerful in ranking ability, it offers interpretability through clear decision rules, which can be valuable for marketing strategy and stakeholder communication.
3. KNN shows the lowest AUC (0.779) among the three, suggesting less reliable lead ranking and a higher likelihood of wasted calls compared to the other models.
4. SVM has been eliminated due to the lengthy processing time and its low score in accounting for potential subscribers, making it ineffective for business.

In a real-world marketing scenario, Logistic Regression would be the preferred choice for operational targeting due to its superior ranking ability, while the Decision Tree could serve as an interpretable tool for understanding customer patterns and refining outreach strategies.

## Lift Curves (Cumulative)

<img width="691" height="547" alt="image" src="https://github.com/user-attachments/assets/330302a7-bc99-455b-91bf-55b41bb57e31" />

This cumulative lift curve compares how well each model ranks customers by likelihood to respond, with the diagonal baseline representing random targeting.
- Logistic Regression consistently outperforms other models in the early deciles, capturing the highest proportion of responders in the top 20–30% of the ranked list. This is valuable in marketing since you can focus calls on a smaller, high-probability group.
- SVM_RBF and KNN perform similarly, slightly trailing Logistic Regression but still well above the baseline, meaning they provide solid targeting efficiency.
- Decision Tree is competitive but lags slightly in the earlier deciles, which matters when you have a constrained budget and need to prioritize the top-scoring customers.

Across both the **ROC** and **cumulative lift curve** analyses, **Logistic Regression** emerges as the most consistent top performer—showing the highest AUC and the strongest early lift in capturing likely responders. SVM_RBF and Decision Tree follow closely, offering competitive ranking ability, while KNN trades early lift for higher overall accuracy. All models significantly outperform random targeting, confirming the dataset’s predictive value for focused marketing.
 
# Interpreting Logistic Regression Coefficients: 

| **Rank** | **Feature**           | **Coefficient** | **Direction** |
| -------- | --------------------- | --------------- | ------------- |
| 1        | emp\_var\_rate        | -2.1005         | Negative      |
| 2        | month\_jun            | -0.7045         | Negative      |
| 3        | month\_may            | -0.6333         | Negative      |
| 4        | month\_nov            | -0.5590         | Negative      |
| 5        | contact\_telephone    | -0.3328         | Negative      |
| 6        | poutcome\_failure     | -0.2798         | Negative      |
| 7        | pdays                 | -0.2683         | Negative      |
| 8        | default\_yes          | -0.2249         | Negative      |
| 9        | month\_apr            | -0.2123         | Negative      |
| 10       | education\_basic.4y   | -0.1990         | Negative      |
| 1        | default\_no           | 0.1585          | Positive      |
| 2        | poutcome\_success     | 0.1755          | Positive      |
| 3        | contact\_cellular     | 0.2664          | Positive      |
| 4        | job\_retired          | 0.3546          | Positive      |
| 5        | month\_aug            | 0.4124          | Positive      |
| 6        | month\_dec            | 0.4559          | Positive      |
| 7        | euribor3m             | 0.6475          | Positive      |
| 8        | education\_illiterate | 0.7173          | Positive      |
| 9        | cons\_price\_idx      | 0.9438          | Positive      |
| 10       | month\_mar            | 1.2187          | Positive      |

The strongest negative predictor is **emp_var_rate**, suggesting that higher employment variation rates sharply reduce the likelihood of subscription. Seasonal patterns are evident, with May, June, November, and April being less favorable months. Contact via telephone and past campaign failures also lower the odds. On the other hand, the strongest positive predictor is **month_mar**, followed by **high cons_price_idx**, **education_illiterate**, and **euribor3m**, indicating economic conditions and timing significantly influence subscription probability. Successful previous contacts, cellular outreach, and specific customer segments (retirees) further boost conversion likelihood.

<img width="1580" height="1580" alt="image" src="https://github.com/user-attachments/assets/c774ab84-e927-49fe-a7a2-c8bc824141d3" />



# Findings, Business Interpretation, and Next Steps

- **Primary Metric:** We use AUC as the main evaluation metric due to class imbalance and the need to rank prospects effectively in a marketing context.
- **Model Ranking:** From the results table, Logistic Regression and SVM (RBF) generally achieve the highest AUC scores when duration is excluded from features, making them strong candidates for deployment.
- **Lift Analysis:** The best model’s lift curve shows a strong concentration of positive responses in the top deciles, meaning the model can prioritize a smaller fraction of the population while capturing most of the likely subscribers.

**Actionable Guidance:**
- Target customers in the top decile predicted by the best model to maximize efficiency.
- Adjust decision thresholds based on the trade-off between call volume and conversion rate.

**Next Steps:**
- Calibrate model probabilities (Platt scaling or isotonic regression) before production use.
- Track actual campaign conversions and recalibrate quarterly.
- Apply cost-sensitive thresholding to balance call costs against expected profits.
- Explore ensemble methods like Random Forest or XGBoost for potential AUC gains.

# Conclusion:
Based on the modeling and evaluation results, Logistic Regression emerges as the most effective model for the Portuguese Bank Marketing dataset when duration is excluded, delivering the best AUC score and a strong balance between recall and precision. This makes it well-suited for marketing applications where identifying the highest-probability prospects is critical. Lift analysis confirms that the model concentrates a large share of likely subscribers within the top deciles, enabling the business to prioritize outreach efficiently and reduce wasted calls. Key drivers identified by the model, such as lower subscription likelihood during certain months (example: May, June) and higher likelihood among customers reached via cellular contact or with specific past outcomes, provide actionable insights for campaign planning. For optimal business impact, the marketing team should focus on targeting the top-decile predictions from Logistic Regression, calibrate probabilities for reliability, monitor real-world conversion rates, and adjust thresholds based on call costs versus expected gains. This approach maximizes ROI while maintaining operational efficiency.

