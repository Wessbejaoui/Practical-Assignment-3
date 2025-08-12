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

## Resource Optimization:
Sales teams have limited capacity. Predicting high-likelihood leads ensures efficient resource allocation.
## Customer Experience:
Minimizing unnecessary calls improves customer perception and reduces attrition risk.
## Regulatory Compliance:
Many regions limit unsolicited marketing, so targeting only likely customers is not just profitable but also compliant.
## Why Pre-Contact Models?
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
