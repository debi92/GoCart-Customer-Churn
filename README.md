## Ecommerce Customer Churn Analysis and Prediction Model

This is a machine learning model to predict if a customer would likely to leave or stay.  

Customer churn, also known as customer attrition, refers to the rate at which customers stop doing business with a company. It's a crucial metric, especially for subscription-based businesses and e-commerce, as it indicates how well a company retains its customers and can signal underlying issues within products, services, or customer experience. 

High churn rates can negatively impact revenue and growth, making it essential for businesses to understand and minimize customer churn. That's why creating a model machine learning to predict customer behaviour is important. After we uncover the patterns of customer behaviour, we could provide several actionable insights to help the company retain the customers. 


### **Business Problem Understanding**
Dataset https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/data

| Variable                      | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| CustomerID                    | Unique customer ID                                                         |
| Churn                         | Churn Flag                                                                 |
| Tenure                        | Tenure of customer in organization                                         |
| PreferredLoginDevice          | Preferred login device of customer                                         |
| CityTier                      | City tier                                                                  |
| WarehouseToHome               | Distance in between warehouse to home of customer                          |
| PreferredPaymentMode          | Preferred payment mode                                                     |
| Gender                        | Gender of customer                                                         |
| HourSpendOnApp                | Number of hours spent on mobile application or website                     |
| NumberOfDeviceRegistered      | Total number of devices registered for a particular customer               |
| PreferedOrderCat              | Preferred order category of customer in last month                         |
| SatisfactionScore             | Satisfactory score of customer on service                                  |
| MaritalStatus                 | Marital status of customer                                                 |
| NumberOfAddress               | Total number of addresses added for a particular customer                 |
| Complain                      | Any complaint has been raised in last month                                |
| OrderAmountHikeFromlastYear   | Percentage increase in order from last year                                |
| CouponUsed                    | Total number of coupons used in last month                                 |
| OrderCount                    | Total number of orders placed in last month                                |
| DaySinceLastOrder             | Days since last order by customer                                          |
| CashbackAmount                | Average cashback in last month                                             |



**Context**

eCommerce churn is the term used to describe the number of customers who stop buying from your online store. The churn rate is the percentage of customers who stop doing business with you over a predetermined period but in this dataset there is no information how the costumer can be called churn, so we do initial exploration before cleaning to analyze this dataset and define what is churn costumer based on this dataset. 
Churned customers are those who:
Churned customers are those who:
- high order amount in a short time 
- Received >$100 cashback per order OR used >3 coupons
- low DaySinceLastOrder (Quick exit)
- low tenure (new user)

There are two types of customer churn that can directly affect the bottom line of an eCommerce business: voluntary churn and involuntary churn. 
- Voluntary churn occurs when a customer actively decides to stop buying from you. This could be due to dissatisfaction with the service
- Involuntary churn occurs when something like a payment failure causes a customer’s purchase to fall through. Voluntary and involuntary churn are measured together to determine your churn rate. 

**Problem Statement**

Marketing Team finds that Customer Churn is an obstacle to GO-CART growth as the major cause has not been specified. 16,8% of 5630 the costumers were churned. Churn represents the number of customers lost during a specific period (30 days), typically measured as a percentage. 

**Key Stakeholders**
1. Customer Service team = to know the connection between costumer complain and satisfaction with customer churn and get recommendation based on data on how to improve costumer experience
2. Sales & Marketing team = to improve sales and get recommendation based on data how to create marketing campaign to retain customers.
3. Product Development team = to know how to improve the product list so the customers will retain

**Cause of Churn**
1. Unsatisfied Customer/ Bad Customer Experience

A 2020 Zendesk report found that 50% of customers would switch to a competitor after just one bad customer service experience. After multiple bad experiences, this figure can rise to 80%. 

According to the Zendesk Customer Experience Trends Report 2023, 73 percent of consumers will switch to a competitor after multiple bad experiences. This can involve bad customer service, unmet expectations, or a lack of personalization—all resulting in customer churn.

2. Attracting wrong customer

It’s surprising how often customers sign up for a product without fully understanding whether that product meets their needs. As soon as they find out your product isn’t a great fit, it’s highly likely they’ll leave you behind for a competitor.

As per our dataset, we can find out that we may have attracted wrong customers (the customers who may be promo/cashback hunters)

3. Better competitor

A major reason for customer churn is when competitors :
A. Give Competitive Offering:
If a competitor provides a product or service that is perceived to be superior, whether in terms of features, quality, or price, customers are likely to switch. 
B. Give Better Customer Experience:
C. Give Better Pricing:
D. Has More Innovations:
Customers may also leave if they feel a company isn't innovating or improving its product to keep up with market trends and customer demands. 

 
**Benefits of lower Churn level**
1. Lower Acquisition Costs

It’s far cheaper to retain customers than to acquire new ones. This is especially true in subscription-based eCommerce businesses, where customers contribute recurring revenue month after month. It costs up to six to seven times more to gain a new customer than to retain a current one. Knowing your churn rate could help you determine what you need to do to hold on to more customers. This, in turn, means that you won’t have to invest as much time and money into new customer acquisition. 
2. Determine Operational Shortfalls

A high churn rate is indicative of problems. Something is driving these people away, and by understanding that a problem exists, you can figure out what it is and stop it, thus keeping people from churning. Some common issues uncovered by a high churn rate include: 
- Bad customer service
- A lack of affordability
- Poor marketing
- A big marketing push from competitors
- A poorly designed website
3. Increase profit

Improving your retention by just 5% can increase your profits anywhere from 25% to 95%. On top of that, it can reduce the probability of re-selling to an existing customer to somewhere between 60% and 70%. To compare, the odds of selling to a new customer are only 5% to 20%. 

**Benefits of Analysis & Create ML Model**
1. Proactive Customer Retention

Early Warning System: AI identifies at-risk customers before they churn by analyzing behavioral patterns (e.g., reduced purchase frequency, increased coupon usage).
Example: Flagging customers who haven’t purchased in 30 days but previously bought weekly.
Hyper-Targeted Interventions: Enables personalized retention strategies (e.g., customized discounts for high-value at-risk customers).

2. Revenue Protection

Cost Efficiency: Retaining customers is cheaper than acquiring new ones. AI optimizes retention spend by focusing on high-probability churners.
Lifetime Value (LTV) Preservation: A 5% reduction in churn can increase profits by 25–95% (Harvard Business Review).

3. Data-Driven Decision Making

Root Cause Analysis: AI pinpoints why customers leave (e.g., poor delivery experience, price sensitivity) by analyzing:


**Goals**

Generating ML models to predict whether consumers will churn or not so the marketing team could anticipate and create a feasible strategy to avoid costumer churn while also increase consumer satisfaction to decrease the costumers' complains.

**Analytic Approach**
Churn Prediction (into 3 category: High risk(>70%), medium risk (40-70%) and low risk churn (<40%)) to give the right treatment to retain costumer. 

**Metric Evaluation**
| Metric               | Why It Matters                                                                                     | Formula                                  |
|----------------------|---------------------------------------------------------------------------------------------------|------------------------------------------|
| Accuracy             | Measures overall correctness, but can be misleading due to class imbalance (16.8% churn rate).    | (TP+TN)/(TP+TN+FP+FN)                   |
| Precision            | Tells you how many flagged "churners" are real churners. Crucial when intervention costs are high. | TP/(TP+FP)                              |
| Recall               | Identifies what percentage of actual churners you catch. Critical because losing a customer is costlier. | TP/(TP+FN)                             |
| F1 Score             | Balances precision and recall equally. Good general-purpose metric when both FP and FN matter.    | 2*(Precision*Recall)/(Precision+Recall) |
| F2 Score             | Prioritizes recall over precision (β=2), since missing a churner (FN) is typically 5-10x costlier. | (1+2²)*(Precision*Recall)/(2²*Precision+Recall) |
| ROC-AUC              | Measures overall ranking ability (how well the model separates churners from non-churners). Robust to imbalance. | Area under ROC curve                   |
| PR-AUC (Precision-Recall AUC) | Better than ROC when classes are imbalanced. Focuses on positive (churn) class performance. | Area under PR curve                    |

**Project Scope**
This project goals is to develop a customer churn prediction model using a supervised machine learning. The project scope includes:

- Analyze and find the pattern of customer behavior and transaction data.
- Finding the most important features that caused customer churn.
- Make a churn prediction model.
- Providing recommendations based on the data and from the model output.

**Project Limitation**
1. Churn definition limitation = It is not clearly stated how churn is defined 
2. Data Limitations:
  * No timestamp or time-series data available, limiting temporal analysis.
  * No total amount transaction
  * Class Imbalance (16.8% churn rate)
  * Missing values
  * Self reported data such as Costumer Satisfaction
3. Model limitation: can not explain the real cause of costumer churn, can't predict the new costumer with no history

**Data Preprocessing**
The dataset has 5630 rows and 20 columns and contains:
Churn: 0 No, 1 Yes

Numerical Columns:
Tenure, Warehouse To Home, Hour Spend On App, Order Amount From last Year, Coupon Used, Order Count, Day Since Last Order, Cashback Amount

Ordinal Columns:
CityTier, Satisfaction Score, Complain

Nominal Columns:
Gender, Marital Status, Preferred Login Device, Preferred Payment Mode, Preferred Order Category

From the initial data exploration we found out that the most churned costumers are the costumer with <7 DaySinceLastOrder so we couldn't solely use DaySinceLastOrder to define costumer churn
Other findings:
1. Cashback Amount Patterns

Churned customers (1) show significantly higher cashback amounts than retained customers (0).
The company may be attracting "cashback hunters" who churn after claiming benefits
Other Action: Investigate if cashback offers are being exploited (e.g., customers only used the application for a short time just to get cashback)

2. Days Since Last Order Patterns

Churned customers have SHORTER time since last order (more recent activity) than retained customers. There are several possibilities:
- recent purchasers might be dissatisfied and actively choosing to leave
- Could indicate transactional (not relational) customers

3. Order Count Behavior

Churned customers have fewer orders (left side of distribution) than retained customers which most churn happens early in customer lifecycle (after 1-2 orders). Supports our hypothesis about "promo and coupon abusers" who make minimal purchases
4. High Churn Concentration Areas:
- Low OrderCount (2-4 orders) + High CashbackAmount (>$150)
- Customers who get >$100 cashback within their first 5 orders have higher churn rate
- The sweet spot for retention appears to be <$100 cashback per order
  
5. Almost no churned costumers (loyal costumer): 
- High OrderCount (10-16) + Low CashbackAmount ($0-$50)

6. <9m cohort (tenure)  shows highest churn
Immediate post-purchase risk: Even customers who made multiple purchases in their first year show elevated churn, suggesting initial experience issues

7. The highest churn rates typically appear among high order  amount below 9 month

8. Customers who survive beyond9 month show significantly lower churn rates
9m+ stability: The most established cohort (9m+) maintains the lowest churn rates regardless of order count

9. Unexpected High-Value Churn
Multi-order churn in early cohorts (<3m): Some customers making 2+ orders within their first month still churn at notable rates


The data suggests a high possibility of "hit-and-run" customer segment that:
- Makes few orders (low OrderCount)
- Extracts maximum value (high CashbackAmount and high UsedCoupons)

**Missing Value**
Missing value with normal distribution : Tenure, HourSpendOnApp, OrderAmountHikeFromLastYear
Right skewed missing value (skewness >1) = WarehouseToHome, CouponUsed, OrderCount, DaySinceLastOrder

Missing value handling:
- for normal distribution ==> imputation using mean
- for WarehouseToHome & OrderCount ==> imputation using median
- for CouponUsed ==> imputation with 0 (maybe the null means 0 coupon used)
- for DaySinceLastOrder ==> imputation with 0 (maybe the null means 0 day since last order)

**Categorical Value**
Unique elements of PreferredLoginDevice are: 
['Mobile Phone' 'Phone' 'Computer']

Unique elements of PreferredPaymentMode are: 
['Debit Card' 'UPI' 'CC' 'Cash on Delivery' 'E wallet' 'COD' 'Credit Card']

Unique elements of Gender are: 
['Female' 'Male']

Unique elements of PreferedOrderCat are: 
['Laptop & Accessory' 'Mobile' 'Mobile Phone' 'Others' 'Fashion' 'Grocery']

Unique elements of MaritalStatus are: 
['Single' 'Divorced' 'Married']

some value have the same meaning:

    1. PreferredLoginDevice : phone and mobile phone

    2. PreferredPaymentMode : CC and Credit Card ; COD and Cash on Delivery

    3. PreferedOrderCat: Mobile and Mobile Phone

So the same meaning value should be standardize using one of the term
- phone ==> mobile phone
- cc ==> credit card
- COD ==> Cash on Delivery
- mobile ==> mobile phone   

**Value that lost because of costumer churn**
Average cashback amount lost per churned customer: $160.37
Average transaction amount lost per churned customer: $1603.69
Total cashback lost from all churned customers: $152030.00
Total transaction amount lost all churned customers: $1520300.00

**Outlier Handling**
There are some outliers in: 
- tenure (we will create a new flag for it)
- WarehouseToHome (max=127, 75th%=20) ==> cap to 50 km
- NumberOfAddress (max=22, 75th%=6) ==> cap to 10
- CouponUsed (we will create a new flag for it)
- DaySinceLastOrder (we will create a new flag for it)
- CashbackAmount (we will create a new flag for it)

**New Variable**
Churn rate when HighCashback_LowOrders = 1: 17.38%
Churn rate when CouponHunter = 1: 17.17%
Churn rate when QuickChurner = 1: 46.24%
Churn rate when HighSpend_New = 1: 36.71%

Critical Churn Risks (Priority for Retention Strategies)

| Feature          | Churn Rate | Insight                                                                 |
|------------------|------------|-------------------------------------------------------------------------|
| QuickChurner     | 46.24%     | Customers who placed an order <7 days ago but have <3 months tenure are ~2.75x more likely to churn than average (16.8%). These are "hit-and-run" users who try the service but leave quickly. |
| HighSpend_New    | 36.71%     | New customers (<6 months) with >20% order growth churn at ~2.2x the average rate. Could be fraud, promo exploiters, or dissatisfied high-value users. |
Action:
1. QuickChurners:
- Trigger win-back campaigns (e.g., personalized discounts) within 3 days of their first order.
- Improve onboarding experience (e.g., tutorial emails, customer support check-ins).

QuickChurners likely experienced:
- Poor first impression (e.g., slow delivery, bad UX).
- Mismatched expectations (e.g., product quality didn’t match ads).

2. HighSpend_New:
- Investigate if these are fraudulent transactions (e.g., bulk purchases for resale).
- Offer loyalty perks (e.g., VIP support) to retain genuine high-spenders.

HighSpend_New customers might:
- Be resellers buying in bulk during promotions.
- Have buyer’s remorse after large purchases.

Moderate Churn Risks (Monitor & Optimize)

| Feature                  | Churn Rate | Insight                                                                 |
|--------------------------|------------|-------------------------------------------------------------------------|
| HighCashback_LowOrders   | 17.38%     | Slightly above average churn. These users may be cashback-focused but not fully engaged. |
| CouponHunter             | 17.17%     | Similar to average. Suggests coupon use alone doesn’t drive churn, but combined with other factors (e.g., low tenure) it might. |

Action:
1. Test gradual cashback rewards (e.g., increase rewards for repeat purchases) to incentivize loyalty.
2. For coupon hunters, limit high-value coupons to customers with >3 orders to filter out exploiters.

**EDA**
- 16.8% of customers ar churning. This is more than the standard churn rate, around 10%, and the higher the churnrate, the higher loss of revenue for losing more customers.
- There are quite few features that affected customer churn that we have to concern about :
1. QuickChurner and Tenure : these two features are similar and giving same result as they are both at the time domain. So we can count them into one analysis to provide the churn and time correlation
2. Complain : A 2020 Zendesk report found that 50% of customers would switch to a competitor after just one bad customer service experience. After multiple bad experiences, this figure can rise to 80%. This also corelate with satisfaction score. This is one of the main features to be concerned about.
3. Cashbackamount : The higher the cashbackamount the lower the churnrate. Customers seems really into discounts, and cashbacks are really one of them. While they are having greater cashback, they tend to retain. Cashback could be one of the main solution, while we have to get another solutions to other problematic features.
4. Daysince last order : Customer higher days since last order are tend to retain. We have capped days since last order so it gives us clue that the higher days since last order is a customer that using the app periodicly.
5. Numberofdevice : the amount of device doesn't gives us faith that they will retain. This feature could be used for more analysis at the model.
6. CityTier, Warehousetohome, Numberofadress : Those three features are similar and could be count as a one analysis for customers place to deliver product
- The majority of churned customer are on the first two tenures. New customers with tenure 0-1 month are more likely to churn, and could be called more loyal customers when they retain at second to thirty-month tenure. With the higher count of complaints on the first two tenures shows that new customers are more likely to have a bad experience at the beginning of their journey. There must be new “Welcome” treatment to the new customers as they might be increasing when Marketing Team set new programs to retain customers and increase order counts.
- With higher order count at the beginning of customer tenure, Customer Service need to maintain their relationship with the customer to make sure they have their best experience. Complain flag indicates bad experience needs to be service within 24 and needs to be clear before 48 hours. That will retain customers more effectively.

**Machine Learning**
A. Logistic Regression

Type: Supervised learning (Classification).
Objective: Predicts the probability of a binary (or multi-class) outcome using a logistic function (sigmoid).
How it works:
- Models the log-odds of the probability as a linear combination of input features.
- Uses maximum likelihood estimation to optimize coefficients.

Pros:
- Simple, interpretable (coefficients show feature importance).
- Efficient for linearly separable data.

Cons:
- Assumes linearity between features and log-odds.
- Poor performance on non-linear relationships.

In this dataset Logistic Regression
Best for:
- Interpretability: Shows how each feature impacts churn (coefficients).
- Baseline Model: Simple to implement and sets a performance benchmark.
- Probabilistic Output: Directly predicts churn probability (0-1).

Limitations:
- may underfit complex patterns
- Requires careful handling of imbalanced data (churn rate = 16.8%).

B. K-Nearest Neighbors (KNN)
Type: Supervised learning (Classification/Regression).
Objective: Predicts based on the majority vote (or average) of the K closest training examples.
How it works:
- No explicit training; stores all training data.
- Uses distance metrics (e.g., Euclidean) to find neighbors.

Pros:
- Simple, no assumptions about data distribution.
- Works well for small datasets with clear local patterns.

Cons:
- Computationally expensive (lazy learner).
- Sensitive to irrelevant features and imbalanced data.

C.Random Forest
Type: Ensemble learning (Bagging) for Classification/Regression.
Objective: Combines multiple decision trees to reduce overfitting.
How it works:
- Trains trees on random subsets of data (bootstrap samples) and features.
- Aggregates predictions via voting (classification) or averaging (regression).

Pros:
- Handles non-linearity and high-dimensional data well.
- Robust to outliers and overfitting.

Cons:
- Less interpretable than single trees.
- Slower than linear models for large datasets.

D. XGBoost (Extreme Gradient Boosting)
Type: Ensemble learning (Boosting) for Classification/Regression.
Objective: Sequentially builds trees to correct errors of previous trees.
How it works:
- Optimizes a loss function using gradient descent.
- Adds trees iteratively, weighting misclassified data more heavily.

Pros:
- High accuracy, handles missing data.
- Regularization prevents overfitting.

Cons:
- Hyperparameter tuning can be complex.
- More prone to overfitting than Random Forest if not tuned properly.

In this dataset XGBoost
Best for:
- Handling Imbalance: Built-in scale_pos_weight to prioritize churn class.
- Nonlinear Patterns: Captures interactions (e.g., HighCashback + LowTenure = High Risk).
- Feature Importance: Ranks drivers of churn (e.g., CashbackAmount matters most).
Limitations:
- Less interpretable than logistic regression.
- Requires hyperparameter tuning.


| Model                  | Type               | Reason                                                                 |How-It-Works                                                                 | Pros                                                                 | Cons                                                                 |
|------------------------|--------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|----------------------------------------------------------------------|----------------------------------------------------------------------|
| **Logistic Regression** | Linear classifier  | Shows exactly how factors like coupons or tenure affect churn odds (e.g., "Coupon users are 2x more likely to churn") | Draws a straight "boundary" to separate churners/non-churners, then converts to probabilities | <ul><li>Easy to explain</li><li>Fast to train</li></ul>              | <ul><li>Misses complex patterns</li><li>Struggles with imbalanced data</li></ul> |
| **KNN**                | Instance-based     | Finds small groups of similar customers who churn (e.g., "Urban 25-30yr olds who use 3+ coupons") | Looks at the 5 most similar customers in history - if 3+ churned, predicts "High Risk"         | <ul><li>No complex math needed</li><li>Good for niche segments</li></ul> | <ul><li>Slow with many customers</li><li>Needs perfect feature scaling</li></ul> |
| **Random Forest**      | Tree ensemble      | Handles messy data (outliers, missing values) while ranking churn drivers                        | Builds 100+ mini decision trees, then combines their votes (like a survey)          | <ul><li>Auto-detects key features</li><li>Hard to overfit</li></ul>   | <ul><li>Medium complexity</li><li>Cannot explain exact rules</li></ul> |
| **XGBoost**           | Advanced tree ensemble | Best accuracy for spotting high-risk combos (e.g., "Low Tenure + High Cashback = 80% churn risk") | Builds trees one-by-one, fixing errors from previous trees (like a student learning from mistakes) | <ul><li>Handles imbalance well</li><li>Top prediction power</li></ul> | <ul><li>Harder to tune</li><li>More "black box"</li></ul>             |

**Evaluation Metrics**
### Evaluation Metrics Explanation

### 1. **Accuracy**
The ratio of correct predictions to the total number of predictions.
\\[
\\{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\\]

---

### 2. **Precision**
The ability of the model to correctly identify only relevant data points (true positives).
\\[
\\{Precision} = \frac{TP}{TP + FP}
\\]


---

### 3. **Recall (Sensitivity / True Positive Rate)**
The ability of the model to find all relevant instances of the positive class.
\\[
\\{Recall} = \frac{TP}{TP + FN}
\\]

---

### 4. **F1 Score**
The harmonic mean of Precision and Recall — useful when you need a balance between the two.
\\[
\\{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\\]

---

### 5. **F2 Score**
Similar to F1 Score, but gives more weight to Recall (minimizing False Negatives).
\\[
\\text{F2 Score} = (1 + 2^2) \cdot \frac{\text{Precision} \cdot text{Recall}}{(2^2 \cdot \text{Precision}) + \text{Recall}}
\\]

---

### 6. **ROC AUC (Receiver Operating Characteristic - Area Under Curve)**
Measures the model's ability to distinguish between positive and negative classes.
- AUC values range from 0.5 (random guess) to 1.0 (perfect classifier).

---

### 7. **PR Score (Average Precision Score)**
Represents the area under the Precision-Recall curve.
- Particularly useful for imbalanced datasets.

---

### Legend:
- **TP**: True Positive  
- **TN**: True Negative  
- **FP**: False Positive  
- **FN**: False Negative
"""

| Metric               | Why It Matters                                                                                     | Formula                                  |
|----------------------|---------------------------------------------------------------------------------------------------|------------------------------------------|
| Accuracy             | Measures overall correctness, but can be misleading due to class imbalance (16.8% churn rate).    | (TP+TN)/(TP+TN+FP+FN)                   |
| Precision            | Tells you how many flagged "churners" are real churners. Crucial when intervention costs are high. | TP/(TP+FP)                              |
| Recall               | Identifies what percentage of actual churners you catch. Critical because losing a customer is costlier. | TP/(TP+FN)                             |
| F1 Score             | Balances precision and recall equally. Good general-purpose metric when both FP and FN matter.    | 2*(Precision*Recall)/(Precision+Recall) |
| F2 Score             | Prioritizes recall over precision (β=2), since missing a churner (FN) is typically 5-10x costlier. | (1+2²)*(Precision*Recall)/(2²*Precision+Recall) |
| ROC-AUC              | Measures overall ranking ability (how well the model separates churners from non-churners). Robust to imbalance. | Area under ROC curve                   |
| PR-AUC (Precision-Recall AUC) | Better than ROC when classes are imbalanced. Focuses on positive (churn) class performance. | Area under PR curve                    |

| Business Scenario            | Key Metrics            | Usage                                      |
|------------------------------|------------------------|--------------------------------------------|
| Budget-Constrained Retention | Precision, F1          | Minimize wasted spend on false positives   |
| High-Value Customers         | Recall, F2             | Ensure no VIP churners are missed          |
| Model Comparison             | ROC-AUC, PR-AUC        | Threshold-independent evaluation           |
| Intervention Optimization    | Precision-Recall Curve | Find the perfect cost/benefit threshold    |


