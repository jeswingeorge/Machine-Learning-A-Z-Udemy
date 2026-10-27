# Adaboost

**AdaBoost** (short for **Adaptive Boosting**) is a popular Ensemble Learning algorithm that combines multiple weak models to create one highly accurate strong model.

Imagine a group of students preparing for an exam. Student 1 takes a practice test and gets a few questions wrong. Student 2 focuses specifically on studying the questions Student 1 got wrong. Student 3 then focuses on the questions both Student 1 and Student 2 missed. When they take the final exam together, their combined focus covers all weak spots. That is how AdaBoost works.

---

### Core Concept Anatomy: Step-by-Step

1. **Weak Learners (Decision Stumps):**
* AdaBoost typically uses single-split Decision Trees called **Decision Stumps** (trees with a depth of 1) as base models. A single stump is barely better than random guessing.


2. **Assigning Sample Weights:**
* Initially, every data point starts with equal weight ($\frac{1}{N}$).


3. **Sequential Training & Re-weighting:**
* A stump is trained on the data.
* Samples that the stump **misclassifies are given higher weights**, while correctly classified samples get lower weights.
* The next stump focuses heavily on the newly emphasized misclassified points.


4. **Amount of Say (Voting Power):**
* Stumps with lower error rates get a higher **Amount of Say** (higher voting weight) in the final prediction, while high-error stumps get very little say.


5. **Final Aggregation:**
* Predictions from all stumps are combined using a weighted majority vote (for classification) or weighted sum (for regression).



---

### AdaBoost vs. Other Ensemble Methods

| Algorithm | Core Idea | Base Learner Type | Sequential or Parallel? | Handling of Errors |
| --- | --- | --- | --- | --- |
| **AdaBoost** | Increases weights of misclassified samples in iterations | Weak learners (e.g., Decision Stumps) | **Sequential** | Focuses explicitly on misclassified points |
| **Random Forest** | Builds independent trees on random subsamples (Bagging) | Deep, fully grown Decision Trees | **Parallel** | Reduces variance by averaging tree votes |
| **Gradient Boosting** | Fits new trees to the **residuals** (errors) of previous trees | Shallow Decision Trees | **Sequential** | Minimizes a loss function using gradient descent |

---

### Practical Use Cases & Trade-offs

* **Best Used For:** Binary classification tasks, customer churn prediction, fraud detection, and face detection (e.g., Viola-Jones algorithm).
* **Key Limitations:** Sensitive to **noisy data and outliers** because it keeps increasing the weight of misclassified noisy points, which can lead to overfitting on noise.

---

### Common Interview Scenarios

#### Scenario 1: Why is AdaBoost vulnerable to noisy data and outliers?

**Answer:** AdaBoost exponentially increases the sample weight of data points that are misclassified at each step. If a dataset has outliers or noisy labels, AdaBoost repeatedly fails to classify them correctly and focuses heavily on fitting those single outlier points, corrupting the overall decision boundary and leading to overfitting.

#### Scenario 2: How does AdaBoost differ from Gradient Boosting (GBM)?

**Answer:** While both build models sequentially, **AdaBoost adjusts sample weights** after each iteration to focus on missed points, whereas **Gradient Boosting fits new models directly to the pseudo-residuals** (the negative gradients of the loss function) of the previous models.

## [AdaBoost, Clearly Explained](https://youtu.be/LsK-xG1cLYA?si=OxcGSX2q0Ke_zzV8)

![](images/ab1.png)

**Stumps** are not good at making accurate classifications.

![](images/ab2.png)

But a stump can only use one variable to make a decision.

![](images/ab3.png)

![](images/ab4.png)

![](images/ab5.png)

![](images/ab6.png)

### Creating Forest of Stumps using Adaboost

![](images/ab7.png)

After we make the 1st stump, these weights will change in order to guide how the next stump is created.

This will be done by finding the variable - **Chest Pain**, **Blocked Arteris** or **Patient Weight** that does the best job in classifying the samples.

![](images/ab8.png)

The Gini Index for the **Patient Weight** is lowest. So it will be the 1st stump in the forest. Now we need to determine how much say this stump will have in the final classification.

> We determine how much say a stump has in the final classification based on how well it classified the samples.







