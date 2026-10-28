# All past papers

Every MJPRU Machine Learning (CS-401T) question we could find, typed out and tagged by unit.

::: warn Only three papers exist online
No older MJPRU CS-401T papers could be found (mjpruonline.com has none). This page has the 2025-26 end-sem, the 2025 Mid-2, and an earlier mid paper (MCQ + short answers, no date on the sheet). If you have another paper, send it in and it will be added.
:::

## End-sem 2025-26 (paper code 25251)

**B.Tech (Seventh Semester) (Main/Back) Examination 2025-26 · CS-401(T) · Computer Science & Information Technology · Machine Learning**<br>
Time: Three Hours · Maximum Marks: 70 · Attempt any five questions. Each carries equal marks, i.e. 14.

| Q | Question | Marks | Unit |
|---|---|---|---|
| 1 (a) | How machine learning differs from Artificial Learning and Deep Learning. | 7 | 1 |
| 1 (b) | Differentiate Supervised, Unsupervised, and Reinforcement learning with an example. | 7 | 1 |
| 2 | Explain the K-NN algorithm. Why it is also called Lazy learner? What are the points to be subjected when choosing the value of k? For the below problem predict for the class of Davis using KNN and assume the value of k = 3. **[numerical, table below]** | 14 | 3 |
| 3 (a) | Define class-conditional probability density function. Also, explain Bayes Decision Theory with an example. | 7 | 1 |
| 3 (b) | What is decision trees, ID3, C4.5? How are continuous attributes handled in decision trees? Explain in detail. | 7 | 3 |
| 4 (a) | Define Support Vector Machine, also explain the working of SVM in detail. Write its advantages and disadvantages. | 7 | 5 |
| 4 (b) | What are hard and soft-margin SVMs? What is the kernel trick in SVM? | 7 | 5 |
| 5 (a) | What is the back-propagation algorithm and explain its significance and how is it used in multilayer neural networks? Explain with a neat diagram. | 10 | 2 |
| 5 (b) | Explain the Widrow-Hoff (LMS) learning algorithm. | 4 | 2 |
| 6 | Write short notes on: Activation Function, Feature Selection, Dimensionality reduction, hypothesis class. | 3.5 × 4 | 2, 1, 1, 4 |
| 7 | Differentiate between: (a) Approximation error and estimation error (b) Cost function vs Loss Function (c) Bias vs Variance (d) Overfitting vs Underfitting | 3.5 × 4 | 4, 2, 4, 4 |
| 8 (a) | What do you mean by Gradient Descent? What are the conditions in which Gradient Descent is applied in terms of Linear Regression? What are the difficulties in applying Gradient Descent? | 9 | 2 |
| 8 (b) | Explain the types of Gradient Descent. | 5 | 2 |

**Data for Q2 (k-NN):**

| Candidate | Age | Income | Points | Response |
|---|---|---|---|---|
| John | 35 | 35 | 3 | Yes |
| Alex | 22 | 50 | 2 | No |
| Rio | 63 | 200 | 1 | No |
| Tim | 59 | 170 | 1 | No |
| Noa | 25 | 40 | 4 | Yes |
| Davis | 37 | 50 | 2 | ? |

::: tip Answer to Q2
Euclidean distances from Davis (37, 50, 2): John 15.17, **Alex 15.00**, Rio 152.24, Tim 122.00, Noa 15.75. The three nearest are Alex (No), John (Yes) and Noa (Yes): **2 Yes vs 1 No → Davis = Yes**. The same three neighbours are chosen after min–max normalisation, so the answer does not change. Full working in [Unit 3](#unit3).
:::

## Mid-2 2025 (second mid-semester exam)

**Department of CSIT, FET, MJPRU · IInd Mid, Machine Learning**<br>
Time: 1 hr · M.M. 10 · All questions are compulsory; each carries 2.5 marks.

| Q | Question | Unit | Came back in end-sem? |
|---|---|---|---|
| 1 | Explain the working of MLP with the help of its architecture. What are some advantages of MLP over a single perceptron? | 2 | **Yes: Q5(a) backpropagation in MLPs** |
| 2 | What are activation functions? Explain ReLU, Sigmoid, Tanh. | 2 | **Yes: Q6 short note** |
| 3 | Explain the term Confusion Matrix, Type I Error, Type-II Error. | 1 | No |
| 4 | What is Gradient Descent? Explain its working with a formula. | 2 | **Yes: Q8, a full 14 marks** |

## Earlier mid paper (MCQ + short answers)

**Machine Learning · Attempt all questions** (no department header or date on the sheet; the numbering skips Q5).

| Q | Question | Answer / unit | Came back in end-sem? |
|---|---|---|---|
| 1 | What is perceptron? (A) A single layer feed-forward neural network with pre-processing (B) A neural network that contains feedback (C) A double layer auto-associative neural network (D) An auto-associative neural network | **A** · Unit 2 | Yes: perceptron/MLP Q5 |
| 2 | Which is true for neural networks? (A) Each node computes its weighted input (B) Node could be in excited state or non-excited state (C) It has set of nodes and connections (D) All of the above | **D** · Unit 2 | — |
| 3 | Which of the following is a disadvantage of decision trees? (A) Factor analysis (B) Decision trees are robust to outliers (C) Decision trees are prone to be overfit (D) None of the above | **C** · Unit 3 | Yes: Q3(b) |
| 4 | Concept learning is basically acquiring the definition of a general category from given sample positive and negative training examples of the category. (A) TRUE (B) FALSE | **TRUE** · Unit 1/4 | — |
| 6 | Define the working of SVM Classifier in detail. | Unit 5 | **Yes: Q4(a)** |
| 7 | (i) Overfitting vs Underfitting (ii) Classification vs Prediction | Units 4, 1 | **Yes: Q7(d)** |

## Syllabus (CS-401T)

| Unit | Syllabus |
|---|---|
| 1 | **Overview and Introduction to Bayes Decision Theory:** Machine intelligence and applications, pattern recognition concepts, classification, regression, feature selection, supervised learning, class-conditional probability distributions, examples of classifiers, Bayes optimal classifier and error, learning classification approaches. |
| 2 | **Linear machines:** general and linear discriminants, decision regions, single layer neural network, linear separability, general gradient descent, perceptron learning algorithm, mean square criterion and Widrow–Hoff learning algorithm; multi-layer perceptrons: two-layers universal approximators, backpropagation learning, on-line, off-line error surface, important parameters. |
| 3 | **Learning decision trees:** inference model, general domains, symbolic decision trees, consistency, learning trees from training examples, entropy, mutual information, ID3 algorithm criterion, C4.5 algorithm, continuous test nodes, confidence, pruning, learning with incomplete data. **Instance-based learning:** nearest neighbour classification, k-nearest neighbour, nearest neighbour error probability. |
| 4 | **Machine learning concepts and limitations:** learning theory, formal model of the learnable, sample complexity, learning in zero-Bayes and realizable case, VC-dimension, fundamental algorithm-independent concepts, hypothesis class, target class, inductive bias, Occam's razor, empirical risk, limitations of inference machines, approximation and estimation errors, tradeoff. **Machine learning assessment and improvement:** statistical model selection, structural risk minimization, bootstrapping, bagging, boosting. |
| 5 | **Support Vector Machines:** margin of a classifier, dual perceptron algorithm, learning nonlinear hypotheses with perceptron, kernel functions, implicit non-linear feature space, theory, zero-Bayes, realizable infinite hypothesis class, finite covering, margin-based bounds on risk, maximal margin classifier. |

**Books (standard for this syllabus):** Duda, Hart and Stork, *Pattern Classification*; Tom Mitchell, *Machine Learning*; Shalev-Shwartz and Ben-David, *Understanding Machine Learning*; Bishop, *Pattern Recognition and Machine Learning*.
