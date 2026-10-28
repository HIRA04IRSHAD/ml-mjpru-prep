# Last-night sheet

Everything to read the evening before the exam: the definitions to underline, the lists examiners count, the formulas, the "differentiate" pairs, and the numerical answers to check against. Every number below is copied from the unit pages and re-checked with Python.

## Definitions (underline these in the exam)

| Term | Definition |
|---|---|
| Machine learning (Mitchell) | <u>A program learns from experience E for tasks T and measure P if its performance at T, measured by P, improves with E</u> |
| AI / ML / DL | <u>AI: machines doing tasks that need human intelligence. ML: subset of AI that learns from data. DL: subset of ML using neural networks with many hidden layers</u> |
| Supervised / unsupervised / reinforcement | <u>Learn from labelled pairs (x, y) / find structure in unlabelled x / an agent learns a policy that maximises total reward</u> |
| Class-conditional density p(x\|ω<sub>j</sub>) ⭐ | <u>The probability density of the feature x given that the true class is ω<sub>j</sub></u> |
| Bayes decision theory ⭐ | <u>Statistical approach that uses priors, class-conditional densities and costs to make the decision with the least expected cost</u> |
| Bayes error | <u>Error of the Bayes optimal classifier: the lowest error possible for the given features, caused by class overlap</u> |
| Feature selection / extraction ⭐ | <u>Keep a subset of the original features / build fewer new features from them (e.g. PCA)</u> |
| Dimensionality reduction ⭐ | <u>Reducing the number of features while keeping as much useful information as possible</u> |
| Confusion matrix, Type I / II ⭐ | <u>Table of TP, FN, FP, TN. Type I = false positive, Type II = false negative</u> |
| Activation function ⭐⭐ | <u>f applied to a neuron's net input, y = f(net); decides how strongly the neuron fires and adds non-linearity</u> |
| Linear separability | <u>Two classes are linearly separable if one hyperplane w·x + b = 0 puts all of one class on each side (AND, OR yes, XOR no)</u> |
| Gradient descent ⭐⭐ | <u>Iterative method that minimises J(θ) by stepping against the gradient: θ ← θ − η∇J(θ)</u> |
| Widrow–Hoff (LMS) ⭐ | <u>Minimises the mean square error J = ½Σ(t − w·x)<sup>2</sup> one sample at a time: w ← w + η(t − y<sub>in</sub>)x</u> |
| MLP / backpropagation ⭐⭐⭐ | <u>Feed-forward net with hidden layers of non-linear units; trained by backpropagation, which sends error terms backwards with the chain rule and updates weights by gradient descent</u> |
| Decision tree / ID3 ⭐⭐ | <u>Tree whose internal nodes test attributes, branches are outcomes, leaves are classes. ID3 grows it top-down choosing the highest information gain</u> |
| Entropy | <u>Entropy(S) = −Σ p<sub>i</sub> log<sub>2</sub> p<sub>i</sub>: the impurity of a set of examples</u> |
| k-NN / lazy learner ⭐ | <u>Classify a query by majority vote of its k nearest training points. Lazy: no model is built, all work waits until a query arrives</u> |
| Hypothesis class H ⭐ | <u>The set of all hypotheses the learner may choose from, fixed before seeing data</u> |
| Approximation / estimation error ⭐ | <u>Error of the best h in H over the Bayes error (H too small) / extra error of the learned h over the best in H (finite sample)</u> |
| Bias / variance ⭐ | <u>Error from too-simple assumptions / error from sensitivity to the particular training set</u> |
| Overfitting / underfitting ⭐⭐ | <u>Train error very low but test error high / both train and test error high</u> |
| PAC-learnable | <u>With probability ≥ 1 − δ the learner outputs h with error ≤ ε, using examples and time polynomial in 1/ε, 1/δ</u> |
| VC dimension | <u>Size of the largest set of points that H can shatter</u> |
| SVM ⭐⭐ | <u>Finds the separating hyperplane w·x + b = 0 with the largest margin. The closest points are the support vectors</u> |
| Hard / soft margin ⭐ | <u>Hard: every point outside the margin, y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1. Soft: slack ξ<sub>i</sub> allowed, penalty CΣξ<sub>i</sub></u> |
| Kernel function ⭐ | <u>K(x, z) = φ(x)·φ(z), a dot product in feature space computed without computing φ</u> |

## Lists examiners count

- **Types of learning:** supervised (classification, regression), unsupervised (clustering, dimensionality reduction), semi-supervised, reinforcement (agent, environment, action, reward, policy).
- **Pattern recognition system:** sensing → segmentation → feature extraction → classification → post-processing. **Design cycle:** collect data, choose features, choose model, train, evaluate.
- **Feature selection methods:** filter, wrapper, embedded. **PCA steps:** centre, covariance, eigenvectors, sort by eigenvalue, project.
- **Bayes rule pieces:** prior P(ω), likelihood p(x|ω), evidence p(x), posterior P(ω|x). Decide the largest posterior.
- **Activation functions:** step, linear, sigmoid (0, 1), tanh (−1, 1), ReLU [0, ∞). Problem: vanishing gradient (sigmoid, tanh).
- **Conditions for GD in linear regression:** differentiable cost, convex MSE bowl, scaled features, suitable η, enough iterations.
- **Difficulties of GD:** choosing η (too small = slow, too big = diverges), local minima, saddle points, plateaus, ravines (zig-zag), feature scaling, vanishing/exploding gradients.
- **Types of GD:** batch (all m samples per update), stochastic (one sample), mini-batch (b samples, e.g. 32).
- **Backpropagation steps:** initialise small random weights → forward pass → output δ → hidden δ (old weights) → update all weights → repeat until error is small.
- **Important BP parameters:** learning rate, momentum, number of hidden units, number of epochs, weight initialisation, activation, stopping rule.
- **Advantages of MLP over perceptron:** solves XOR and non-linear problems, universal approximator, learns features in hidden layers, multi-class outputs.
- **ID3 steps:** compute entropy → gain of each attribute → highest gain is the node → branch per value → repeat → stop when pure or no attributes.
- **What C4.5 adds:** gain ratio, continuous attributes (thresholds), missing values, pruning (pessimistic, CF = 25%), rule post-pruning.
- **Choosing k:** odd k for two classes, k ≈ √n, cross-validation, small k = noisy (high variance), large k = over-smooth (high bias), scale features, choose the distance metric.
- **SVM working:** data → hyperplane → margin 2/‖w‖ → minimise ½‖w‖<sup>2</sup> → support vectors (α > 0) → kernel if not separable → classify by sign(w·x + b).
- **SVM advantages:** maximum margin, convex (one global minimum), works in high dimensions, kernels, only support vectors matter. **Disadvantages:** slow on large data, choice of kernel and C, no direct probabilities, sensitive to noise and scaling.
- **Kernels:** linear, polynomial, Gaussian/RBF, sigmoid. Valid kernel: Mercer's condition (Gram matrix positive semi-definite).
- **Unit 4 concepts:** hypothesis class, target class, inductive bias, Occam's razor, empirical risk / ERM, No Free Lunch.
- **Model selection:** hold-out (70/30), k-fold cross-validation, SRM, bootstrap. **Ensembles:** bagging (parallel, cuts variance, random forest), boosting (sequential, cuts bias, AdaBoost).

## Formulas

| Item | Formula |
|---|---|
| Bayes theorem | P(ω<sub>j</sub>\|x) = p(x\|ω<sub>j</sub>) P(ω<sub>j</sub>) / p(x), with p(x) = Σ p(x\|ω<sub>j</sub>) P(ω<sub>j</sub>) |
| Minimum-error rule, LRT | Decide ω<sub>1</sub> if p(x\|ω<sub>1</sub>)/p(x\|ω<sub>2</sub>) > P(ω<sub>2</sub>)/P(ω<sub>1</sub>) |
| Conditional risk | R(α<sub>i</sub>\|x) = Σ<sub>j</sub> λ<sub>ij</sub> P(ω<sub>j</sub>\|x); choose the smallest risk |
| Gaussian density | p(x\|ω) = (1/(√(2π) σ)) exp(−(x − μ)<sup>2</sup> / 2σ<sup>2</sup>) |
| Metrics | Accuracy = (TP + TN)/total; Precision = TP/(TP + FP); Recall = TP/(TP + FN); Specificity = TN/(TN + FP); F1 = 2PR/(P + R) |
| Regression errors | MAE = (1/n)Σ\|e\|; MSE = (1/n)Σe<sup>2</sup>; RMSE = √MSE |
| Linear discriminant | g(x) = w·x + w<sub>0</sub>; distance from boundary r = g(x)/‖w‖ |
| Sigmoid, tanh, ReLU | σ(z) = 1/(1 + e<sup>−z</sup>), σ′ = σ(1 − σ); tanh′ = 1 − tanh<sup>2</sup>; ReLU = max(0, z) |
| Perceptron rule | y = step(w·x + b); if y ≠ t: w<sub>i</sub> ← w<sub>i</sub> + η(t − y)x<sub>i</sub>, b ← b + η(t − y) |
| Gradient descent | θ ← θ − η ∂J/∂θ |
| GD for linear regression | J = (1/2m)Σ(ŷ − y)<sup>2</sup>; ∂J/∂w = (1/m)Σ(ŷ − y)x; ∂J/∂b = (1/m)Σ(ŷ − y) |
| LMS (Widrow–Hoff) | y<sub>in</sub> = w·x + b; e = t − y<sub>in</sub>; w ← w + η e x; b ← b + η e |
| Backprop (sigmoid) | δ<sub>k</sub> = (t<sub>k</sub> − o<sub>k</sub>) o<sub>k</sub>(1 − o<sub>k</sub>); δ<sub>j</sub> = h<sub>j</sub>(1 − h<sub>j</sub>) Σ δ<sub>k</sub>w<sub>jk</sub>; w<sub>jk</sub> ← w<sub>jk</sub> + η δ<sub>k</sub> h<sub>j</sub> |
| Momentum | Δw(n) = −η ∂E/∂w + α Δw(n − 1) |
| Entropy, gain | Entropy(S) = −Σ p<sub>i</sub> log<sub>2</sub> p<sub>i</sub>; Gain(S, A) = Entropy(S) − Σ (\|S<sub>v</sub>\|/\|S\|) Entropy(S<sub>v</sub>) |
| C4.5 | SplitInfo = −Σ (\|S<sub>v</sub>\|/\|S\|) log<sub>2</sub>(\|S<sub>v</sub>\|/\|S\|); GainRatio = Gain / SplitInfo |
| Distances | Euclidean d = √Σ(x<sub>i</sub> − y<sub>i</sub>)<sup>2</sup>; Manhattan d = Σ\|x<sub>i</sub> − y<sub>i</sub>\|; min–max x′ = (x − min)/(max − min) |
| Cover–Hart | R* ≤ R ≤ 2R*(1 − R*) ≤ 2R* |
| PAC (finite H, realizable) | m ≥ (1/ε)(ln\|H\| + ln(1/δ)); agnostic: m ≥ (1/(2ε<sup>2</sup>))(ln(2\|H\|) + ln(1/δ)) |
| Error split | L(ĥ) = Bayes error + approximation error + estimation error; expected error = Bias<sup>2</sup> + Variance + noise |
| Vapnik (SRM) | R(h) ≤ R<sub>emp</sub>(h) + √{[d(ln(2N/d) + 1) − ln(η/4)] / N} |
| AdaBoost | α<sub>t</sub> = ½ ln((1 − ε<sub>t</sub>)/ε<sub>t</sub>); w<sub>i</sub> ← w<sub>i</sub> e<sup>−α<sub>t</sub> y<sub>i</sub> h<sub>t</sub>(x<sub>i</sub>)</sup> / Z<sub>t</sub>; H(x) = sign(Σ α<sub>t</sub> h<sub>t</sub>(x)) |
| Bootstrap | P(example left out) = (1 − 1/m)<sup>m</sup> → 1/e = 0.368 (63.2% included) |
| SVM hard margin | minimise ½‖w‖<sup>2</sup> s.t. y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1; margin width = 2/‖w‖ |
| SVM soft margin | minimise ½‖w‖<sup>2</sup> + C Σξ<sub>i</sub> s.t. y<sub>i</sub>(w·x<sub>i</sub> + b) ≥ 1 − ξ<sub>i</sub>, ξ<sub>i</sub> ≥ 0 |
| SVM dual | w = Σ α<sub>i</sub> y<sub>i</sub> x<sub>i</sub>, Σ α<sub>i</sub> y<sub>i</sub> = 0; support vectors have α<sub>i</sub> > 0 |
| Kernels | Linear x·z; polynomial (x·z + c)<sup>d</sup>; RBF exp(−‖x − z‖<sup>2</sup>/2σ<sup>2</sup>); sigmoid tanh(κ x·z + θ) |
| Margins | Functional γ̂ = y(w·x + b); geometric γ = y(w·x + b)/‖w‖ |

## Differences to remember

The four End Q7 pairs come first (3.5 marks each: write a 5–6 row table in the exam).

| Pair | First | Second |
|---|---|---|
| **Approximation vs estimation error** ⭐ (End Q7a) | Best in H minus Bayes error; caused by a small/restricted H; falls as H grows; does not fall with more data | Learned h minus best in H; caused by a finite sample; rises as H grows; falls with more data |
| **Cost vs loss function** ⭐ (End Q7b) | Cost J: average (or sum) over the whole training set, e.g. (1/2m)Σe<sup>2</sup>; the thing GD minimises | Loss: error on one example, e.g. (ŷ − y)<sup>2</sup> |
| **Bias vs variance** ⭐ (End Q7c) | Error from too-simple assumptions; high in simple models; causes underfitting; fix: richer model, more features | Error from sensitivity to the training set; high in complex models; causes overfitting; fix: more data, regularisation, bagging |
| **Overfitting vs underfitting** ⭐⭐ (End Q7d, Mid Q7 i) | Train error low, test error high; high variance; too complex; fix: prune, regularise, more data, early stopping | Train and test error both high; high bias; too simple; fix: more complex model, more features, train longer |
| AI vs ML vs DL ⭐ (End Q1a) | AI: whole field, rules or learning | ML: learns from data, hand-made features; DL: many layers, learns features itself, needs big data and GPUs |
| Supervised vs unsupervised vs RL ⭐ (End Q1b) | Supervised: labelled data, predicts y (spam filter) | Unsupervised: no labels, finds groups (customer clusters); RL: reward signal, learns a policy (game playing) |
| Classification vs regression/prediction ⭐ (Mid Q7 ii) | Discrete class label; accuracy, F1 | Continuous value; MAE, MSE, RMSE |
| Feature selection vs extraction | Keeps a subset of original features; meaning kept | Builds new combined features (PCA); meaning lost |
| Type I vs Type II error | False positive (healthy called sick); rate FP/(FP + TN) | False negative (sick called healthy); rate FN/(FN + TP) |
| Perceptron vs LMS | Thresholded output; updates only on mistakes; stops only if separable | Linear output y<sub>in</sub>; updates on every sample; settles near the least-squares solution even if not separable (exactly only if η shrinks) |
| Batch vs stochastic vs mini-batch GD | Batch: all samples per step, smooth, slow on big data | SGD: one sample per step, noisy, fast; mini-batch: b samples, the usual compromise |
| On-line vs off-line learning | Update after every pattern | Add up changes over the epoch, update once |
| ID3 vs C4.5 | Information gain; categorical only; no pruning, no missing values | Gain ratio; continuous thresholds; missing values; pessimistic pruning |
| Lazy vs eager learner | k-NN: no training, slow prediction, stores all data | Decision tree, SVM: builds a model, fast prediction |
| Hard vs soft margin ⭐ (End Q4b) | No errors allowed; needs separable data; sensitive to outliers | Slack ξ and C; allows violations; large C → nearly hard, small C → wider margin |
| Functional vs geometric margin | y(w·x + b); changes when w, b are scaled | y(w·x + b)/‖w‖; true distance, scale-free |
| Perceptron vs SVM | Any separating line; depends on order of data | Maximum-margin line; unique; convex optimisation |
| Bagging vs boosting | Parallel, bootstrap samples, equal votes, cuts variance | Sequential, re-weighted samples, weighted votes α<sub>t</sub>, cuts bias |
| Realizable vs agnostic | Target in H, zero error possible, m ∝ 1/ε | Target not in H (noise), m ∝ 1/ε<sup>2</sup> |
| Generative vs discriminative | Models p(x\|ω) and P(ω) (Naive Bayes) | Models P(ω\|x) or the boundary directly (logistic regression, SVM) |

## The numericals: answers to check against

| Question | Answer |
|---|---|
| **k-NN, Davis (37, 50, 2), k = 3, Euclidean, unscaled (End Q2)** | Distances John 15.17, Alex 15.00, Rio 152.24, Tim 122.00, Noa 15.75 → nearest Alex (No), John (Yes), Noa (Yes) → **Davis = Yes** (same with min–max scaling and with Manhattan) |
| √n rule, n = 5 | 2.24 → k = 3 |
| Regression errors, y = (3, 5, 7), ŷ = (2, 5, 9) | MAE 1.0, MSE 1.667, RMSE 1.291 |
| PCA of (2,1), (3,3), (4,3), (5,5), divisor n − 1 | Eigenvalues 4.228, 0.105; PC1 = (0.615, 0.788); keeps 97.6% of variance |
| Medical test P(D) = 0.01, P(+\|D) = 0.95, P(+\|ND) = 0.10 | P(+) = 0.1085, P(D\|+) = 0.0876 |
| Three classes, priors 0.5/0.3/0.2, likelihoods 0.2/0.4/0.5 | Evidence 0.32; posteriors 0.3125/0.375/0.3125 → ω<sub>2</sub> |
| Fruit, P(ω<sub>1</sub>) = 0.6, Low/Medium/High | P(ω<sub>1</sub>\|x) = 0.882/0.600/0.333 → ω<sub>1</sub>, ω<sub>1</sub>, ω<sub>2</sub>; Bayes error 0.28 |
| Minimum risk, λ<sub>12</sub> = 2, λ<sub>21</sub> = 1, posteriors 0.6/0.4 | R(α<sub>1</sub>) = 0.8, R(α<sub>2</sub>) = 0.6 → ω<sub>2</sub> |
| 1-D Gaussian, μ = 2, 4, σ = 1, priors 0.7/0.3 | Boundary x* = 3.42; Bayes error 0.139 |
| Minimum distance, m<sub>1</sub> = (2,2), m<sub>2</sub> = (6,5), x = (3,4) | 2.236 vs 3.162 → ω<sub>1</sub>; boundary 4x<sub>1</sub> + 3x<sub>2</sub> = 26.5 |
| Naive Bayes PlayTennis, Sunny/Cool/High/Strong | Yes 0.0053, No 0.0206 → **No** (0.795) |
| Confusion matrix TP 40, FN 10, FP 5, TN 45 | Accuracy 0.85, precision 0.889, recall 0.80, specificity 0.90, F1 0.842 |
| Linear discriminant g = 3x<sub>1</sub> + 4x<sub>2</sub> − 12 | ‖w‖ = 5; (4,3): g = 12, r = +2.4 |
| Activations at z = −2, 0, 2 | Sigmoid 0.1192/0.5/0.8808; tanh −0.964/0/0.964; ReLU 0/0/2 |
| Perceptron AND, w = (0.3, −0.1), b = −0.2, η = 0.1 | 5 epochs, 7 updates; final w = (0.2, 0.2), b = −0.3 |
| Perceptron OR, zero start, η = 1 | Final w = (1, 1), b = 0 after 4 updates |
| LMS one epoch, bipolar AND, w = b = 0.1, η = 0.1 | w<sub>1</sub> = 0.24013, w<sub>2</sub> = 0.21673, b = −0.13413 |
| Batch GD, x = (1,2,3), y = (2,4,6), η = 0.1 | Iter 1: w = 0.9333, b = 0.4, J = 1.8815; iter 2: w = 1.3511, b = 0.5733, J = 0.4028; → w = 2, b = 0 |
| Same data, η = 0.5 | Diverges: J = 29.26, 91.92, 288.97, 908.58 |
| Backprop 2-2-1, x = (0.5, 1.0), t = 1, η = 0.5 | h = 0.4750, 0.5498; o = 0.5542, E = 0.09938; δ<sub>o</sub> = 0.11015; after update o = 0.5756 |
| Momentum, η = 0.5, gradient 0.2, previous step −0.1, α = 0.9 | Δw = −0.19 |
| Entropy | [9+,5−] 0.940; [2+,3−] 0.971; [3+,1−] 0.811; [8+,2−] 0.722; [7+,7−] 1.000 |
| ID3 PlayTennis root | Gains Outlook 0.247, Humidity 0.152, Wind 0.048, Temperature 0.029 → **Outlook**; Sunny → Humidity, Rain → Wind |
| C4.5 gain ratio | SplitInfo(Outlook) 1.577, GainRatio 0.156 |
| Continuous threshold 40N, 48N, 60Y, 72Y, 80Y, 90N | Candidates 54, 85; best Temperature ≤ 54 (gain 0.459) |
| Cover–Hart, R* = 0.1 | R ≤ 0.18 (three classes: 0.185) |
| PAC, ε = 0.1, δ = 0.05, \|H\| = 1000 | m = 100 (ε = 0.05 → 199; δ = 0.01 → 116; \|H\| = 10<sup>6</sup> → 169; agnostic → 530) |
| Bootstrap left-out fraction | m = 1000: 0.3677; limit 0.368 (63.2% included) |
| Bagging, independent classifiers of error 0.3 | T = 3: 0.216; T = 5: 0.163; T = 11: 0.078; T = 21: 0.026 |
| AdaBoost round 1, ε<sub>1</sub> = 0.30 | α<sub>1</sub> = 0.4236; new weights 0.0714 (correct), 0.1667 (wrong); α<sub>2</sub> = 0.6496, α<sub>3</sub> = 0.7520 |
| SVM, (3,3)+ and (1,1)− | x<sub>1</sub> + x<sub>2</sub> = 4; w = (0.5, 0.5), b = −2; margin 2.8284 |
| SVM, (0,0)−, (2,0)+, (0,2)+ | x<sub>1</sub> + x<sub>2</sub> − 1 = 0; w = (1, 1), b = −1; margin 1.4142 |
| SVM, eight points | x<sub>1</sub> = 2; w = (1, 0), b = −2; margin 2; SVs (1,0), (3,1), (3,−1) |
| Kernel check x = (1,2), z = (3,4) | (x·z)<sup>2</sup> = 121 = φ(x)·φ(z); (x·z + 1)<sup>2</sup> = 144; RBF σ = 1: 0.0183 |
| Kernel perceptron on XOR, K = (x·z + 1)<sup>2</sup> | α = (1, 1, 1, 1), f(x) = −8x<sub>1</sub>x<sub>2</sub> |

## Diagrams to practise once

1. Traditional programming vs ML (data + rules vs data + answers).
2. AI ⊃ ML ⊃ DL nested circles; RL agent–environment loop.
3. Pattern recognition system block diagram.
4. Two overlapping class-conditional density curves with the decision boundary and the shaded Bayes error.
5. Single neuron: inputs, weights, bias, Σ, f, output.
6. Sigmoid, tanh and ReLU curves on one axis.
7. AND / OR lines and the XOR square that no line separates.
8. Gradient-descent bowl with steps going down (small η vs large η).
9. MLP (input, hidden, output layers) with the forward arrows and the backward error arrows.
10. PlayTennis decision tree (Outlook → Humidity / Wind).
11. k-NN scatter with the query and a circle around its 3 nearest neighbours.
12. Bias–variance U-curve; underfit line vs good fit vs overfit wiggle.
13. SVM road: w·x + b = 0, the two margin lines ±1, support vectors, width 2/‖w‖; soft margin with one point inside.
14. Kernel trick: XOR points in 2-D lifted to 3-D where a plane separates them.
15. Bagging (parallel) vs boosting (sequential) flow.

## The night-before plan

```mermaid
flowchart LR
  A[Read this sheet<br/>40 min] --> B[k-NN Davis<br/>on paper]
  B --> C[Backprop and GD<br/>formulas aloud]
  C --> D[Q7 pairs<br/>as tables]
  D --> E[Draw SVM road<br/>and MLP]
  E --> F[Sleep]
```

1. Read this sheet (40 minutes).
2. Do the End Q2 k-NN for Davis on paper without looking: distance table, sort, k = 3 vote, **Yes**. Check against the table.
3. Write the perceptron rule, the LMS rule, the GD update and the two backprop δ formulas from memory. Unit 2 gave about a third of last year's paper.
4. Say aloud the four End Q7 pairs (approximation/estimation, cost/loss, bias/variance, overfitting/underfitting) as 5-row tables.
5. Draw the SVM road diagram and the MLP diagram once each: no diagram, no full marks.
6. Read your Mid-2 paper: three of its four questions came back in last year's end-sem.
7. Sleep. A rested brain writes 4–5 pages per question.
