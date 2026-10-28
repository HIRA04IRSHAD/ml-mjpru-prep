# PYQ analysis: what MJPRU actually asks

Three MJPRU CS-401T papers analysed: the 2025-26 end-sem, Mid-2 2025, and an earlier mid paper (MCQs + short answers). With so few papers, the strongest signal is how the end-sem reused the mids.

::: tip The one-line summary
**The end-sem was built from the mids.** Three of the four Mid-2 questions came back (MLP → backpropagation in MLPs, activation functions → short note, gradient descent → a full 14-mark question), and the earlier mid's SVM and overfitting-vs-underfitting came back too. **Unit 2 (neural networks and gradient descent) gave about a third of the paper.** Add the k-NN numerical (a full 14-mark question), Bayes decision theory, decision trees and the Unit 4 "differentiate" pairs, and you can choose 5 of 8 comfortably.
:::

## Topic frequency (how many of the 3 papers asked it)

| Rank | Topic | Unit | Asked in | Count | Chance this year |
|---|---|---|---|---|---|
| 1 | **Perceptron, MLP and backpropagation** | 2 | End Q5(a); Mid-2 Q1; MCQ Q1, Q2 | 3 / 3 | Very high |
| 2 | **Gradient descent: working, types, linear regression** | 2 | End Q8 (14 marks); Mid-2 Q4 | 2 | Very high |
| 3 | **Activation functions: sigmoid, tanh, ReLU** | 2 | End Q6; Mid-2 Q2 | 2 | Very high |
| 4 | **SVM: working, hard vs soft margin, kernel trick** | 5 | End Q4(a)(b); MCQ Q6 | 2 | Very high |
| 5 | **Overfitting vs underfitting; bias vs variance** | 4 | End Q7(c)(d); MCQ Q7(i) | 2 | Very high |
| 6 | **Decision trees: ID3, C4.5, continuous attributes** | 3 | End Q3(b); MCQ Q3 | 2 | High |
| 7 | **k-NN algorithm with a numerical** | 3 | End Q2 (14 marks) | 1 | Very high |
| 8 | **Class-conditional density; Bayes decision theory** | 1 | End Q3(a) | 1 | High |
| 9 | **AI vs ML vs DL; supervised, unsupervised, reinforcement** | 1 | End Q1 | 1 | High |
| 10 | **Widrow–Hoff (LMS) learning** | 2 | End Q5(b) | 1 | High |
| 11 | Confusion matrix, Type I and Type II errors | 1 | Mid-2 Q3 | 1 | Medium-high |
| 12 | Feature selection, dimensionality reduction, hypothesis class | 1, 4 | End Q6 | 1 | Medium-high |
| 13 | Approximation vs estimation error; cost vs loss | 4, 2 | End Q7(a)(b) | 1 | Medium-high |
| 14 | Classification vs prediction; concept learning | 1 | MCQ Q4, Q7(ii) | 1 | Medium |
| — | **Not yet asked:** linear discriminants and decision regions, linear separability (XOR), perceptron numerical, entropy/ID3 numerical, pruning, PAC learning, sample complexity, VC dimension, inductive bias, Occam's razor, empirical risk, structural risk minimisation, bootstrapping, bagging, boosting, dual perceptron, margin-based bounds | 1–5 | never | 0 | Medium: the Unit 4 theory is overdue |

```mermaid
flowchart LR
  A[Unit 1<br/>~25% of marks] --- B[Unit 2<br/>~31%]
  B --- C[Unit 3<br/>~19%]
  C --- D[Unit 4<br/>~12.5%]
  D --- E[Unit 5<br/>~12.5%]
```

## Unit-wise weight (2025-26 end-sem, 112 marks on paper)

| Unit | Questions | Marks |
|---|---|---|
| Unit 1 · ML basics and Bayes | Q1(a), Q1(b), Q3(a), Q6 (feature selection, dimensionality reduction) | 28 |
| Unit 2 · Linear machines and neural networks | Q5(a), Q5(b), Q8, Q6 (activation function), Q7(b) (cost vs loss) | 35 |
| Unit 3 · Decision trees and k-NN | Q2, Q3(b) | 21 |
| Unit 4 · Learning theory | Q7(a)(c)(d), Q6 (hypothesis class) | 14 |
| Unit 5 · SVM | Q4 | 14 |

You need 5 of 8 questions. **Units 1, 2 and 3 cover six of the eight questions.** Unit 5 (SVM) is one full question that repeats; Unit 4 appears through short notes and "differentiate" pairs.

## The numericals

| Numerical | Asked | Where solved |
|---|---|---|
| **k-NN: predict Davis with k = 3** | End Q2 (14 marks) | [Unit 3](#unit3): Davis = **Yes** |
| Entropy, information gain, ID3 tree | Not yet (but Q3b asks ID3) | [Unit 3](#unit3) |
| Perceptron weight updates (AND/OR) | Not yet | [Unit 2](#unit2) |
| LMS / gradient-descent weight update | Not yet (Q5b, Q8 are theory) | [Unit 2](#unit2) |
| Bayes posterior from priors and likelihoods | Not yet (Q3a is theory) | [Unit 1](#unit1) |
| Confusion-matrix metrics (accuracy, precision, recall) | Not yet (Mid-2 Q3 is theory) | [Unit 1](#unit1) |

**Write the formula first, then a table, then the answer in a box**, and say how ties and scaling are handled (k-NN) or which logarithm base you use (entropy, base 2).

## Repeat-question patterns

- **Mid-2 is a preview of the end-sem.** Three of its four questions came back. When your Mid-2 paper arrives, treat every question on it as a likely end-sem question.
- **"Differentiate between" (Q7) and "short notes" (Q6)** are 3.5 marks each: one compact table or half a page.
- **Long theory questions ask "what, why, how, advantages"** (k-NN, SVM, backpropagation, gradient descent): definition → diagram → algorithm steps → formula → advantages/disadvantages.
- **One full numerical question** (k-NN in 2025-26). Expect a numerical on k-NN, ID3 entropy, perceptron or Bayes.

## Predicted end-sem 2026-27 (most likely 8 questions)

| Q | (a) | (b) |
|---|---|---|
| 1 | ML vs AI vs DL; applications; types of learning | Classification vs regression; feature selection vs extraction; dimensionality reduction |
| 2 | Bayes decision theory, class-conditional density, Bayes optimal classifier and error | Bayes numerical (posterior, decision) / confusion matrix and metrics |
| 3 | Perceptron learning algorithm with a numerical; linear separability (XOR) | Widrow–Hoff / LMS; gradient descent and its types |
| 4 | MLP architecture and backpropagation with a diagram | Activation functions; on-line vs off-line learning; important parameters |
| 5 | Decision trees: entropy, information gain, ID3 numerical | C4.5: gain ratio, continuous attributes, pruning |
| 6 | k-NN with a numerical; choice of k; lazy learning | Nearest-neighbour error probability |
| 7 | SVM: margin, maximal margin classifier, hard vs soft margin | Kernel trick and kernel functions; dual perceptron |
| 8 | Bias–variance, overfitting–underfitting, approximation–estimation error | PAC learning, VC dimension, Occam's razor; bagging and boosting |

**Mock Paper 1** on this site is built on this prediction.
