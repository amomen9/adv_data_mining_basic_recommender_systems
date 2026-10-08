# Advances in Data Mining: Assignment 1, Recommender Systems

My solution to the first programming assignment of the Advances in Data Mining course (2026-27, semester 1, week 2). The task is to build three basic recommender system parts from scratch on a table of movie ratings: cosine similarity between users or movies, user-based and item-based collaborative filtering and matrix factorization with gradient descent.

The graded file is **assignment_4993128.py**. Everything else is the assignment text, the lecture material and a notebook I wrote to understand the gradient descent part properly.

---

## Repository contents

| File | What it is |
| --- | --- |
| **assignment_4993128.py** | my implementation, the file that gets submitted |
| **validate_submission.py** | submission checks that came with the assignment (pytest) |
| **Assignment_1_Recommender_Syst.pdf** | assignment description |
| **data/u.data** | the ratings dataset |
| **Matrix_fact_gradient_descent.ipynb** | my notes on matrix factorisation and gradient descent, with worked examples |
| **Recommender systems.pptx** | lecture slides of week 2 |
| **cosine sim.xlsx**, **collaborative filtering.xlsx**, **UV-decomposition.xlsx** | lecture spreadsheets with the same calculations done by hand |
| **Table of Contents.html** | index page of the week 2 material, exported from the course site |
| **requirements.txt** | package versions given with the assignment |
| **LICENSE** | MIT |

---

## Dataset

**data/u.data** has 100,000 ratings, one per line and tab separated:

    UserID  MovieID  Rating  Timestamp
    196     242      3       881250949

- 943 users and 1682 movies
- ratings are whole numbers from 1 to 5
- every user rated at least 20 movies
- only about 6.3% of the user-movie table is filled, so most cells are empty

These numbers match the MovieLens 100K dataset. The script turns the file into a user-item matrix with **pivot_table** (rows are users, columns are movies) and the missing ratings stay **NaN**. The file is kept as is, the assignment asks not to change it.

---

## What is implemented

The four required functions keep the exact signatures of the skeleton, the validator checks them.

### 1. Cosine similarity

Function **similarity_matrix(matrix, k=5, axis=0)**. With **axis=0** it compares users (rows) and with **axis=1** movies (columns).

For two users (or two movies) $A$ and $B$, let $n$ be the number of entries both of them have rated and $a_i$, $b_i$ their ratings on the $i$-th of those entries:

$$\text{cos}(A,B) = \frac{\sum_{i=1}^{n} a_i \, b_i}{\sqrt{\sum_{i=1}^{n} a_i^2 \cdot \sum_{i=1}^{n} b_i^2}}$$

How I did it:

- only the entries rated by both sides go into the formula; a pair with nothing in common is skipped
- the formula is written out by hand, with **np.dot** and **np.linalg.norm** only as helpers (no ready made cosine function, which the assignment forbids)
- each pair is computed one time and stored for both sides
- the result is a dict keyed by the real user or movie ID, not the row or column position; each value is a list of **(id, similarity)** tuples sorted from high to low and cut at **k**

### 2. Collaborative filtering

Functions **user_based_cf(user_id, movie_id, user_similarity, user_item_matrix, k=5)** and **item_based_cf(user_id, movie_id, item_similarity, user_item_matrix, k=5)**.

The predicted rating $\hat{r}_{u,i}$ of user $u$ for movie $i$ is a weighted average over $N_k(u)$, the $k$ users most similar to $u$. Here $r_{v,i}$ is the rating neighbour $v$ gave to movie $i$ and $\text{sim}(u,v)$ the cosine similarity from above:

$$\hat{r}_{u,i} = \frac{\sum_{v \in N_k(u)} \text{sim}(u,v) \, r_{v,i}}{\sum_{v \in N_k(u)} \text{sim}(u,v)}$$

Item-based is the same idea turned around. The neighbours are the $k$ movies most similar to movie $i$, and the ratings used are the ones user $u$ gave to those movies.

Some details:

- neighbours come from the dict built by **similarity_matrix**, so **k** here can not be bigger than the **k** used there
- the k neighbours are picked first and a neighbour without a rating for the target is skipped, so a prediction can rest on fewer than k ratings
- if none of them has a rating the function returns **NaN**

### 3. Matrix factorization

Function **matrix_factorization(utility_matrix, feature_dimension=2, learning_rate=0.001, regularization=0.02, n_steps=2000)**, with two helpers **loss_function** and **gradient_loss_function**.

The utility matrix $R$ ($m$ users by $n$ items) is approximated by $UV^T$. $U$ is $m \times K$, $V$ is $n \times K$ and $K$ is **feature_dimension**; $u_i$ is row $i$ of $U$ and $v_j$ row $j$ of $V$. The loss adds up the squared errors over the known ratings only ($\Omega$ is the set of known cells $(i,j)$) plus a penalty with strength $\lambda$ (**regularization**) on the size of the two matrices:

$$L(U,V) = \sum_{(i,j) \in \Omega} \left(r_{ij} - u_i \cdot v_j\right)^2 + \lambda \left(\lVert U \rVert^2 + \lVert V \rVert^2\right)$$

$\lVert \cdot \rVert^2$ is the sum of all squared entries of a matrix. With $E$ the error matrix ($e_{ij} = r_{ij} - u_i \cdot v_j$ on known cells and 0 on the missing ones) the gradients are

$$\frac{\partial L}{\partial U} = -2EV + 2\lambda U, \qquad \frac{\partial L}{\partial V} = -2E^T U + 2\lambda V$$

and every step moves both matrices against their gradient, with $\eta$ the **learning_rate**:

$$U \leftarrow U - \eta \, \frac{\partial L}{\partial U}, \qquad V \leftarrow V - \eta \, \frac{\partial L}{\partial V}$$

$U$ and $V$ start random (**np.random.rand**, no fixed seed) so two runs give slightly different factors. Each step uses all known ratings at once, so this is plain batch gradient descent and not the stochastic version of slide 47.

---

## Running it

### Requirements

The packages in **requirements.txt**, plus **pytest** for the validator:

    pip install -r requirements.txt pytest

I use my global Python for this, no virtual environment. On my machine I ran everything with Python 3.14 and newer package versions than the pinned ones (numpy 2.4, pandas 3.0, scikit-learn 1.8) and it worked fine.

### Run the script

    python assignment_4993128.py

The main block at the end of the file:

1. prints the 5 users most similar to user 3
2. prints the 5 movies most similar to movie 10
3. predicts the rating of user 13 for movie 100, user-based and item-based
4. factorizes a small 5 x 5 matrix with two missing cells (the one of slide 40) in 5000 steps and prints $UV^T$

The similarity part loops over every pair in plain Python, around 444 thousand user pairs and 1.4 million movie pairs. So it is not fast; the whole run took about 67 seconds on my machine.

Output of one run, as is:

    [(np.int64(5), np.float64(1.0)), (np.int64(37), np.float64(1.0)), (np.int64(51), np.float64(1.0)), (np.int64(55), np.float64(1.0)), (np.int64(106), np.float64(1.0))]
    [(np.int64(493), np.float64(1.0000000000000002)), (np.int64(787), np.float64(1.0000000000000002)), (np.int64(835), np.float64(1.0000000000000002)), (np.int64(986), np.float64(1.0000000000000002)), (np.int64(1024), np.float64(1.0000000000000002))]
    predicted user 13 rating for movie 100, according to user-based collaborative filtering is: 4.66
    predicted user 13 rating for movie 100, according to item-based collaborative filtering is: 1.00
    Current guess:
     [[4.52582632 2.22421614 3.87302011 4.54083899 2.85494677]
     [3.37788329 0.66080283 2.31436489 3.40878245 1.1541434 ]
     [1.51030594 3.81293192 3.06338958 1.45479553 3.95399286]
     [2.47590269 4.79933883 4.18491113 2.41350693 5.06339607]
     [4.08484298 4.14476806 4.72825294 4.05626935 4.66572481]]

The last matrix changes a bit from run to run because of the random start. The two cells that were missing in the input (row 3 column 2 and row 5 column 5) now have predictions, about 3.8 and 4.7 here.

### Check the submission

Run it from repository folder, the validator looks for the assignment file in the current directory:

    python -m pytest validate_submission.py

It checks three things:

- there is exactly one file named like **assignment_1234567.py** (the student number without the leading s)
- nothing runs on import; only functions, classes, imports and the main block are allowed at top level
- the four function signatures match the skeleton

All three pass on the current file.

---

## Notes on the results

The output above looks odd at first, every top-5 similarity is 1.0. It is a side effect of the formula together with using only the shared ratings, not a bug:

- the similarity only looks at the entries both sides rated. User 3 shares exactly one movie with each of users 5, 37, 51, 55 and 106, and with one shared rating $\frac{a \cdot b}{\sqrt{a^2 b^2}} = 1$ whatever the two ratings are
- ratings are all positive (1 to 5), so similarities are never negative and tend to be high anyway
- ties keep the order in which they were found, so among tied pairs the lowest IDs end up in the top-k

The movie side has the same issue but less extreme: movie 10 and its neighbours share 2 to 8 raters, and those raters gave both movies proportional ratings.

This also explains why the two predictions for user 13 and movie 100 are so far apart. Out of the 5 movies most similar to movie 100, user 13 rated only two (movies 437 and 439) and gave both a 1. So the item-based prediction is just 1.00, while the user-based one comes from other users and lands at 4.66.

A common fix is to ask for a minimum number of shared ratings or to subtract each user's mean rating first (centered cosine). I left these out since the assignment fixes the formula.

---

## The notebook

**Matrix_fact_gradient_descent.ipynb** is my own walk through of the matrix factorization part, built on the 5 x 5 rating table of slide 40 (the same table the main block uses). It goes from the model and the error function to the partial derivatives, does the first update steps by hand and then checks them in code. After that it covers:

- the error surface when only two numbers are free (slides 43 and 44)
- what different learning rates do
- why the start has to be random and why there are many equally good answers
- stochastic gradient descent and regularisation (slides 47 and 48)

---

## License

MIT, see **LICENSE**.
