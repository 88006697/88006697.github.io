"""
Builds 2/index.html. The code listings and console output are read from the
files in 2/code so the page matches what the code does.
Run from the site folder:  python tools/build2.py
"""

import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
CODE = os.path.join(SITE, "2", "code")

NAME = "Kevin Fan 21005203"


def read(fname):
    with open(os.path.join(CODE, fname)) as fh:
        return html.escape(fh.read().rstrip())


page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SYDE 572 Assignment 2</title>
<link rel="stylesheet" href="../assets/style.css">
</head>
<body>

<h1>SYDE 572 Assignment 2</h1>
<p>{NAME}</p>

<nav><a href="../index.html">Home</a></nav>


<h2>Part 1: polynomial basis functions</h2>

<h3>Setup</h3>

<p>The data is the Boston housing set: 506 tracts, 13 features, and the target
MEDV in $1000s. I used the first 100 rows for training and the other 406 rows
for testing.</p>

<p>Each feature was normalized to zero mean and unit variance using all 506
rows (Xn in the handout) before the split. The target was not normalized, so all
RMS errors are in $1000s. CHAS is 0 in all 100 training rows, so its normalized
column is constant and parallel to the bias column, and the pseudo-inverse
splits the constant term between the bias and the CHAS weight.</p>

<p>Without regularization the weights are the pseudo-inverse solution, computed
with <code>np.linalg.lstsq</code>. With regularization I used the ridge
regression form from the lectures,
w = (&Phi;<sup>T</sup>&Phi; + &lambda;I)<sup>&minus;1</sup>&Phi;<sup>T</sup>t,
which is (XX<sup>T</sup> + &lambda;I)<sup>&minus;1</sup>Xy with &Phi; = X<sup>T</sup>.
&Phi; includes the bias column (x<sub>0</sub> = 1), so the bias weight is
penalized as well. RMS error is the square root of the mean squared error. The
handout does not say how to assign the folds, so for cross-validation I used 10
folds of 10 consecutive rows, like the diagram in the tutorial, and averaged the
validation RMS error over the folds.</p>

<p>The basis is the powers of each feature on its own (x<sub>1</sub>,
x<sub>1</sub><sup>2</sup>, &hellip;, x<sub>2</sub>, x<sub>2</sub><sup>2</sup>,
&hellip;) with no cross terms, plus a bias.</p>

<h3>(a) Training and test error against degree</h3>

<img src="media/a_poly_error.png" alt="Training and test RMS error against polynomial degree">
<p class="caption">Figure 1. Training and test RMS error against polynomial
degree, all 13 features. The y-axis is a log scale.</p>

<table>
<tr><th>Degree</th><th>Training RMS</th><th>Test RMS</th></tr>
<tr><td>1</td><td>2.04</td><td>36.8</td></tr>
<tr><td>2</td><td>1.58</td><td>1.10 &times; 10<sup>3</sup></td></tr>
<tr><td>3</td><td>1.45</td><td>4.87 &times; 10<sup>4</sup></td></tr>
<tr><td>4</td><td>1.27</td><td>2.19 &times; 10<sup>7</sup></td></tr>
<tr><td>5</td><td>1.10</td><td>1.84 &times; 10<sup>10</sup></td></tr>
<tr><td>6</td><td>1.07</td><td>5.02 &times; 10<sup>12</sup></td></tr>
<tr><td>7</td><td>0.98</td><td>3.67 &times; 10<sup>12</sup></td></tr>
</table>

<p>Training error goes down steadily as the degree goes up, from 2.04 to 0.98.
Test error is already 36.8 at degree 1 and grows by orders of magnitude after
that, so the model overfits badly. Part of this is extrapolation: CRIM is between
0.006 and 1.61 in the training rows, but 54 test tracts have CRIM above 10, and
higher powers of those values give huge predictions. The high-degree fits are
also ill-conditioned, so their exact test errors, including the small drop at
degree 7, depend on the solver and should only be read as very large.</p>

<h3>(b) Degree 1 weights</h3>

<table>
<tr><th>Weight</th><th>Feature</th><th>Value</th></tr>
<tr><td>bias</td><td></td><td>8.762</td></tr>
<tr><td>w0</td><td>CRIM</td><td>&minus;29.697</td></tr>
<tr><td>w1</td><td>ZN</td><td>0.125</td></tr>
<tr><td>w2</td><td>INDUS</td><td>&minus;1.157</td></tr>
<tr><td>w3</td><td>CHAS</td><td>&minus;2.389</td></tr>
<tr><td>w4</td><td>NOX</td><td>2.491</td></tr>
<tr><td>w5</td><td>RM</td><td>6.030</td></tr>
<tr><td>w6</td><td>AGE</td><td>&minus;2.228</td></tr>
<tr><td>w7</td><td>DIS</td><td>0.016</td></tr>
<tr><td>w8</td><td>RAD</td><td>&minus;1.217</td></tr>
<tr><td>w9</td><td>TAX</td><td>&minus;1.285</td></tr>
<tr><td>w10</td><td>PTRATIO</td><td>0.334</td></tr>
<tr><td>w11</td><td>B</td><td>0.606</td></tr>
<tr><td>w12</td><td>LSTAT</td><td>0.187</td></tr>
</table>

<p>w5, the weight on the 6th feature (RM), is 6.030. Because RM was normalized,
this is the change in MEDV for one standard deviation of RM (0.70 rooms), or
about $8.6k per extra room. A large positive effect from house size seems
reasonable.</p>

<p>The weight on the 8th feature (DIS), w7, is 0.016, which is basically zero.
This seems reasonable for this training set but not as a general statement about
distance. In the training rows DIS is only weakly related to MEDV and is
correlated with ZN and NOX, which are already in the model, so it has little left
to explain. With correlated features &Phi;<sup>T</sup>&Phi; gets closer to
singular and the individual weights are not stable, so I would not read much
into this value.</p>

<h3>(c) One feature only: ZN</h3>

<p>For this part I used only the 2nd feature, ZN, and fit degrees 1 to 7.
Figures 2 to 4 show degrees 1, 4 and 7.</p>

<img src="media/c_zn_deg1.png" alt="Degree 1 fit on ZN">
<p class="caption">Figure 2. Degree 1 polynomial on ZN.</p>

<img src="media/c_zn_deg4.png" alt="Degree 4 fit on ZN">
<p class="caption">Figure 3. Degree 4 polynomial on ZN.</p>

<img src="media/c_zn_deg7.png" alt="Degree 7 fit on ZN">
<p class="caption">Figure 4. Degree 7 polynomial on ZN.</p>

<p>ZN on its own explains very little: 63 of the 100 training tracts have
ZN = 0 (the vertical column at &minus;0.49), with MEDV anywhere from 12.7 to 43.8.
Degrees 1 and 4 both show a gentle upward trend and have similar test RMS (9.35
and 9.39). Degree 7 overfits. The training rows have no ZN values between 28 and
75 (0.71 to 2.73 on the axis), and in that gap the curve rises to about 57 while
the test tracts there have MEDV between 17.1 and 37.0, so its test RMS goes up to
10.42.</p>

<h3>(d) Regularized degree 8 polynomial on ZN</h3>

<img src="media/d_poly_reg.png" alt="Average validation RMS against lambda for the degree 8 polynomial">
<p class="caption">Figure 5. Average 10-fold validation RMS error against
&lambda;, degree 8 polynomial on ZN. &lambda; = 0 cannot be shown on the log
axis. Its value, 1.51 &times; 10<sup>5</sup>, is reported here instead of as a
baseline line because it is far above the other points.</p>

<table>
<tr><th>&lambda;</th><th>Average validation RMS</th></tr>
<tr><td>0</td><td>1.51 &times; 10<sup>5</sup></td></tr>
<tr><td>0.01</td><td>58.5</td></tr>
<tr><td>0.1</td><td>54.4</td></tr>
<tr><td>1</td><td>9.82</td></tr>
<tr><td>10</td><td>7.32</td></tr>
<tr><td>100</td><td>24.4</td></tr>
<tr><td>1000</td><td>36.9</td></tr>
</table>

<p>I would choose &lambda; = 10, which has the lowest average validation error
(7.32).</p>

<h3>Console output</h3>
<pre>{read("out_part1.txt")}</pre>

<h3>part1.py</h3>
<pre>{read("part1.py")}</pre>


<h2>Part 2: Gaussian basis functions</h2>

<p>The lectures write the Gaussian basis as
&phi;<sub>k</sub>(x) = exp(a||x &minus; c<sub>k</sub>||<sup>2</sup>), which only
gives a bump when a is negative, so I used
&phi;<sub>j</sub>(x) = exp(&minus;a||x &minus; c<sub>j</sub>||<sup>2</sup>) with
a &gt; 0, on all 13 normalized features, plus a bias. Neither the handout nor the
lectures give a value for a, so I used a = 1/8. The basis uses Euclidean
distance, so the normalization works like the scaled Euclidean distance from the
lectures: each feature axis is rescaled so that features with large units, such
as TAX and B, do not dominate. The data split, normalization, ridge form and
cross-validation are the same as in Part 1.</p>

<h3>(e) Training and test error against number of basis functions</h3>

<p>For the centres I made one random permutation of the 100 training rows (numpy
seed 0) and took the first K rows.</p>

<img src="media/e_gauss_error.png" alt="Training and test RMS error against number of Gaussian basis functions">
<p class="caption">Figure 6. Training and test RMS error against the number of
Gaussian basis functions K, a = 1/8, no regularization.</p>

<table>
<tr><th>K</th><th>Training RMS</th><th>Test RMS</th></tr>
<tr><td>5</td><td>2.83</td><td>8.31</td></tr>
<tr><td>15</td><td>2.25</td><td>8.35</td></tr>
<tr><td>25</td><td>1.95</td><td>9.60</td></tr>
<tr><td>35</td><td>1.59</td><td>8.99</td></tr>
<tr><td>45</td><td>1.42</td><td>10.48</td></tr>
<tr><td>55</td><td>1.30</td><td>13.81</td></tr>
<tr><td>65</td><td>0.65</td><td>19.79</td></tr>
<tr><td>75</td><td>0.51</td><td>16.21</td></tr>
<tr><td>85</td><td>0.45</td><td>14.90</td></tr>
<tr><td>95</td><td>0.22</td><td>21.44</td></tr>
</table>

<p>Training error falls steadily from 2.83 at K = 5 to 0.22 at K = 95. Test error
is lowest at K = 5 and 15 (8.31 and 8.35) and mostly gets worse as K grows,
reaching 21.44 at K = 95, so more basis functions lead to overfitting. The test
curve is noisy, and its exact shape depends on which points are picked as
centres. The test errors stay far below the polynomial ones in (a) because every
basis value is between 0 and 1, so the predictions cannot blow up the way high
powers do.</p>

<h3>(f) Regularized Gaussian regression, 90 basis functions</h3>

<p>In each fold I used the 90 training points of that fold as the centres.</p>

<img src="media/f_gauss_reg.png" alt="Average validation RMS against lambda for 90 Gaussian basis functions">
<p class="caption">Figure 7. Average 10-fold validation RMS error against
&lambda;, 90 Gaussian basis functions on all 13 features. The dashed line is
&lambda; = 0 (17.6).</p>

<table>
<tr><th>&lambda;</th><th>Average validation RMS</th></tr>
<tr><td>0</td><td>17.6</td></tr>
<tr><td>0.01</td><td>2.49</td></tr>
<tr><td>0.1</td><td>2.69</td></tr>
<tr><td>1</td><td>3.29</td></tr>
<tr><td>10</td><td>4.73</td></tr>
<tr><td>100</td><td>7.03</td></tr>
<tr><td>1000</td><td>11.66</td></tr>
</table>

<p>I would choose &lambda; = 0.01, which has the lowest average validation error
(2.49).</p>

<h3>Console output</h3>
<pre>{read("out_part2.txt")}</pre>

<h3>part2.py</h3>
<pre>{read("part2.py")}</pre>


<h2>Running the code</h2>
<pre>pip install numpy matplotlib
cd 2/code
python part1.py
python part2.py</pre>

<p>Both read <code>housing.data</code> from the same folder, print the results
above and save their plots into <code>2/media</code>. <code>part2.py</code>
imports the shared functions from <code>part1.py</code>. Python names cannot
contain hyphens, so the functions from the handout are
<code>polynomial_regression</code>, <code>polynomial_regression_1d_vis</code>,
<code>polynomial_regression_reg</code>, <code>gaussian_regression</code> and
<code>gaussian_regression_reg</code>.</p>

</body>
</html>
"""

out = os.path.join(SITE, "2", "index.html")
with open(out, "w") as fh:
    fh.write(page)
print("wrote", out)
