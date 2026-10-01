"""
Builds 1/index.html. The code listings and console output are read from the
files in 1/code so the page matches what the code does.
Run from the site folder:  python tools/build.py
"""

import html
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.dirname(HERE)
CODE = os.path.join(SITE, "1", "code")

NAME = "Kevin Fan 21005203"


def read(fname):
    with open(os.path.join(CODE, fname)) as fh:
        return html.escape(fh.read())


def scan(files, caption):
    imgs = "\n".join(f'<img src="media/{f}" alt="{caption}">' for f in files)
    return f'<div class="scan">\n{imgs}\n<p class="caption">{caption}</p>\n</div>'


page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SYDE 572 Assignment 1</title>
<link rel="stylesheet" href="../assets/style.css">
</head>
<body>

<h1>SYDE 572 Assignment 1</h1>
<p>{NAME}</p>

<nav><a href="../index.html">Home</a></nav>


<h2>Part 1: shortest distance from a point to a function</h2>

<h3>The objective function</h3>

<p>The distance from a point (x0, y0) to a point (x, y) on the curve y = f(x)
is sqrt((x - x0)^2 + (f(x) - y0)^2). A square root is expensive to compute and
it is monotonically increasing, so whichever x minimizes the distance also
minimizes the squared distance. Both methods below work on the squared distance
instead, which also keeps the derivatives simple:</p>

<p>D(x) = (x - x0)^2 + (f(x) - y0)^2<br>
D'(x) = 2(x - x0) + 2(f(x) - y0) f'(x)<br>
D''(x) = 2 + 2 f'(x)^2 + 2(f(x) - y0) f''(x)</p>

<p>Setting D'(x) = 0 rearranges to (x - x0) + (f(x) - y0) f'(x) = 0, which is
a dot product of the vector from the point to the curve, (x - x0, f(x) - y0),
with the tangent direction (1, f'(x)). So the condition says the shortest line
from the point to the curve has to be perpendicular to the tangent of the curve
at the point where it lands. That is the geometric meaning of the minimum.</p>

<p>For this assignment f(x) = x^2 + 5, so f'(x) = 2x and f''(x) = 2. Every
given point has y0 = 0, which gives</p>

<p>D(x) = (x - x0)^2 + (x^2 + 5)^2<br>
D'(x) = 4x^3 + 22x - 2x0<br>
D''(x) = 12x^2 + 22</p>

<p>D''(x) = 12x^2 + 22 is positive for every x, so D is convex and has exactly
one stationary point, which must be the minimum. This matters for both
algorithms: it means Newton-Raphson cannot converge to a local maximum or to
the wrong minimum, and any bracket that contains the minimum is valid for
golden section. The handwritten pages below have the analytical solution and
both numerical methods worked out for all five points.</p>

{scan(["scan1.jpg", "scan2.jpg"], "Part 1 by hand: analytical solution, Newton-Raphson and golden section")}

<h3>Algorithm A: Newton-Raphson</h3>

<p>Newton-Raphson finds a root of D' by replacing D' with its tangent line at
the current guess and jumping to where that line crosses zero:</p>

<p>x(k+1) = x(k) - D'(x(k)) / D''(x(k))</p>

<p>For this parabola that works out to
x(k+1) = x(k) - (4x(k)^3 + 22x(k) - 2x0) / (12x(k)^2 + 22).
Every run starts at x = 0 and stops when the step size drops under 1e-7. The
table below is the point (-4, 0). The step shrinks very quickly, roughly
squaring the error each time, so the number of correct digits about doubles per
iteration. That is why 4 iterations are enough to hit the tolerance.</p>

<table>
<tr><th>k</th><th>x(k)</th><th>D'(x(k))</th><th>D''(x(k))</th></tr>
<tr><td>0</td><td>0.000000000</td><td>8.000000000</td><td>22.000000</td></tr>
<tr><td>1</td><td>-0.363636364</td><td>-0.192336589</td><td>23.586777</td></tr>
<tr><td>2</td><td>-0.355481939</td><td>-0.000287990</td><td>23.516409</td></tr>
<tr><td>3</td><td>-0.355469693</td><td>-0.000000001</td><td>23.516304</td></tr>
<tr><td>4</td><td>-0.355469693</td><td>0.000000000</td><td>23.516304</td></tr>
</table>
<p class="caption">Newton-Raphson for the point (-4, 0), starting at x = 0.</p>

<p>Starting at x = 0 is a safe choice here. D'(0) = -2x0, so the first step is
always x1 = x0/11, and since the 4x^3 term has the same sign as x the root is
closer to zero than x0/11 is. So the first step lands just past the root and
every step after that approaches it from one side without overshooting. You can
see this in the table: after x1 the values only move in one direction.</p>

<img src="media/p1_newton.png" alt="Newton-Raphson steps for the five points">
<p class="caption">Intermediate steps for all five points. The top row plots
D'(x) with the iterates marked on it, so the method is looking for where the
black curve crosses zero. The bottom row shows the same iterates as points on
the parabola, with a line drawn from the given point to each one.</p>

<h3>Algorithm B: golden section search</h3>

<p>Golden section search works on D(x) directly and never uses a derivative,
so it still works when f' and f'' are hard to write down. It keeps a bracket
[a, b] known to contain the minimum and two interior points</p>

<p>x1 = a + (2 - phi)(b - a) and x2 = b - (2 - phi)(b - a),
where phi = (1 + sqrt(5))/2 = 1.618 and 2 - phi = 0.381966</p>

<p>If D(x1) &lt; D(x2) the minimum cannot be to the right of x2, so the new
bracket is [a, x2]. Otherwise it is [x1, b]. The reason for the golden ratio
spacing is that it makes one of the two old interior points land exactly where
a new interior point is needed, so each iteration only costs one new evaluation
of D instead of two. The bracket shrinks by a factor 1/phi = 0.618 every step,
which is a fixed rate that does not depend on the function.</p>

<p>The bracket has to contain the minimum for the method to be valid. Here
D'(0) = -2x0 and D'(x0) = 4x0^3 + 20x0 have opposite signs, so the minimum is
always between 0 and x0, and that is the bracket the code uses. For (0, 0) the
minimum is at x = 0 itself so it uses [-1, 1].</p>

<table>
<tr><th>k</th><th>a</th><th>b</th><th>x1</th><th>x2</th><th>b - a</th></tr>
<tr><td>0</td><td>-4.000000</td><td>0.000000</td><td>-2.472136</td><td>-1.527864</td><td>4.000000</td></tr>
<tr><td>1</td><td>-2.472136</td><td>0.000000</td><td>-1.527864</td><td>-0.944272</td><td>2.472136</td></tr>
<tr><td>2</td><td>-1.527864</td><td>0.000000</td><td>-0.944272</td><td>-0.583592</td><td>1.527864</td></tr>
<tr><td>3</td><td>-0.944272</td><td>0.000000</td><td>-0.583592</td><td>-0.360680</td><td>0.944272</td></tr>
<tr><td>4</td><td>-0.583592</td><td>0.000000</td><td>-0.360680</td><td>-0.222912</td><td>0.583592</td></tr>
<tr><td>5</td><td>-0.583592</td><td>-0.222912</td><td>-0.445825</td><td>-0.360680</td><td>0.360680</td></tr>
</table>
<p class="caption">Golden section for the point (-4, 0) on the bracket
[-4, 0].</p>

<p>The point reuse shows up between k = 0 and k = 1: the old x1 of -1.527864
becomes the new x2, so only one new value of D has to be computed. The number
of iterations needed is fixed in advance by the starting width and the
tolerance, since the bracket shrinks by the same factor each time:
N = ln(tol / (b - a)) / ln(0.618), which gives 37 for this bracket.</p>

<img src="media/p1_golden.png" alt="Golden section brackets shrinking">
<p class="caption">Intermediate steps. The bars are the bracket [a, b] at each
iteration and the dots are the two interior points x1 and x2. The bracket
closes in on the minimum from both sides.</p>

<h3>Results for y = x^2 + 5</h3>

<table>
<tr><th>Point</th><th>x Newton</th><th>d Newton</th><th>steps</th>
<th>x golden</th><th>d golden</th><th>steps</th></tr>
<tr><td>(0, 0)</td><td>0.0000000</td><td>5.0000000</td><td>1</td><td>0.0000000</td><td>5.0000000</td><td>35</td></tr>
<tr><td>(-4, 0)</td><td>-0.3554697</td><td>6.2898454</td><td>4</td><td>-0.3554697</td><td>6.2898454</td><td>37</td></tr>
<tr><td>(-8, 0)</td><td>-0.6720781</td><td>9.1334195</td><td>5</td><td>-0.6720781</td><td>9.1334195</td><td>38</td></tr>
<tr><td>(2, 0)</td><td>0.1807446</td><td>5.3513964</td><td>4</td><td>0.1807446</td><td>5.3513964</td><td>35</td></tr>
<tr><td>(6, 0)</td><td>0.5199037</td><td>7.6031254</td><td>4</td><td>0.5199037</td><td>7.6031254</td><td>38</td></tr>
</table>

<p>Both numerical methods agree with each other and with the analytical answers
from the handwritten pages to 7 decimal places, which is the check that the
routine works. The two methods get there very differently though. Newton needs
4 or 5 iterations because its error is squared each step, but it needs both f'
and f'' and a starting guess that is close enough. Golden section needs 35 to
38 iterations no matter which point is used, because its rate is fixed at 0.618
per step, but it only ever evaluates f and it cannot fail as long as the
bracket really does contain a single minimum. Newton is the better choice when
the derivatives are easy, golden section when they are not.</p>

<p>One thing worth noticing in the results: moving the given point further out
along the x axis barely moves the closest point. Going from (2, 0) to (6, 0)
triples x0 but x* only moves from 0.18 to 0.52. That is because the parabola
sits 5 units above the axis and gets steep quickly, so the nearest point stays
close to the vertex.</p>

<img src="media/p1_overview.png" alt="The parabola with the five points and the shortest distance to each">
<p class="caption">The five shortest distances. Each dashed line meets the
curve at a right angle to the tangent there, which is the D'(x) = 0
condition.</p>

<h3>Other parabolas and points</h3>

<p>Nothing in either function is specific to y = x^2 + 5. They take f, f' and
f'' as arguments, so any curve works. <code>parabola(a, b, c)</code> builds the
three functions for y = a x^2 + b x + c, and the table below runs four other
combinations of parabola and point, including a downward opening one and points
on both sides of the curve.</p>

<table>
<tr><th>Curve</th><th>Point</th><th>x Newton</th><th>d Newton</th><th>d golden</th></tr>
<tr><td>y = 0.5x^2 - 2x + 1</td><td>(3, -2)</td><td>2.4734658</td><td>1.2304354</td><td>1.2304354</td></tr>
<tr><td>y = 0.5x^2 - 2x + 1</td><td>(-1, 4)</td><td>-1.1474302</td><td>0.1546926</td><td>0.1546926</td></tr>
<tr><td>y = -2x^2 + 4x</td><td>(4, 1)</td><td>1.8918845</td><td>2.1893681</td><td>2.1893681</td></tr>
<tr><td>y = x^2 - x + 2</td><td>(2, 0)</td><td>0.8189171</td><td>2.1963105</td><td>2.1963105</td></tr>
</table>

<p>The two methods agree on all four. For these the golden section bracket was
just set to [-10, 10] rather than being worked out per case, which is fine
because each of these has a single minimum inside that range.</p>

<img src="media/p1_other.png" alt="Four other parabolas with points and shortest distances">
<p class="caption">Open circles are the Newton iterates, the dashed line is the
shortest distance.</p>

<h3>Non-polynomial functions</h3>

<p>Neither method assumes f is a polynomial anywhere. Newton only needs f, f'
and f'' to be computable at a point, and golden section only needs f. So the
only change is passing different functions in. The four cases below cover an
exponential, a logarithm, one with the variable in the denominator, and a
radical.</p>

<table>
<tr><th>Curve</th><th>Point</th><th>x Newton</th><th>d Newton</th><th>d golden</th></tr>
<tr><td>y = e^x</td><td>(2, 1)</td><td>0.582871</td><td>1.6230250</td><td>1.6230250</td></tr>
<tr><td>y = ln(x)</td><td>(0, 2)</td><td>1.314097</td><td>2.1699914</td><td>2.1699914</td></tr>
<tr><td>y = 1/x</td><td>(3, 1)</td><td>2.923000</td><td>0.6623765</td><td>0.6623765</td></tr>
<tr><td>y = sqrt(x)</td><td>(4, 0)</td><td>3.500000</td><td>1.9364917</td><td>1.9364917</td></tr>
</table>

<p>The square root case can be checked by hand, which is a useful test that the
code is right. With f(x) = sqrt(x) and the point (4, 0),
D(x) = (x - 4)^2 + x, so D'(x) = 2x - 7 and x* = 3.5, giving
d = sqrt(0.25 + 3.5) = 1.9364917. That matches both methods exactly. Newton
gets it in only 2 steps here because D is a quadratic in this case, so the
tangent line of D' is not an approximation, it is exact.</p>

<p>One practical note: for ln(x) and 1/x the function is undefined at or below
x = 0, so the starting guess and the bracket both have to be kept on the
positive side. Golden section needed a bracket like [0.05, 5] rather than one
starting at 0.</p>

<img src="media/p1_nonpoly.png" alt="Exponential, logarithmic, reciprocal and square root curves">

<h3>Console output</h3>
<pre>{read("out_part1.txt")}</pre>

<h3>part1.py</h3>
<pre>{read("part1.py")}</pre>


<h2>Part 2: fitting a line and a parabola</h2>

<p>Data: (0, 0.5), (2, 3.5), (1, 1.5), (3, 7.5).</p>

<h3>Setting up the MSE</h3>

<p>For a model yhat(x) with parameters theta, the mean squared error over the
n = 4 data points is</p>

<p>MSE(theta) = (1/n) * sum of (yhat(x_i) - y_i)^2</p>

<p>Two models are fitted:</p>

<ul>
<li>line, y = mx + b, with slope m and intercept b, so 2 parameters</li>
<li>parabola, y = ax^2 + bx + c, so 3 parameters</li>
</ul>

<p>Both models are linear in their parameters even though the parabola is not
linear in x. That makes the MSE a quadratic bowl in the parameters, so it has a
single minimum and no local minima to worry about. The sums that come up when
the partial derivatives are expanded are n = 4, sum x = 6, sum x^2 = 14,
sum x^3 = 36, sum x^4 = 98, sum y = 13, sum xy = 31 and sum x^2 y = 83.</p>

<h3>Analytical solution</h3>

<p>Setting every partial derivative to zero gives one linear equation per
parameter, and those are solved directly. For the line that is</p>

<p>14m + 6b = 31<br>
6m + 4b = 13</p>

<p>and for the parabola</p>

<p>98a + 36b + 14c = 83<br>
36a + 14b + 6c = 31<br>
14a + 6b + 4c = 13</p>

<p>The full working is on the handwritten page below.</p>

{scan(["scan3.jpg"], "Part 2 by hand: line and parabola")}

<h3>Newton-Raphson, one parameter at a time</h3>

<p>The numerical method optimizes one parameter at a time: fix every parameter
except one, take a Newton step in that one, then move to the next parameter and
repeat. For parameter j the update is</p>

<p>theta_j &lt;- theta_j - (dMSE/dtheta_j) / (d2MSE/dtheta_j^2)</p>

<p>Because the MSE is quadratic in each parameter on its own, this single
Newton step lands exactly on the best value of that parameter for the current
values of the others. There is no approximation in the inner step at all, so
all of the iterating happens in the outer loop that cycles through the
parameters. Substituting the sums, the two updates for the line work out to</p>

<p>m &lt;- (31 - 6b) / 14<br>
b &lt;- (13 - 6m) / 4</p>

<p>and the three for the parabola to</p>

<p>a &lt;- (83 - 36b - 14c) / 98<br>
b &lt;- (31 - 36a - 6c) / 14<br>
c &lt;- (13 - 14a - 6b) / 4</p>

<p>Starting every parameter at 0 gives the tables below. The parabola cycles
a, then b, then c, and only counts as one sweep once all three have been
updated.</p>

<table>
<tr><th>sweep</th><th>parameter</th><th>m</th><th>b</th><th>MSE</th></tr>
<tr><td>0</td><td>start</td><td>0.000000</td><td>0.000000</td><td>17.750000</td></tr>
<tr><td>1</td><td>m</td><td>2.214286</td><td>0.000000</td><td>0.589286</td></tr>
<tr><td>1</td><td>b</td><td>2.214286</td><td>-0.071429</td><td>0.584184</td></tr>
<tr><td>2</td><td>m</td><td>2.244898</td><td>-0.071429</td><td>0.580904</td></tr>
<tr><td>2</td><td>b</td><td>2.244898</td><td>-0.117347</td><td>0.578795</td></tr>
<tr><td>3</td><td>m</td><td>2.264577</td><td>-0.117347</td><td>0.577440</td></tr>
<tr><td>3</td><td>b</td><td>2.264577</td><td>-0.146866</td><td>0.576568</td></tr>
</table>
<p class="caption">First updates for the line, starting from m = 0, b = 0.</p>

<table>
<tr><th>sweep</th><th>parameter</th><th>a</th><th>b</th><th>c</th><th>MSE</th></tr>
<tr><td>0</td><td>start</td><td>0.000000</td><td>0.000000</td><td>0.000000</td><td>17.750000</td></tr>
<tr><td>1</td><td>a</td><td>0.846939</td><td>0.000000</td><td>0.000000</td><td>0.176020</td></tr>
<tr><td>1</td><td>b</td><td>0.846939</td><td>0.036443</td><td>0.000000</td><td>0.171372</td></tr>
<tr><td>1</td><td>c</td><td>0.846939</td><td>0.036443</td><td>0.231050</td><td>0.117988</td></tr>
<tr><td>2</td><td>a</td><td>0.800544</td><td>0.036443</td><td>0.231050</td><td>0.065253</td></tr>
<tr><td>2</td><td>b</td><td>0.800544</td><td>0.056722</td><td>0.231050</td><td>0.063814</td></tr>
<tr><td>2</td><td>c</td><td>0.800544</td><td>0.056722</td><td>0.363012</td><td>0.046400</td></tr>
</table>
<p class="caption">First updates for the parabola, starting from a = b = c = 0.</p>

<p>The first sweep gets most of the way there, dropping the MSE from 17.75 to
0.584 for the line, but the remaining error shrinks slowly and steadily after
that. The error in m goes -0.0857, -0.0551, -0.0354 and so on, which is a
constant ratio rather than the doubling accuracy Newton gives on a single
variable. The reason is that each update is only optimal for one parameter
while the others are held wrong, so the method has to keep cycling. It takes 48
sweeps to reach a tolerance of 1e-10 for the line and 476 for the parabola. The
parabola is much slower because its three columns x^2, x and 1 are strongly
correlated over x in [0, 3], so changing one parameter throws the other two off
by a lot and each sweep only claws back a small part of the error.</p>

<img src="media/p2_contours.png" alt="MSE contours for the line with the path of the updates">
<p class="caption">Intermediate steps drawn on the MSE surface for the line.
Because only one parameter changes per update, every move is either horizontal
or vertical and the path is a staircase down the valley. This plot starts at
(0, 2.5) rather than (0, 0) so the steps are visible, since from (0, 0) the
first step already lands almost on top of the answer. The parabola has three
parameters so its surface cannot be drawn as a 2D contour, but the next plot
shows its intermediate fits.</p>

<img src="media/p2_steps.png" alt="The fitted curve after each of the first updates">
<p class="caption">The same first six updates seen as fitted curves, with the
final answer dashed.</p>

<h3>Results</h3>

<table>
<tr><th>Model</th><th>Fitted equation</th><th>MSE</th></tr>
<tr><td>line</td><td>y = 2.3x - 0.2</td><td>0.5750000</td></tr>
<tr><td>parabola</td><td>y = 0.75x^2 + 0.05x + 0.55</td><td>0.0125000</td></tr>
</table>

<p>The one parameter at a time method reaches the same answers as the
analytical solution, which is the check that it is working. The parabola fits
far better, with an MSE 46 times smaller. That makes sense from the data: the y
values 0.5, 1.5, 3.5, 7.5 grow faster and faster as x increases, which is
curvature that a straight line has no way to follow. The line residuals are
-0.7, 0.6, 0.9 and -0.8 going left to right, so it sits too low at both ends
and too high in the middle, which is the pattern you get when a model is
missing curvature. The parabola residuals are 0.05, -0.15, 0.15 and -0.05, much
smaller and with no obvious pattern left.</p>

<p>Worth noting that with 4 data points and 3 parameters the parabola is close
to interpolating the data, so a lower MSE on its own does not prove it is the
better model. It only takes one more parameter to fit the 4 points exactly.</p>

<img src="media/p2_fits.png" alt="The two fitted curves and the data">

<p>One assumption worth stating: the MSE here divides by n = 4. If it is
defined as the plain sum of squared errors instead, the two values become 2.30
and 0.0510. The fitted parameters are the same either way, since dividing by a
constant does not move the minimum.</p>

<h3>Console output</h3>
<pre>{read("out_part2.txt")}</pre>

<h3>part2.py</h3>
<pre>{read("part2.py")}</pre>


<h2>Running the code</h2>
<pre>pip install numpy matplotlib
cd 1/code
python part1.py
python part2.py</pre>

<p>Both print the tables above and save their plots into <code>1/media</code>.
To try a different point or curve, change the <code>points</code> list or the
<code>cases</code> list near the bottom of <code>part1.py</code>.</p>

</body>
</html>
"""

out = os.path.join(SITE, "1", "index.html")
with open(out, "w") as fh:
    fh.write(page)
print("wrote", out)
