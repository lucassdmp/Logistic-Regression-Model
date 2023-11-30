<title>Logistic Regression with Gradient Descent</title>

<h1>Logistic Regression with Gradient Descent</h1>

<p>This script performs logistic regression using gradient descent on the Iris dataset. It predicts the species of iris flowers, considering three classes: Iris-setosa, Iris-versicolor, and Iris-virginica.</p>

<h2>Prerequisites</h2>

<ul>
<li>Python 3.x</li>
<li>NumPy</li>
</ul>

<h2>Installation</h2>

<ol>
<li>Clone the repository:</li>
<pre><code>git clone https://github.com/lucassdmp/Logistic-Regression-Model.git
cd Logistic-Regression-Model</code></pre>

<h2>Usage</h2>

<p>Execute the script with the following command:</p>
<pre><code>python main.py --learning_rate &lt;your_learning_rate&gt; --iterations &lt;your_iterations&gt;</code></pre>

<p>Replace &lt;your_learning_rate&gt; and &lt;your_iterations&gt; with your desired values.</p>

<h2>Options</h2>

<ul>
<li><code>--learning_rate</code>: Learning rate for gradient descent (default: 0.01)</li>
<li><code>--iterations</code>: Number of iterations for gradient descent (default: 1000)</li>
</ul>

<h2>Dataset</h2>

<p>The Iris dataset is loaded from a CSV file (<code>Iris.csv</code>). The script separates the data into training and testing sets, standardizes features, and then performs logistic regression to predict the iris species.</p>
