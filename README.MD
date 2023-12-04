<h1>Logistic Regression with Gradient Descent</h1>

<p>This script performs logistic regression using gradient descent on the Iris dataset. It predicts the species of iris flowers, considering three classes: Iris-setosa, Iris-versicolor, and Iris-virginica.</p>

<h2>Prerequisites</h2>

<ul>
<li>Python 3.x</li>
</ul>

<h2>Installation</h2>

<ol>
<li>Clone the repository:</li>
<pre><code>git clone https://github.com/lucassdmp/Logistic-Regression-Model.git
cd Logistic-Regression-Model</code></pre>

<h2>Usage</h2>

<p>Execute the script with the following command:</p>

(All arguments are optional)
<pre><code>python main.py --sd &lt;seed_to_randomize_data&gt; --lr &lt;your_learning_rate&gt; --it &lt;your_iterations&gt; --csvn &lt;csv_file_name_for_results&gt;</code></pre>

<p>Replace &lt;seed_to_randomize_data&gt;, &lt;your_learning_rate&gt;, &lt;your_iterations&gt; and &lt;csv_file_name_for_results&gt; with your desired values.</p>

<h2>Options</h2>

<ul>
  <li><code>--sd</code>: Seed for randomization during data selection (default: random)</li>
  <li><code>--lr</code>: Learning rate for gradient descent (default: 0.01)</li>
  <li><code>--it</code>: Number of iterations for gradient descent (default: 1000)</li>
  <li><code>--csvn</code>: Name for the CSV file containing results (default: Seed-Learing_rate-Iterations.csv)</li>
</ul>

<h2>Dataset</h2>

<p>The Iris dataset is loaded from a CSV file (<code>Iris.csv</code>). The script separates the data into training and testing sets, standardizes features, and then performs logistic regression to predict the iris species.</p>
