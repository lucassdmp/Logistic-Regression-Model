import random
import numpy as np
import argparse
import csv

from Standizer import Standizer
from logistic_prog import gradient_descent, predict
from Iris_data import load_data, get_train_and_test_dataset

def main(learning_rate_p, iterations_p, seed_p, csv_name):
    if csv_name == "":
        csv_name = str(seed_p)+"-"+str(learning_rate_p)+"-"+str(iterations_p)+".csv"
    
    irises, species = load_data()

    Train_set, Train_species, Test_set, Test_species = get_train_and_test_dataset(irises, species, random_seed=seed_p)

    standizer = Standizer()
    Train_set = standizer.fit_transform(Train_set)
    Test_set = standizer.transform(Test_set)

    Train_set = np.hstack((np.ones((len(Train_set), 1)), Train_set))
    Test_set = np.hstack((np.ones((len(Test_set), 1)), Test_set))

    theta = np.zeros((Train_set.shape[1], len(set(Train_species))))

    learning_rate = learning_rate_p
    iterations = iterations_p

    for i in range(len(set(Train_species))):
        current_species_train = np.array([1 if x == i else 0 for x in Train_species])

        current_theta = gradient_descent(Train_set, current_species_train, theta[:, i], learning_rate, iterations)
        theta[:, i] = current_theta

    y_pred = predict(Test_set, theta)

    correct_predictions = y_pred == Test_species

    with open(csv_name, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Test Case', 'True Species', 'Predicted Species', 'Result'])

        for i, (is_correct, true_species, pred_species) in enumerate(zip(correct_predictions, Test_species, y_pred), start=1):
            species_str = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
            result = 'Correct' if is_correct else 'Incorrect'
            writer.writerow([i, species_str[true_species], species_str[pred_species], result])

    accuracy = np.mean(correct_predictions)
    print("Results saved to", csv_name)
    print("Accuracy:", accuracy)

if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for gradient descent')
    parser.add_argument('--it', type=int, default=1000, help='Number of iterations for gradient descent')
    parser.add_argument('--sd', type=int, default=random.randint(0, 99999), help='Random seed')
    parser.add_argument('--csvn', type=str, default="",help='Output CSV file name')

    args = parser.parse_args()
    main(args.lr, args.it, args.sd, args.csvn)