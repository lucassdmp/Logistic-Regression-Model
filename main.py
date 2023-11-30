from Standizer import Standizer
import random
import numpy as np
import argparse

def load_data():
    with open('Iris.csv', 'r') as iris:
        collumns = iris.readline()
        irises = []
        species = []
                
        for line in iris:
            line = (line.strip().split(','))[1:]
            irises.append(np.array(line[:4], dtype=float))
            species.append(line[4])
            
    return irises, species

def encode_species_labels(species):
    label_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    return [label_dict[s] for s in species]

def get_train_and_test_dataset(irises, species, percentage=0.8, random_seed=None):
    length = len(irises)
    train_length = int(length * percentage)

    if random_seed is not None:
        random.seed(random_seed)

    train_indexes = random.sample(range(length), train_length)
    test_indexes = list(set(range(length)) - set(train_indexes))

    train_data = [irises[i] for i in train_indexes]
    train_species = encode_species_labels([species[i] for i in train_indexes])

    test_data = [irises[i] for i in test_indexes]
    test_species = encode_species_labels([species[i] for i in test_indexes])

    return train_data, train_species, test_data, test_species


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    return -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= learning_rate * gradient
    return theta

def predict(X, theta):
    h = sigmoid(X @ theta)
    return np.argmax(h, axis=1)

def main(learning_rate_p, iterations_p, seed_p):
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
        current_species_test = np.array([1 if x == i else 0 for x in Test_species])
        
        current_theta = gradient_descent(Train_set, current_species_train, theta[:, i], learning_rate, iterations)
        theta[:, i] = current_theta
    
    y_pred = predict(Test_set, theta)

    accuracy = np.mean(y_pred == Test_species)
    print("Accuracy:", accuracy)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for gradient descent')
    parser.add_argument('--it', type=int, default=1000, help='Number of iterations for gradient descent')
    parser.add_argument('--sd', type=int, default=random.randint(0,99999), help='Random seed')

    args = parser.parse_args()
    main(args.lr, args.it, args.sd)
