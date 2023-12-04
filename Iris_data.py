import random
import numpy as np

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
