import numpy as np # import Numpy library to generate

# Initialisation des variables

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases

#on imprime les poids et les biais

print(weights)
print(biases)

#Calcul de la sortie pour une entrée donnée x1 et x2

x_1 = 0.5 # input 1
x_2 = 0.85 # input 2
print('x1 is {} and x2 is {}'.format(x_1, x_2))

# Valeur du premier noeud de la couche cachee calcul de la somme pondérée des entrées z1,1

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]
print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

#  Valeur du deuxieme noeud de la couche cachee

z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1] #Question 1
print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(z_12))

# Valeur d'activation du premier noeud de la couche cachee

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

# Valeur d'activation du deuxieme noeud de la couche cachee

a_12 = 1.0 / (1.0 + np.exp(-z_12))  #Question 2
print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

# Valeur du noeud de la couche de sortie

z_2 = a_12 * weights[5] + a_11 * weights[4] + biases[2] #Question 3
print('The weighted sum of the inputs at the exit lawyer is {}'.format(z_2))

# Valeur d'activation du noeud de la couche de sortie

a_2 = 1.0 / (1.0 + np.exp(-z_2)) #Question 4
print('The activation of the node in the output layer is {}'.format(np.around(a_2, decimals=4)))
