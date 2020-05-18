import numpy as np
import csv

class NeuralNetwork():




    #TODOLIST
    # - Need a dataset that has 4 yes/no datapoints and a result (likeliness to succeed, predicted income by factors...)
    # - Change number of variables to match a CSV dataset
    # - Predict something real
    #
    #
    
    def __init__(self):
        
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and
        # mean of 0
        self.weights =  2 * np.random.random((3, 1)) - 1

    

    def sigmoidFunction(self, input):
        # 1 over 1 + e to the negative input
        return 1 / (1 + np.exp(-input))

    def getSigmoidFunctionDerivative(self, input):
        return (1-input) * input

    def train(self, trainingInput, trainingOutput):

        
        for iteration in range(100):
            
            guessedOutput = self.think(trainingInput) # Get result with existing coefficients

            error = trainingOutput - guessedOutput # See how far this is off from what we expected

            adjustments = np.dot(trainingInput.T, error * self.getSigmoidFunctionDerivative(guessedOutput))

            self.weights += adjustments


    def think(self, input):
        
        input = input.astype(float)
        output = self.sigmoidFunction(np.dot(input, self.weights))
        return output
        

        





if __name__ == "__main__":

    neural_network = NeuralNetwork()

    # load csv file
    f = open('LondonOlympicsDataCSV.csv')
    csvFile = csv.reader(f)

    inputData = np.zeros((568, 3)) # These need to be between 0 and 1...
    outputData = np.zeros((568))
    p=0
    for row in csvFile:

        if (p != 0):
            inputData[p-1][0] = row[0]
            inputData[p-1][1] = row[1]
            inputData[p-1][2] = row[2]

            outputData[p-1] = row[3]
        
        p+=1

    outputData = np.array([outputData]).T
    print(outputData)
    

        

    # Normalize inputData
    i=0
    for iteration in range(len(inputData)):
        inputData[i][0] = (float(inputData[i][0]) - 15.0) / 23.0
        inputData[i][1] = (float(inputData[i][1]) - 160.0) / 61.0
        inputData[i][2] = (float(inputData[i][2]) - 50.0) / 80.0
        i+=1
    

    neural_network.train(inputData, outputData)

    print("Weights after training: ")
    print(neural_network.weights)

    

    inputOne = str(input("Age: "))
    inputTwo = str(input("Height: "))
    inputThree = str(input("Weight: "))

    inputOne = (float(inputOne) - 15.0) / 23.0
    inputTwo = (float(inputTwo) - 160.0) / 61.0
    inputThree = (float(inputThree) - 50.0) / 80.0

    print("Outcome: ")
    print(neural_network.think(np.array([inputOne, inputTwo, inputThree])))
    

    
                                
