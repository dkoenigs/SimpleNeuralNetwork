import numpy as np

class NeuralNetwork():




    #TODOLIST
    #
    # - Add to my github
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

        
        for iteration in range(1000):
            
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

    inputData = np.array([[1,0,1],[0,1,0],[1,1,1],[0,0,0]])
    outputData = np.array([[1,0,1,0]]).T

    neural_network.train(inputData, outputData)

    print("Weights after training: ")
    print(neural_network.weights)

    inputOne = str(input("Input One: "))
    inputTwo = str(input("Input Two: "))
    inputThree = str(input("Input three: "))

    print("Outcome: ")
    print(neural_network.think(np.array([inputOne, inputTwo, inputThree])))
    

    
                                
