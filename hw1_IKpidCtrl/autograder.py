import numpy as np

class Autograder(object):

    def __init__(self) -> None:
        pass

    def testQ1(self, hat, unhat):
        test_vector = np.array([1.,-2.,3.])
        if np.isclose(test_vector, unhat(hat(test_vector))).all():
            print('Passed Debug test')
        else:
            print('did not pass debug test')

autograder = Autograder()