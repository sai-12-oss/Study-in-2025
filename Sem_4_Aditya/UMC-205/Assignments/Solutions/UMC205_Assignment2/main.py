import utils as ut
import unittest

class TestMembership(unittest.TestCase):

    def test_stringMembership(self):
        with open("input.txt", "r") as inputFile:
            numRegex = inputFile.readline()
            expOutput = open("expOutput.txt", "r")

            for i in range(int(numRegex)):
                regex = inputFile.readline()
                print("Building automaton for regex : " + regex)

                eTree = ut.ETree()
                eTree.parseRegex(regex.rstrip())
                root = eTree.getTree()
                if  root != None:
                    nfa = eTree.buildNFA(root)

                numTest = inputFile.readline()
                for i in range(int(numTest)):
                    expOp = expOutput.readline()
                    test = inputFile.readline()
                    if  nfa.simulate(test.rstrip()):
                        print("Checking : " + test.rstrip())
                        self.assertEqual("Yes", expOp.rstrip())
                        print(test.rstrip() + " belongs to the language")
                    else:
                        print("Checking : " + test.rstrip())
                        self.assertEqual("No", expOp.rstrip())
                        print(test.rstrip() + " does not belong to the language")

        inputFile.close()
        expOutput.close()

# __main__ module is invoked when the script is invoked at the commandline
if __name__ == "__main__":
    unittest.main()