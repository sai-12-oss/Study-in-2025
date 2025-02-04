from scipy.linalg import block_diag

# this class characterizes an automaton
class FSA:
    def __init__ (self, numStates = 0, startStates=None, finalStates=None, alphabetTransitions=None) :
        self.numStates = numStates
        self.startStates = startStates
        self.finalStates = finalStates
        self.alphabetTransitions = alphabetTransitions

class NFA(FSA):
    def simulate(self, ipStr):
        S = set(self.startStates)
        newS = set()
        for i in range(len(ipStr)):
            symbol = ipStr[i]
            tm = self.alphabetTransitions[symbol]
            for state in S:
                trs = tm[state]
                for tr in range(len(trs)):
                    if trs[tr] == 1:
                        newS.add(tr)
            S = set(newS)
            newS = set()
        if len(self.finalStates) > 0 and not S.isdisjoint(self.finalStates):
            print("String Accepted")
            return True
        else:
            print("String Rejected")
            return False

    def getNFA(self):
        return self

class ETree:
    root = None
    nfa = None
    class ETNode:
        def __init__(self, val=" ", left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def compute(self, operands, operators):
            operator = operators.pop()
            if operator == "*":
                left = operands.pop()
                operands.append(self.ETNode(val=operator, left=left))
            elif operator == "+":
                right, left = operands.pop(), operands.pop()
                operands.append(self.ETNode(val=operator, left=left, right=right))
            elif operator == ".":
                right, left = operands.pop(), operands.pop()
                operands.append(self.ETNode(val=operator, left=left, right=right))

    def parseRegex(self, regex):
        operands, operators = [], []
        for i in range(len(regex)):
            if regex[i].isalpha():
                operands.append(self.ETNode(val=regex[i]))
            elif regex[i] == '(':
                operators.append(regex[i])
            elif regex[i] == ')':
                while operators[-1] != '(':
                    self.compute(operands, operators)
                operators.pop()
            else :
                operators.append(regex[i])
        while operators:
            self.compute(operands, operators)

        if len(operators) == 0:
            self.root = operands[-1]
        else :
            print("Parsing Regex failed.")

    def getTree(self):
        return self.root

    ###################################################################
    # IMPLEMENTATION STARTS AFTER THE COMMENT
    # Implement the following functions

    # In the below functions to be implemented delete the pass statement
    # and implement the functions. You may define more functions according
    # to your need.
    ###################################################################
    # .
    def operatorDot(self, fsaX, fsaY):
        # to take dot, add transitions from prefinal state of fsaX to start state of fsaY
        fsa = FSA()
        fsa.numStates = fsaX.numStates + fsaY.numStates  # total states is the sum of the two nfa's

        fsa.startStates = fsaX.startStates  # start states are the start states of fsaX
        if not fsaX.startStates.isdisjoint(fsaX.finalStates):  # if fsaX accepts e, add start states of fsaY
            for i in fsaY.startStates: 
                fsa.startStates.add(i + fsaX.numStates)

        fsa.finalStates = set()  # final states are the final states of fsaY
        for i in fsaY.finalStates:
            fsa.finalStates.add(i + fsaX.numStates)

        letters = ["a","b","c","e"]
        fsa.alphabetTransitions = {}

        for letter in letters:
            if letter in fsaX.alphabetTransitions.keys():  # transition for letter in fsaX 
                A = fsaX.alphabetTransitions[letter]
            else:
                A = [[0]*fsaX.numStates for i in range(fsaX.numStates)]

            if letter in fsaY.alphabetTransitions.keys():  # transition for letter in fsaY
                B = fsaY.alphabetTransitions[letter]
            else:
                B = [[0]*fsaY.numStates for i in range(fsaY.numStates)]

            C = block_diag(A, B)  # combine the transitions along diagonal
            fsa.alphabetTransitions[letter] = C

            for i in fsaX.finalStates:
                for j in range(fsaX.numStates):
                    if fsa.alphabetTransitions[letter][j][i] == 1:  # transitions to final states of fsaX
                        for k in fsaY.startStates:
                            fsa.alphabetTransitions[letter][j][k + fsaX.numStates] = 1  # add them to start states of fsaY

        return fsa
    
    # +
    def operatorPlus(self, fsaX, fsaY):
        # to take plus, run the two nfa's in parallel 
        fsa = FSA()
        fsa.numStates = fsaX.numStates + fsaY.numStates  # total states is the sum of the two nfa's
        
        fsa.startStates = fsaX.startStates
        for i in fsaY.startStates:
            fsa.startStates.add(i + fsaX.numStates)  # start states are the start states of fsaX and fsaY
        
        fsa.finalStates = fsaX.finalStates
        for i in fsaY.finalStates:
            fsa.finalStates.add(i + fsaX.numStates)  # final states are the final states of fsaX and fsaY
        
        letters = ["a","b","c","e"]
        fsa.alphabetTransitions = {}

        for letter in letters:
            if letter in fsaX.alphabetTransitions.keys():  # transition for letter in fsaX
                A = fsaX.alphabetTransitions[letter]
            else:
                A = [[0]*fsaX.numStates for i in range(fsaX.numStates)]

            if letter in fsaY.alphabetTransitions.keys():  # transition for letter in fsaY
                B = fsaY.alphabetTransitions[letter]
            else:
                B = [[0]*fsaY.numStates for i in range(fsaY.numStates)]

            C = block_diag(A, B)  # combine the transitions along diagonal
            fsa.alphabetTransitions[letter] = C

        return fsa
                
    # *
    def operatorStar(self, fsaX):
        # We follow the Glushkov construction
        # to take star, first add a new state 
        # add transitions from new state to the successors of the start states of fsaX 
        # add transitions from prefinal states of fsaX to the new state
        # finally, make the new state the only start and final state
        fsa = FSA()
        fsa.numStates = fsaX.numStates + 1
        fsa.startStates = set([fsaX.numStates])  # new state is the only start state
        fsa.finalStates = set([fsaX.numStates])  # new state is the only final state

        letters = fsaX.alphabetTransitions.keys()
        fsa.alphabetTransitions = {}

        for letter in letters:
            if letter in fsaX.alphabetTransitions.keys():  # transition for letter in fsaX
                A = fsaX.alphabetTransitions[letter]
            else:
                A = [[0]*fsaX.numStates for i in range(fsaX.numStates)]

            B = [[0]]  # empty transition for new state
            C = block_diag(A, B)  # combine the transitions along diagonal
            fsa.alphabetTransitions[letter] = C

            for i in fsaX.startStates:  # transitions from new state to the successors of the start states of fsaX
                for j in range(fsaX.numStates):
                    if fsa.alphabetTransitions[letter][i][j] == 1:
                        fsa.alphabetTransitions[letter][fsaX.numStates][j] = 1

            for i in fsaX.finalStates:  # transitions from prefinal states of fsaX to the new state
                for j in range(fsa.numStates):
                    if fsa.alphabetTransitions[letter][j][i] == 1:
                        fsa.alphabetTransitions[letter][j][fsaX.numStates] = 1

        return fsa

    # a, b, c and e for epsilon
    def alphabet(self, symbol):
        # make simplest nfa to accept the symbol
        fsa = FSA()

        if symbol == "e":  # epsilon requires only one state
            fsa.numStates = 1
            fsa.startStates = set([0])
            fsa.finalStates = set([0])
            fsa.alphabetTransitions = {"e": [[1]]}  # epsilon transition to final state

        else:
            fsa.numStates = 2  # for a, b, c, we need two states
            fsa.startStates = set([0])
            fsa.finalStates = set([1])
            fsa.alphabetTransitions = {symbol: [[0,1],[0,0]]}  # transition from start to final state
        
        return fsa

    # Traverse the regular expression tree(ETree)
    # calling functions on each node and hence
    # building the automaton for the regular
    # expression at the root.

    def helperNFA(self, root):  # helper function to traverse the regex tree recursively
        if root.val == ".":  # if the root is operator dot
            return self.operatorDot(self.helperNFA(root.left), self.helperNFA(root.right))
        elif root.val == "+":  # if the root is operator plus
            return self.operatorPlus(self.helperNFA(root.left), self.helperNFA(root.right))
        elif root.val == "*":  # if the root is operator star
            return self.operatorStar(self.helperNFA(root.left))
        else:  # if the root is a symbol
            return self.alphabet(root.val)

    def buildNFA(self, root):
        if root == None:
            print("Tree not available")
            exit(0)

        numStates = 0
        initialState = set()
        finalStates = set()
        transitions = {}

        # write code to populate the above datastructures for a regex tree
        nfa = self.helperNFA(root)  # get the nfa from the regex tree using the helper function

        # get the details of the nfa constructed above
        numStates = nfa.numStates  
        initialState = nfa.startStates
        finalStates = nfa.finalStates
        transitions = nfa.alphabetTransitions

        self.nfa = NFA(numStates, initialState, finalStates, transitions)

        # print NFA
        print("NFA for the regular expression: ")
        print("Number of states: " + str(self.nfa.numStates))
        print("Start states: " + str(self.nfa.startStates))
        print("Final states: " + str(self.nfa.finalStates))
        print("Transitions: ")
        for key, value in self.nfa.alphabetTransitions.items():
            print(key, ":", value)
        
        return self.nfa

    ######################################################################