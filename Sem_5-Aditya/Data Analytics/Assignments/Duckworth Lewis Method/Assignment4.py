import os
import math
import pickle
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt


if not os.path.exists('../models'):
    os.makedirs('../models')
if not os.path.exists('../plots'):
    os.makedirs('../plots')


class DLModel:
    """
    Model Class to approximate the Z function as defined in the assignment.
    """

    def __init__(self):
        """Initialize the model."""
        self.Z0 = [None] * 10
        self.L = [None] * 10
    
    def update(self, data):
        """
        Update the L value of the model.
        Takes a weighted average of the L values
        """
        L_new = 0
        for w in range(1, 11):
            temp = data[data['Wickets.in.Hand'] == w]
            L_new += self.L[w-1] * len(temp)
        
        self.L = [L_new / len(data)]
                
    def get_predictions(self, params):
        """
        Get the predictions for the given parameters.
        Also precalculates some values for the loss function.
        """
        preds = np.zeros([10,50])
        precalc = np.zeros([10,50])
        
        Z0 = params[:10]
        L = params[10:] if len(params) > 10 else [self.L[0]]*10
        
        for w in range(10):
            for u in range(50):
                y = Z0[w]*(1 - math.exp(-L[w] * (u+1) /Z0[w]))
                preds[w][u] = y
                precalc[w][u] = (y + 1) * math.log(y + 1) - y
        
        return preds, precalc

    def calculate_loss(self, params, data):
        """ 
        Calculate the loss for the given parameters and datapoints.
        """

        preds, precalc = self.get_predictions(params)

        loss = 0
        for i in range(data.shape[0]):
            w = data['Wickets.in.Hand'].iloc[i]
            u = data['Overs.Remaining'].iloc[i]
            
            y = data['Runs.Remaining'].iloc[i]
            y_pred = preds[w-1][u-1]
            y_precalc = precalc[w-1][u-1]
            
            loss += y_precalc + y - (y_pred + 1) * math.log(y + 1)

        return loss/data.shape[0]
    
    def save(self, path):
        """
        Save the model to the given path.
        """
        with open(path, 'wb') as f:
            pickle.dump((self.L, self.Z0), f)
    
    def load(self, path):
        """
        Load the model from the given path.
        """
        with open(path, 'rb') as f:
            (self.L, self.Z0) = pickle.load(f)


def get_data(data_path):
    """
    Loads the data from the given path and returns a pandas dataframe.
    """
    df = pd.read_csv(data_path)
    return df


def preprocess_data(data):
    """
    Preprocesses the dataframe by
    (i)    considering only the first innings data,
    (ii)   removing rows with missing values,
    (iii)  removing unnecessary columns,
    (iv)   removing rows with overs equal to 50,
    (v)    removing rows with wickets equal to 0,
    (vi)   removing rows with negative runs remaining,
    (vii)  converting overs to overs remaining,
    (viii) adding new datapoints using total score of the match.
    """
    data.dropna(inplace=True)
    data = data[data['Innings'] == 1]
    
    new = data[['Over', 'Wickets.in.Hand', 'Runs.Remaining']]
    new = new[new['Over'] != 50]
    new = new[new['Wickets.in.Hand'] != 0]
    new = new[new['Runs.Remaining'] >= 0]
    
    new['Over'] = 50 - new['Over']
    new.rename(columns={'Over': 'Overs.Remaining'}, inplace=True)
    
    for m in data['Match'].unique():
        temp = data[data['Match'] == m]
        total = temp['Innings.Total.Runs'].values[0]
        new = pd.concat(
            [new, pd.DataFrame([{'Overs.Remaining': 50, 'Wickets.in.Hand': 10, 'Runs.Remaining': total}])],
            ignore_index=True
        )
                
    return new


def train_model(data, model):
    """
    Trains the model
    
    Initially, it trains the model with 10 Z0 values and 10 L values.
    Then, it trains the model with 10 Z0 values, given the L value.
    """
    if len(model.L) > 1:
        # initial guess for z0 and l
        z0 = []
        l = []
        for w in range(1, 11):
            temp = data[data['Wickets.in.Hand'] == w]
            z0.append(np.mean(temp['Runs.Remaining']))
            
            runs = temp['Runs.Remaining'].values
            overs = temp['Overs.Remaining'].values
            l.append(np.median([runs[i]/overs[i] for i in range(len(runs)) if overs[i] != 0]))
        
        # bounds for z0 and l since they are positive
        bounds = [(0, None)]*20
        
        result = sp.optimize.minimize(
            model.calculate_loss,
            x0=z0+l, # optimise over 20 parameters
            args=(data),
            tol=1e-5, # added for faster convergence
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # update the model parameters
        model.Z0 = result.x[:10]
        model.L = result.x[10:]
        
    else:
        # read the initial guess for z0 from model
        z0 = model.Z0
        
        # bounds for z0 since they are positive
        bounds = [(0, None)]*10       
        
        result = sp.optimize.minimize(
            model.calculate_loss,
            x0=z0, # optimise over 10 parameters
            args=(data),
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        # update the model parameters
        model.Z0 = result.x
    
    return model


def plot(model, plot_path):
    """ 
    Plots the model predictions against the number of overs
    remaining according to wickets in hand.
    """
    Z0 = model.Z0
    L = model.L if len(model.L) > 1 else model.L * 10
    
    x = np.linspace(0, 80, 8000)
    
    for i in range(10):
        y = Z0[i]*(1 - np.exp(-L[i] * x / Z0[i]))
        plt.plot(x, y, label=f'Wickets in Hand: {i+1}')
    
    plt.xlabel('Overs remaining')
    plt.ylabel('Average runs obtainable')
    plt.legend(prop={'size': 6.9})

    plt.savefig(plot_path)
    plt.close()


def print_model_params(model):
    """
    Prints the model parameters.
    """
    print("Model Parameters:")
    if len(model.L) > 1:
        for i in range(10):
            print(f'Z0[{i+1}]: {model.Z0[i]}, L[{i+1}]: {model.L[i]}')
    else:
        for i in range(10):
            print(f'Z0[{i+1}]: {model.Z0[i]}')
        print(f'L: {model.L[0]}')
    print()
    

def calculate_loss(model, data):
    """
    Calculates the normalised squared error loss for the given model and data
    """
    loss = 0
    for i in range(data.shape[0]):
        w = data['Wickets.in.Hand'].iloc[i]
        u = data['Overs.Remaining'].iloc[i]
        
        y = data['Runs.Remaining'].iloc[i]
        y_pred = model.Z0[w-1]*(1 - math.exp(-model.L[0] * u / model.Z0[w-1]))    

        loss += (y - y_pred)**2

    return loss/data.shape[0]


def main(args):
    """Main Function"""

    # loading the data
    data = get_data(args['data_path'])
    print("Data loaded.")
    
    # Preprocess the data
    data = preprocess_data(data)
    print("Data preprocessed.\n")
    
    model = DLModel()  # Initializing the model
    print("Model initialized.")
    
    model = train_model(data, model)  # Training the model
    
    model.save(args['model_path1'])  # Saving the model
    plot(model, args['plot_path1'])  # Plotting the model
    print_model_params(model)  # Printing the model parameters
    
    model.update(data) # Updating the model
    print("Model updated.")
    
    model = train_model(data, model) # Retraining the model
    
    model.save(args['model_path2'])  # Saving the new model
    plot(model, args['plot_path2'])  # Plotting the new model
    print_model_params(model)  # Printing the new model parameters  
    
    # Calculate the normalised squared error
    loss = calculate_loss(model, data)
    print(f"Normalised Squared Error Loss: {loss}")
    

if __name__ == '__main__':
    args = {
        "data_path": "../data/04_cricket_1999to2011.csv",
        "model_path1": "../models/22205_model1.pkl",
        "model_path2": "../models/22205_model2.pkl",
        "plot_path1": "../plots/22205_Plot1.png",
        "plot_path2": "../plots/22205_Plot2.png",
    }
     
    main(args)
