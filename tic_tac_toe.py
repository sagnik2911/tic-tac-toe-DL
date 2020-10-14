import sys

import pandas as pd
import numpy as np
import classifiers_and_regressors
import pickle


class PlayTictactoe():
    
    def __init__(self,model,type,board):
        self.model = model
        self.type = type
        print("\nWelcome to ML Powered TIC-TAC-TOE \n")
        print("Here is a positional reference board for you:")
        self.display_board(["x1",'x2','x3','x4','x5','x6','x7','x8','x9'])
        
        print("YOUR CURRENT BOARD:")
        self.display_board(board)
        print(self.start_game(board))

    #Function to display the board
    def display_board(self,board):
        di = {-1:"-1",0:"0",+1:"+1","x1":"x1","x2":"x2","x3":"x3","x4":"x4","x5":"x5","x6":"x6","x7":"x7","x8":"x8","x9":"x9"}
        print(di[board[0]] + '|' + di[board[1]] + '|' + di[board[2]])
        print(di[board[3]] + '|' + di[board[4]] + '|' + di[board[5]])
        print(di[board[6]] + '|' + di[board[7]] + '|' + di[board[8]])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    def start_game(self,board):
        while(board.count(0) !=0):

            #COMPUTER'S MOVE
            if (self.type == "default") :
                y_pred = self.model.predict(np.asarray(board).reshape(1,-1))

                df = pd.DataFrame(y_pred,columns=['y0','y1','y2','y3','y4','y5','y6','y7','y8'])
                df[df > 0.5] = 1
                df[df <= 0.5] = 0
                available_moves = [i for i, val in enumerate(df.to_numpy()[0]) if val == 1]
            else:
                y_pred_test = np.empty((1, 9))
                board_biased = np.append(np.asarray(board).reshape(1,-1), np.ones([1, 1]), 1)
                for col in range(9):
                    y_pred_test_col = np.dot(board_biased, model[col])
                    y_pred_test[:, col] = y_pred_test_col
                y_pred = np.where(y_pred_test < 0.5, 0, 1)
                df = pd.DataFrame(y_pred,columns=['y0','y1','y2','y3','y4','y5','y6','y7','y8'])
                available_moves = [i for i, val in enumerate(df.to_numpy()[0]) if val == 1]
                if (len(available_moves) == 0) :
                    available_moves = [i for i in range(9)]
            computers_move = np.random.choice(a = np.array(available_moves))
            board[computers_move] = -1
            print('COMPUTERS TURN')
            self.display_board(board)

            #Check if anyone won
            if ((board[0],board[1],board[2]) == (-1,-1,-1) or (board[3],board[4],board[5]) == (-1,-1,-1)  or (board[6],board[7],board[8]) == (-1,-1,-1) or (board[0],board[3],board[6]) == (-1,-1,-1) or (board[1],board[4],board[7]) == (-1,-1,-1) or (board[2],board[5],board[8]) == (-1,-1,-1) or (board[0],board[4],board[8]) == (-1,-1,-1) or (board[2],board[4],board[6]) == (-1,-1,-1)): 
                return("Computer won")
                
            if ((board[0],board[1],board[2]) == (1,1,1) or (board[3],board[4],board[5]) == (1,1,1)  or (board[6],board[7],board[8]) == (1,1,1) or (board[0],board[3],board[6]) == (1,1,1) or (board[1],board[4],board[7]) == (1,1,1) or (board[2],board[5],board[8]) == (1,1,1) or (board[0],board[4],board[8]) == (1,1,1) or (board[2],board[4],board[6]) == (1,1,1) ): 
                return("Human won")
                
            if board.count(0) == 0 :
                break

            #HUMAN's MOVE
            humans_move = input("YOUR TURN ! - Enter cell number to play: ")
            di_positions = {'x1':0,'x2':1,'x3':2,'x4':3,'x5':4,'x6':5,'x7':6,'x8':7,'x9':8}
            board[di_positions[humans_move]] = 1
            self.display_board(board)


            #Check if anyone won
            if ((board[0],board[1],board[2]) == (-1,-1,-1) or (board[3],board[4],board[5]) == (-1,-1,-1)  or (board[6],board[7],board[8]) == (-1,-1,-1) or (board[0],board[3],board[6]) == (-1,-1,-1) or (board[1],board[4],board[7]) == (-1,-1,-1) or (board[2],board[5],board[8]) == (-1,-1,-1) or (board[0],board[4],board[8]) == (-1,-1,-1) or (board[2],board[4],board[6]) == (-1,-1,-1)): 
                return ("Computer won")
                break
            if ((board[0],board[1],board[2]) == (1,1,1) or (board[3],board[4],board[5]) == (1,1,1)  or (board[6],board[7],board[8]) == (1,1,1) or (board[0],board[3],board[6]) == (1,1,1) or (board[1],board[4],board[7]) == (1,1,1) or (board[2],board[5],board[8]) == (1,1,1) or (board[0],board[4],board[8]) == (1,1,1) or (board[2],board[4],board[6]) == (1,1,1) ): 
                return ("Human won")
                
        return "GAME DRAW"



if __name__ == "__main__":
    """Extracting the board from the command line input.
       if board provided as input, pass this board.
       else pass empty board for new game
    """
    regressors = classifiers_and_regressors.Regressors()
    type = "default"
    while(True):
            #Play Game
        if len(sys.argv[1:]) > 0:
            board = sys.argv[1:][0]
            board = [int(x) for x in board.split(",")]
            print("\nTraining regression models for playing the game (AI)")
            with open("Trained_MLP_D3.pkl", 'rb') as file:
                model = pickle.load(file)
            game = PlayTictactoe(model,type,board)
        else:
            model = None
            i = int(input(
                "Enter AI Choice:\nEnter 1 for MLP:\nEnter 2 for KNR:\nEnter 3 for Linear Regression \nEnter: "))
            print("\nTraining regression models for playing the game (AI)")
            if (i == 1):
                with open("Trained_MLP_D3.pkl", 'rb') as file:
                    model = pickle.load(file)
            elif (i == 2):
                with open("Trained_KNR_D3.pkl", 'rb') as file:
                    model = pickle.load(file)
            elif (i == 3):
                with open("Trained_Linearreg_D3.txt", 'rb') as file:
                    model = pickle.load(file)
                type = "custom"
            else:
                print("Error Input \n");
                break
            game = PlayTictactoe(model,type,[0,0,0,0,0,0,0,0,0])