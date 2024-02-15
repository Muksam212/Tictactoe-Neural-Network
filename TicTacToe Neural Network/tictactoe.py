from tictactoe_model import TicTacToeModel

from game import play, simulate

from players import NNPlayer

from tictactoe_state import TicTacToeState

def tictactoe_NN(simulations = 10000, epochs = 3):
	model = TicTacToeModel()
	state = TicTacToeState() #board configuration 
	
	plays = simulate(state, simulations)
	model.train(plays, epochs = epochs) #complete model to use
	
	#testing the model against human player
	autoplayer = NNPlayer(model)
	
	from tictactoe_window import TicTacToeWindow
	TicTacToeWindow(autoplayer = autoplayer).show()
	
if __name__ == "__main__":
	tictactoe_NN()
