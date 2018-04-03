# Alphazero_for_Gomoku

* Implemented Alphazero model for Gomoku game
* Also tried different UCBs for balancing exploration and exploitation
* Model 664, 6 by 6 board, 4 in a row; Model 995, 9 by 9 board, 5 in a row
* Metrics: compare with Monte Carlo Search Tree with much more palyouts (664 game, alphazero mcts with 400 playouts can compete with mcts with 2400 playouts)

Residual network still in progress

A visualization example for self-playing (heatmap represents the distribution, black patch means current decision)


![self-play](https://media.giphy.com/media/fQAB9htawlyyI7uyeC/giphy.gif)
