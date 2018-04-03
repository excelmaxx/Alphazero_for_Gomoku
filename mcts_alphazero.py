# -*- coding: utf-8 -*-
"""
Monte Carlo Search Tree, tree node class
with more information, more fields

@author: Wen Liang
"""

import numpy as np
import copy
import random
import math

DEBUG = False


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class MCTS(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000, disp=False):

        self.s_root = None
        self._policy = policy_value_fn
        # defulat policy for rollout
        # self.default_polciy =
        #
        self.c_puct = c_puct
        self.n_playout = int(n_playout)
        self.hashtostate = {}
        self._disp = disp

        # check boards
        # the board ends works as a set of all states we have explored

        self.board_ends = {}  # also work as a quick check for the end of game
        # self.valids = {}

        self.Qsa = {}  # Q value for state action pair
        # key: tuple(int hash(numpy-state), int action)
        # value: float q value

        self.Nsa = {}  # N explored for state action pair
        # key: tuple(int hash(numpy-state), int action)
        # value: int number of explored

        self.Ns = {}  # N visited for a state (used a total visited during U value computation)
        # key: hash(numpy-state)
        # value: int number of visited of current state

        self.Ps = {}  # P value from policy
        # keyL hash(numpy-state)
        # value: another dict{action: probability}
        # notice: have already filtered valid actions, have already re-normalized distributions

        # self.s_current = state  # the current state, may be useful

        # other params
        self.EPS = 1.0  # work to avoid divided by 0

    def _playout(self, state):
        """

        :param state: The current state
        :return:
        """
        if DEBUG:
            print('_____playout step_____')
            print state.current_state2()

        board_hash = state.current_state_hash()

        # handle the end of game
        if board_hash not in self.board_ends:
            self.board_ends[board_hash] = state.game_end()
        end, winner = self.board_ends[board_hash]
        # if end:  # means the playout end
        #     if winner == -1: # tie
        #         leaf_value = 0.0
        #     else:
        #         leaf_value = (1.0 if winner == state.get_current_player() else -1)
        if end:  # means the playout end
            if winner == -1:  # tie
                # maybe a solution for frustration
                leaf_value = (-0.5 if state.current_player == state.start_player else 0.9)
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)
            return -leaf_value

        # handle the leaves of the tree
        if board_hash not in self.Ps:
            if DEBUG:
                print "here expanding the leaf"
            action_probs, leaf_value = self._policy(state)
            # filter the actions for available actions
            sum_prob = 0
            for action, prob in action_probs:
                # print action, prob
                if action in state.availables:
                    sum_prob += abs(prob)

            sum_prob = (sum_prob if sum_prob > 0 else 1.0)
            temp_probs = {}
            for action, prob in action_probs:
                temp_probs[action] = prob / sum_prob

            # before return, update current state
            self.hashtostate[board_hash] = state.current_state2().copy()
            self.Ps[board_hash] = temp_probs
            self.Ns[board_hash] = 0.0
            return -leaf_value

        # handle the next step, pick the action with highest upper confidence.
        valid_moves = state.availables
        cur_best = -1000000
        best_action = None

        for a in valid_moves:
            if (board_hash, a) in self.Qsa:
                u = self.Qsa[(board_hash, a)] + self.c_puct * self.Ps[board_hash][a] * math.sqrt(self.Ns[board_hash]) \
                                                / (1 + self.Nsa[(board_hash, a)])
            else:
                u = self.c_puct * self.Ps[board_hash][a] * math.sqrt(self.Ns[board_hash] + self.EPS)

            if u > cur_best:
                cur_best = u
                best_action = a

        if DEBUG:
            for a in valid_moves:
                print('action', a,
                      ' Qsa=', (self.Qsa[(board_hash, a)] if (board_hash, a) in self.Qsa else 0),
                      ' Nsa=', (self.Nsa[(board_hash, a)] if (board_hash, a) in self.Nsa else 0))

        if DEBUG:
            print("the best action for this step:", best_action)

        state.do_move(best_action)
        value = self._playout(state) * 1.0

        if DEBUG:
            print "returned value:", value

        # updating
        self.Ns[board_hash] += 1.0

        if (board_hash, best_action) in self.Qsa:
            n_old = self.Nsa[(board_hash, best_action)]
            self.Qsa[(board_hash, best_action)] = (self.Qsa[(board_hash, best_action)] * n_old + value) \
                                                  / (1 + n_old)
            self.Nsa[(board_hash, best_action)] = n_old + 1
        else:
            # first time
            self.Qsa[(board_hash, best_action)] = value
            self.Nsa[(board_hash, best_action)] = 1.0

        return -value

    def get_move_probs(self, state, temper=1e-3, return_prob=0):

        for n in range(self.n_playout):
            if DEBUG:
                print(n, " playout begin!")
            state_temp = copy.deepcopy(state)
            self._playout(state_temp)
            if DEBUG:
                print("final result for number", n, "playout:")
                print(state_temp.current_state2())

        root_hash = state.current_state_hash()
        acts = state.availables
        act_probs = []

        if len(acts) == 0 and DEBUG:
            print "WARNING!!! fulled, but should not come to here"
            return [], []

        for a in acts:
            if (root_hash, a) in self.Nsa:
                act_probs.append(self.Nsa[(root_hash, a)])
            else:
                act_probs.append(return_prob)
        act_probs2 = softmax(1.0 / temper * np.log(np.array(act_probs) + 1e-10))

        if DEBUG:
            print zip(acts, act_probs2, act_probs)

        if self._disp:
            return acts, act_probs2, np.log(act_probs)
        else:
            return acts, act_probs2

    def update_with_move(self, last_move):
        print "please use update with state"
        raise Exception("update_with_move  not supported right now, use update with state()")

    def update_with_state(self, new_state):
        self.s_root = new_state


class MCTSPlayer(object):
    def __init__(self, policy_value_fn, c_puct=5, n_playout=500, is_selfplay=0, disp=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout, disp)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._is_selfplay = is_selfplay
        self._disp = disp

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):

        # print "reset_player function still in test"
        # raise Exception('reset_player not supported right now, not needed actually')
        self.mcts = MCTS(self._policy, self._c_puct, self._n_playout, self._disp)

    def get_action(self, board, temp=1e-3, return_prob=0):
        valid_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width * board.height)
        if len(valid_moves) > 0:
            if not self._disp:
                acts, probs = self.mcts.get_move_probs(board, temp)
            else:
                acts, probs, probs_orig = self.mcts.get_move_probs(board, temp)
            move_probs[acts] = probs
            if self._is_selfplay:
                # add Dirichlet Noise for exploration (needed for
                # self-play training)
                # Add Noise for training exploration
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                # self.mcts.update_with_state()
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                # for playing with human, just choose the one with highest probability is okay
                if self._disp:
                    print zip(acts, probs_orig)
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.reset_player()
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)



