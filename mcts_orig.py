# -*- coding: utf-8 -*-
"""
An corrected implementation of mcts with rollout

@author: wen Liang
"""

import numpy as np
import copy
import math
from operator import itemgetter

DEBUG = False


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


def rollout_policy_fn1(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    if DEBUG:
        print "rollout policy 1, random"
    # rollout randomly
    action_probs = np.random.rand(len(board.availables))
    # then normalize
    action_probs = action_probs / np.sum(action_probs)
    return zip(board.availables, action_probs)


def policy_value_fn_naive(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    if DEBUG:
        print "tree policy naive: all 1 / n"
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


# It would be better to use some more sophisticated rollout policy function and policy value functions
# Here I just use random.


class MCTS(object):
    def __init__(self, policy_value_fn, rollout_policy_fn, c_puct=5, n_playout=2000, disp=False):

        self.s_root = None
        # for alpha go 2016, the policy network and value network are 2 net works
        # here, you can use a policy network or a combined network
        self._policy = policy_value_fn
        self._rollout_policy = rollout_policy_fn
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
        if end:  # means the playout end
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)
            return (-leaf_value)

        # handle the leaves of the tree
        if board_hash not in self.Ps:
            if DEBUG:
                print "here expanding the leaf"
            action_probs, _ = self._policy(state)
            # filter the actions for available actions
            sum_prob = 0
            for action, prob in action_probs:
                if action in state.availables:
                    sum_prob += abs(prob)

            sum_prob = (sum_prob if sum_prob > 0 else 1.0)
            temp_probs = {}
            for action, prob in action_probs:
                if action in state.availables:
                    temp_probs[action] = prob / sum_prob

            if DEBUG:
                'came into the rollout part'
            leaf_value = self.evaluate_rollout(copy.deepcopy(state)) * 1.0
            if DEBUG:
                print 'the rollout end, the rollout evaluate for this state:', leaf_value
            # before return, update current state
            self.hashtostate[board_hash] = state.current_state2().copy()
            self.Ps[board_hash] = temp_probs
            self.Ns[board_hash] = 0.0
            return -leaf_value * 1.0

        # handle the next step, pick the action with highest upper confidence.
        valid_moves = state.availables
        cur_best = -1000000
        best_action = None

        # still use ucb here
        for a in valid_moves:
            if (board_hash, a) in self.Qsa:
                u = self.Qsa[(board_hash, a)] + self.c_puct * self.Ps[board_hash][a] * math.sqrt(self.Ns[board_hash] + self.EPS) \
                                                / (1 + self.Nsa[(board_hash, a)])
            else:
                u = self.c_puct * self.Ps[board_hash][a] * math.sqrt(self.Ns[board_hash] + self.EPS)
            # print 'score for each pair:', (a, u)
            if u > cur_best:
                cur_best = u
                best_action = a

        if DEBUG:
            print 'The best action I took:', best_action
            print 'Before'
            for a in valid_moves:
                print('action', a,
                      ' Qsa=', (self.Qsa[(board_hash, a)] if (board_hash, a) in self.Qsa else 0.0),
                      ' Nsa=', (self.Nsa[(board_hash, a)] if (board_hash, a) in self.Nsa else 0.0))

        if DEBUG:
            print("the best action for this step:", best_action)

        state.do_move(best_action)
        value = self._playout(state) * 1.0

        if DEBUG:
            print "returned value:", value

        # updating
        self.Ns[board_hash] += 1.0

        # print
        if (board_hash, best_action) in self.Qsa:
            n_old = self.Nsa[(board_hash, best_action)]
            self.Qsa[(board_hash, best_action)] = (self.Qsa[(board_hash, best_action)] * n_old + value * 1.0) \
                                                  / (1 + n_old)
            self.Nsa[(board_hash, best_action)] = n_old + 1.0
        else:
            # first time

            self.Qsa[(board_hash, best_action)] = value
            self.Nsa[(board_hash, best_action)] = 1.0

        if DEBUG:
            print 'after the updated!!!!!!!'
            for a in valid_moves:
                print('action', a,
                      ' Qsa=', (self.Qsa[(board_hash, a)] if (board_hash, a) in self.Qsa else 0),
                      ' Nsa=', (self.Nsa[(board_hash, a)] if (board_hash, a) in self.Nsa else 0))

        return -value

    def evaluate_rollout(self, state, limit=1000):
        """
        Use default policy to rollout until the end of the game,
        return +1 if the current player wins, -1 if lose
        return 0 if it is a tie.
        """
        player = state.get_current_player()

        for i in range(limit):
            end, win = state.game_end()
            if end:
                break
            action_probs = self._rollout_policy(state)
            if DEBUG:
                print "action_probs:", action_probs
            action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(action)
            if DEBUG:
                print state.current_state2()
        else:
            print("warning: rollout reaches move limit")
            return 0
        if win == -1:
            return 0
        else:
            # print win, player
            return 1 if win == player else -1

    def get_move(self, state, temper=1e-3, return_prob=0):
        for n in range(self.n_playout):
            if DEBUG:
                print(n, " playout begin!")
            state_temp = copy.deepcopy(state)
            self._playout(state_temp)
            if DEBUG:
                print("final result for number", n, "playout:")
                print(state_temp.current_state2())

        root_hash = state.current_state_hash()
        acts = []
        act_probs = []

        if len(state.availables) == 0 and DEBUG:
            print "WARNING!!! fulled, but should not come to here"
            return []

        for a in state.availables:
            if (root_hash, a) in self.Nsa:
                # print self.Nsa[(root_hash, a)], a
                act_probs.append(self.Nsa[(root_hash, a)])
                acts.append(a)

        act2_nsa = zip(acts, act_probs)
        if DEBUG:
            print act2_nsa
        # print act2_nsa

        return max(act2_nsa, key=itemgetter(1))[0]

    def update_with_move(self, last_move):
        print "please use update with state"
        raise Exception("update_with_move  not supported right now, use update with state()")

    def update_with_state(self, new_state):
        self.s_root = new_state


class MCTSPlayer(object):

    def __init__(self, policy_value_fn=policy_value_fn_naive, rollout_policy_fn=rollout_policy_fn1, c_puct=5, n_playout=500, is_selfplay=0, disp=False):
        # self, policy_value_fn, rollout_policy_fn, c_puct=5, n_playout=2000, disp=False
        self.mcts = MCTS(policy_value_fn=policy_value_fn,
                         rollout_policy_fn=rollout_policy_fn,
                         c_puct=c_puct,
                         n_playout=n_playout,
                         disp=disp)
        self._policy = policy_value_fn
        self._rollout_policy = rollout_policy_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._is_selfplay = is_selfplay
        self._disp = disp

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        # print "reset_player function still in test"
        # raise Exception('reset_player not supported right now, not needed actually')
        self.mcts = MCTS(policy_value_fn=self._policy,
                         rollout_policy_fn=self._rollout_policy,
                         c_puct=self._c_puct,
                         n_playout=self._n_playout,
                         disp=self._disp)

    def get_action(self, board, temp=1e-3, return_prob=0):
        valid_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        assert return_prob == 0 # currently not support return prob
        if len(valid_moves) > 0:
            if not self._disp:
                move = self.mcts.get_move(board)
            else:
                move, probs_orig = self.mcts.get_move(board)
                self.reset_player()

            if self._is_selfplay:
                pass
                # update the root node and reuse the search tree
                # self.mcts.update_with_state()
            else:   # human play
                if self._disp:
                    print "the Probs orig result:"
                    print zip(probs_orig)
                # reset the root node
                self.reset_player()
                # location = board.move_to_location(move)
                # print("AI move: %d,%d\n" % (location[0], location[1]))

            return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

