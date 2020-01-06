#!/usr/bin/env python2.7
"""
mp4.py
"""

import argparse, random
from math import floor
import math
import pygame

# Globals describing board in discrete terms
BOARD_HEIGHT = 12.0
BOARD_WIDTH = 12.0
PLAYER1_PADDLE_HEIGHT = 0.2
PLAYER2_PADDLE_HEIGHT = 0.2
ALPHA = 0.2
GAMMA = 0.85
Ne = 20.0
EC = 0.01
Q = {}
N = {}
S = None
A = None
R = None
T = 0.0
C = 0.3

class State:
    def __init__(self, ball_x, ball_y, velocity_x, velocity_y, paddle1_y, paddle2_y):
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.paddle_y = [paddle1_y, paddle2_y]
    
    def discretize(self):
        """ Return a discretized state representation as a 5-tuple """
        (ball_x_d, ball_y_d) = discretize_position(self)
        velocity_x_d = -1 if self.velocity_x < 0.0 else 1
        velocity_y_d = 0 if abs(self.velocity_y) < 0.015 else (-1 if self.velocity_y < 0.0 else 1)
        # Always discretize paddle position
        paddle_y_d = floor(BOARD_HEIGHT * self.paddle_y[0] / (1.0 - PLAYER1_PADDLE_HEIGHT))
        if paddle_y_d > (BOARD_HEIGHT - 1.0):
            paddle_y_d = (BOARD_HEIGHT - 1.0) # This can happen if we haven't corrected collision yet
        paddle2_y_d = 0.0
        if (1.0 - PLAYER2_PADDLE_HEIGHT) != 0:
            paddle2_y_d = floor(BOARD_HEIGHT * self.paddle_y[1] / (1.0 - PLAYER2_PADDLE_HEIGHT))
        return (ball_x_d, ball_y_d, velocity_x_d, velocity_y_d, paddle_y_d)
    
    def __str__(self):
        return str((self.ball_x, self.ball_y, self.velocity_x, self.velocity_y, self.paddle_y))

# NOTE: We're going to hack with our decide functions a little bit :(
# They can mutate state and return True or False in case of collision
def decide_q_learning(state, ball_pos):
    global Q, N, S, A, R, ALPHA
    actions = [0.0, 0.04, -0.04]
    sp = state.discretize()
    Rp = Rsignal(state)
    # Check terminal (ball's x-coord > paddle)
    if state.ball_x >= 1.0:
        Q[(sp, 0.0)] = Rp
    if S:
        if (S, A) not in N:
            N[(S, A)] = 0.0
        N[(S, A)] += 1.0
        if (S, A) not in Q:
            Q[(S, A)] = 0.0
        ALPHA = C / (C + N[(S, A)])
        maxAP = None
        for ap in actions:
            if (sp, ap) not in Q:
                Q[(sp, ap)] = 0.0
            maxAP = max(maxAP, Q[(sp, ap)]) 
        Q[(S, A)] = Q[(S, A)] + (ALPHA * (N[(S, A)]) * (R + (GAMMA * maxAP - Q[(S, A)])))
    # Update state
    S = sp
    A = None
    maxVal = None
    for ap in actions:
        if (sp, ap) not in Q:
            Q[(sp, ap)] = 0.0
        if (sp, ap) not in N:
            N[(sp, ap)] = 0.0
        fVal = exploration_fn(Q[(sp, ap)], N[(sp, ap)])
        if maxVal < fVal:
            maxVal = fVal
            A = ap
    R = Rp
    # Take action:
    #print "Taking: {}".format(A)
    state.paddle_y[0] += A
    # Make sure paddle stays in bounds
    (p_upper, p_lower) = get_paddle_bounds(state.paddle_y[0], PLAYER1_PADDLE_HEIGHT)
    if p_upper < 0.0:
        state.paddle_y[0] = 0.0
    elif p_lower > 1.0:
        diff = p_lower - 1.0
        state.paddle_y[0] -= diff
    # Check for collision (i.e. positive reward)
    if Rp == 1.0:
        return True
    return False # No bounce

def exploration_fn(u, n):
    if random.random() < EC:
        return 1.0
    #if n < Ne:
    #    return 1.0
    return u

def Rsignal(state):
    (paddle_upper, paddle_lower) = get_paddle_bounds(state.paddle_y[0], PLAYER1_PADDLE_HEIGHT)
    ball_x = state.ball_x
    ball_y = state.ball_y
    if ball_x >= 1.0 and ball_y >= state.paddle_y[0] and ball_y <= (state.paddle_y[0] + PLAYER1_PADDLE_HEIGHT):
        return 1.0 # Hit 
    elif ball_x >= 1.0:
        return -1.0 # Miss
    return 0.0 # No change

def get_paddle_bounds(paddle_upper, paddle_height):
    return (paddle_upper, paddle_upper + paddle_height)

def decide_wall(state, ball_pos):
    if state.ball_x <= 0.0:
        state.ball_x = -state.ball_x
        state.velocity_x = -state.velocity_x
    return False # Never do randomized bounces for wall

def decide_hardcoded(state, ball_pos):
    # Move paddle closer to having center aligned with ball
    center = state.paddle_y[1] + (PLAYER2_PADDLE_HEIGHT / 2.0)
    difference = center - state.ball_y
    if difference < 0.0:
        #print center, state.ball_y
        # Move paddle up
        state.paddle_y[1] += 0.02
    elif difference > 0.0:
        # Move paddle down
        state.paddle_y[1] -= 0.02
    # Make sure paddle stays in bounds
    (p_upper, p_lower) = get_paddle_bounds(state.paddle_y[1], PLAYER2_PADDLE_HEIGHT)
    if p_upper < 0.0:
        state.paddle_y[1] = 0.0
    elif p_lower > 1.0:
        diff = p_lower - 1.0
        state.paddle_y[1] -= diff
    return state.ball_x <= 0.0 and state.ball_y >= state.paddle_y[1] and state.ball_y <= (state.paddle_y[1] + PLAYER2_PADDLE_HEIGHT)

def discretize_point(x, boundary):
    return floor(x * boundary)

def discretize_position(state):
    return (discretize_point(state.ball_x, BOARD_WIDTH), discretize_point(state.ball_y, BOARD_HEIGHT))

def handle_collision(state, paddle_x):
    U = random.uniform(-0.015, 0.015)
    V = random.uniform(-0.03, 0.03)
    state.ball_x = (2.0 * paddle_x) - state.ball_x
    state.velocity_x = -state.velocity_x + U
    state.velocity_y = state.velocity_y + V
    if abs(state.velocity_x) <= 0.03:
        sgn = -1 if state.velocity_x < 0.0 else 1
        state.velocity_x = sgn * 0.03

def main():
    global PLAYER2_PADDLE_HEIGHT
    args = parse_args()
    player2_strat = decide_hardcoded
    # Player 2 paddle is entire height of board for part 1
    if args.part == 1:
        PLAYER2_PADDLE_HEIGHT = 1.0
        player2_strat = decide_wall

    print "Training..."
    # Train model
    for trainingIteration in range(args.num_train):
        # State tuple as defined in write up
        state = State(0.5, 0.5, 0.03, 0.01, 0.5 - (PLAYER1_PADDLE_HEIGHT / 2.0), 0.5 - (PLAYER2_PADDLE_HEIGHT / 2.0))
        if trainingIteration % 10000 == 0:
            print "Completed {}/{} rounds...".format(trainingIteration, args.num_train)
        #    print Q
        simulation_loop(state, decide_q_learning, player2_strat)

    # Sanity checks
    seenStates = set([key[0] for key in Q])
    numStates = len(seenStates)
    print "Num states seen: {}".format(numStates)
    #print seenStates
    assert numStates <= 10368 # Sanity check: Should only have at most 10,368 states

    print "Testing..."
    totalScore = 0.0
    for testingIteration in range(args.num_test):
        state = State(0.5, 0.5, 0.03, 0.01, 0.5 - (PLAYER1_PADDLE_HEIGHT / 2.0), 0.5 - (PLAYER2_PADDLE_HEIGHT / 2.0))
        (player1_score, player2_score) = simulation_loop(state, decide_q_learning, player2_strat)
        if args.part == 1:
            # Score for player 1 bounces
            totalScore += player1_score
        elif args.part == 2:
            # Win percentage for player1 (ball always moves to player1 first so it will always have more bounces on a win)
            totalScore += 1.0 if player1_score > player2_score else 0.0
    avgScore = totalScore / float(args.num_test)
    if args.part == 1:
        print "Final average score: {}".format(avgScore)
    elif args.part == 2:
        print "Win percentage of player 1: {}".format(avgScore)

    print "Simulating a game for viewing..."
    state = State(0.5, 0.5, 0.03, 0.01, 0.5 - (PLAYER1_PADDLE_HEIGHT / 2.0), 0.5 - (PLAYER2_PADDLE_HEIGHT / 2.0))
    score = simulation_loop(state, decide_q_learning, player2_strat, True)
    print "Viewed simulation score: {}".format(score)

def simulation_loop(state, player1_strat, player2_strat, show_pygame=False):
    global ALPHA, T
    screen = None
    clock = None
    if show_pygame:
        pygame.init()
        screen = pygame.display.set_mode((240, 240))
        clock = pygame.time.Clock()
    # Simulation loop for single game
    # Condition: check ball is still in bounds
    player1_score = 0.0
    player2_score = 0.0
    while state.ball_x <= 1.0 and state.ball_x > 0.0:
        if show_pygame:
            pygame_loop(state, screen, 240, player1_score, player2_score)
            pygame.display.flip()
            clock.tick(30)
        # Ball moves according to vector
        state.ball_x += state.velocity_x
        state.ball_y += state.velocity_y
        # Wall bounce conditions:
        if state.ball_y < 0.0:
            state.ball_y = -state.ball_y
            state.velocity_y = -state.velocity_y
        elif state.ball_y > 1.0:
            state.ball_y = 2.0 - state.ball_y
            state.velocity_y = -state.velocity_y
        ball_pos = discretize_position(state)
        # Move paddle and handle collision
        if player2_strat(state, ball_pos):
            handle_collision(state, 0.0)
            player2_score += 1.0
        if player1_strat(state, ball_pos):
            handle_collision(state, 1.0)
            player1_score += 1.0
        # TODO: Reduce alpha at rate of 1/t
        # Times seen:
        ALPHA = C / (C + N[(S, A)])
        T += 1.0
    return (player1_score, player2_score)

def pygame_loop(state, screen, length, p1, p2):
    paddle1_upper = state.paddle_y[0] * length
    paddle2_upper = state.paddle_y[1] * length
    ball_x = int(state.ball_x * length)
    ball_y = int(state.ball_y * length)
    # Background:
    bg = pygame.Surface(screen.get_size())
    bg = bg.convert()
    bg.fill((0,0,0))
    # score:
    font = pygame.font.Font(None, 16)
    p1score = font.render("P1: {}".format(p1), 1, (250, 250, 250))
    p1pos = p1score.get_rect()
    p1pos.centerx = bg.get_rect().centerx + 25
    p2score = font.render("P2: {}".format(p2), 1, (250, 250, 250))
    p2pos = p2score.get_rect()
    p2pos.centerx = bg.get_rect().centerx - 25
    bg.blit(p1score, p1pos)
    bg.blit(p2score, p2pos)
    # Blank screen
    screen.blit(bg, (0,0))
    pygame.display.flip()
    #screen.fill((0,0,0))
    # Draw ball
    pygame.draw.circle(screen, (255, 255, 0), (ball_x, ball_y), 5, 5)
    # Draw paddle 1
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(length - 10, paddle1_upper, 10, PLAYER1_PADDLE_HEIGHT * length))
    # Draw paddle 2
    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0.0, paddle2_upper, 10, PLAYER2_PADDLE_HEIGHT * length))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", "-p", help="MP part (default: 1)", type=int, default=1)
    parser.add_argument("--num-train", "-n", help="Number of games to play for training", type=int, default=100000)
    parser.add_argument("--num-test", "-t", help="Number of games to play for testing", type=int, default=1000)
    return parser.parse_args()

if __name__ == "__main__":
    main()
