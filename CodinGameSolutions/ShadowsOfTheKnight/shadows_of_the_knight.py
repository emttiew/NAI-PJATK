import sys
import math

# Author Mateusz Woźniak
# link: https://www.codingame.com/training/medium/shadows-of-the-knight-episode-1

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

# w: width of the building.
# h: height of the building.
w, h = [int(i) for i in input().split()]
n = int(input())  # maximum number of turns before game over.
x0, y0 = [int(i) for i in input().split()]

jump_x = x0
jump_y = y0
low_x = 0
high_x = w
low_y = 0
high_y = h
# game loop
while True:
    bomb_dir = input()  # the direction of the bombs from batman's current location (U, UR, R, DR, D, DL, L or UL)
    
    if "R" in bomb_dir:
        low_x = x0 + 1
        x0 += 1
        jump_x = int((high_x + x0) // 2)
        x0 = jump_x
    if "L" in bomb_dir:
        high_x = x0 - 1
        x0 -= 1
        jump_x = int((low_x + x0) // 2)
        x0 = jump_x 
    if "U" in bomb_dir:
        high_y = y0 - 1
        y0 -= 1
        jump_y = int((low_y + y0) // 2)
        y0 = jump_y
    if "D" in bomb_dir:
        low_y = y0 + 1
        y0 += 1
        jump_y = int((high_y + y0) // 2)
        y0 = jump_y 


    jump = str(jump_x) + " " + str(jump_y)
    print(jump)
