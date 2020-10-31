import sys
import math
# Author Mateusz Woźniak
# Problem: https://www.codingame.com/training/easy/the-descent
# game loop
while True:
    theHighest = 0
    index = 0
    for i in range(8):
        mountain_h = int(input())
        if mountain_h > theHighest:
            theHighest = mountain_h
            index = i
    print(index)
