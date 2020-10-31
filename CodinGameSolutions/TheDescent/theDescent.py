import sys
import math

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
