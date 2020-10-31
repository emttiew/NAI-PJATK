import sys
import math


# light_x: the X position of the light of power
# light_y: the Y position of the light of power
# initial_tx: Thor's starting X position
# initial_ty: Thor's starting Y position
light_x, light_y, initial_tx, initial_ty = [int(i) for i in input().split()]

# game loop
while True:
    remaining_turns = int(input())  
    if light_x > initial_tx and light_y == initial_ty:
        print("E")
        initial_tx += 1
    if light_x < initial_tx and light_y == initial_ty:
        print("W")
        initial_tx -= 1
    if light_y > initial_ty and light_x == initial_tx:
        print("S")
        initial_ty += 1
    if light_y < initial_ty and light_x == initial_tx:
        print("N")
        initial_ty -= 1
    if light_x > initial_tx and light_y > initial_ty:
        print("SE")
        initial_ty += 1
        initial_tx += 1
    if light_x > initial_tx and light_y < initial_ty:
        print("NE")
        initial_ty -= 1
        initial_tx += 1
    if light_x < initial_tx and light_y < initial_ty:
        print("NW")
        initial_ty -= 1
        initial_tx -= 1
    if light_x < initial_tx and light_y > initial_ty:
        print("SW")
        initial_ty += 1
        initial_tx -= 1

    

