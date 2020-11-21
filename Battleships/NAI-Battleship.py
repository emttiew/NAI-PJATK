## Author: Mateusz Woźniak s18182
## Author: JAkub Włoch s16912
## code reference: https://discuss.codecademy.com/t/excellent-battleship-game-written-in-python/430605
## game description: https://en.wikipedia.org/wiki/Battleship_(game)
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#TODO zmienić nazwy funkcji z _
#TODO dokumentacja

from random import randint
import random
import os
from enum import Enum


#Settings Variables 
row_size = 9 #number of rows
col_size = 9 #number of columns
num_ships = 4
max_ship_size = 5
min_ship_size = 2
num_turns = 40    

#Ship Class
class Ship:
  def __init__(self, size, orientation, location, board, board_display):
    self.size = size
    self.board = board
    self.board_display = board_display
    
    if orientation == 'horizontal' or orientation == 'vertical':
      self.orientation = orientation
    else:
      raise ValueError("Value must be 'horizontal' or 'vertical'.")
    
    if orientation == 'horizontal':
      if location['row'] in range(row_size):
        self.coordinates = []
        for index in range(size):
          if location['col'] + index in range(col_size):
            self.coordinates.append({'row': location['row'], 'col': location['col'] + index})
          else:
            raise IndexError("Column is out of range.")
      else:
        raise IndexError("Row is out of range.")
    elif orientation == 'vertical':
      if location['col'] in range(col_size):
        self.coordinates = []
        for index in range(size):
          if location['row'] + index in range(row_size):
            self.coordinates.append({'row': location['row'] + index, 'col': location['col']})
          else:
            raise IndexError("Row is out of range.")
      else:
        raise IndexError("Column is out of range.")

    if self.filled():
      print(" ".join(str(coords) for coords in self.coordinates))
      raise IndexError("A ship already occupies that space.")
    else:
      self.fillBoard()
  
  def filled(self):
    for coords in self.coordinates:
      if self.board[coords['row']][coords['col']] == 1:
        return True
    return False
  
  def fillBoard(self):
    for coords in self.coordinates:
      self.board[coords['row']][coords['col']] = 1

  def contains(self, location):
    for coords in self.coordinates:
      if coords == location:
        return True
    return False
  
  def destroyed(self):
    for coords in self.coordinates:
      if self.board_display[coords['row']][coords['col']] == 'O':
        return False
      elif self.board_display[coords['row']][coords['col']] == '*':
        raise RuntimeError("Board display inaccurate")
    return True

#Player Class
class Player:
    board = []
    board_display = []
    ship_list = []
    guess_coords = {}  
    index = 0
    lastImpactPoint = {}
    
    def __init__(self, index, player_type, row_size, col_size):
        self.index=index
        self.player_type = player_type
        self.row_size = row_size
        self.col_size = col_size
        self.board = [[0] * col_size for x in range(self.row_size)]
        self.board_display = [["O"] * col_size for x in range(self.row_size)]
        self.createShipList()
    
    def setLastImpact(self, coords):
        self.lastImpactPoint = coords
    
    def createShipList(self): 
        temp = 0
        while temp < num_ships:
          ship_info = self.random_location()
          if ship_info == 'None':
            continue
          else:
            self.ship_list.append(Ship(ship_info['size'], ship_info['orientation'], ship_info['location'], self.board, self.board_display))
            temp += 1
        del temp        
        
    def random_location(self):
        size = randint(min_ship_size, max_ship_size)
        orientation = 'horizontal' if randint(0, 1) == 0 else 'vertical'

        locations = self.search_locations(size, orientation)
        if locations == 'None':
            return 'None'
        else:
            return {'location': locations[randint(0, len(locations) - 1)], 'size': size,'orientation': orientation}
    
    def search_locations(self, size, orientation):
      locations = []
    
      if orientation != 'horizontal' and orientation != 'vertical':
        raise ValueError("Orientation must have a value of either 'horizontal' or 'vertical'.")
    
      if orientation == 'horizontal':
        if size <= self.col_size:
          for r in range(self.row_size):
            for c in range(self.col_size - size + 1):
              if 1 not in self.board[r][c:c+size]:
                locations.append({'row': r, 'col': c})
      elif orientation == 'vertical':
        if size <= self.row_size:
          for c in range(self.col_size):
            for r in range(self.row_size - size + 1):
              if 1 not in [self.board[i][c] for i in range(r, r+size)]:
                locations.append({'row': r, 'col': c})    
      if not locations:
        return 'None'
      else:
        return locations
    
    def guessLocation(self, enemy):
        if self.player_type == 2:
            self.AIguessLocation(enemy)
        else:
            while True:
              self.guess_coords['row'] = self.get_row()
              self.guess_coords['col'] = self.get_col()
              if enemy.board_display[self.guess_coords['row']][self.guess_coords['col']] == 'X' or \
                enemy.board_display[self.guess_coords['row']][self.guess_coords['col']] == '*':
                print("\nYou guessed that one already.")
              else:
                  break      
    def AIguessLocation(self, enemy):
        while True: 
            self.guess_coords['row'] = randint(0, self.row_size - 1)
            self.guess_coords['col'] = randint(0, self.col_size - 1)
            if enemy.board_display[self.guess_coords['row']][self.guess_coords['col']] == 'X' or \
              enemy.board_display[self.guess_coords['row']][self.guess_coords['col']] == '*':
              print("\nYou guessed that one already.")
            else:
                break 
             
    def get_row(self):
      while True:
        try:
          guess = int(input("Row Guess: "))
          if guess in range(1, self.row_size + 1):
            return guess - 1
          else:
            print("\nOops, that's not even in the ocean.")
        except ValueError:
          print("\nPlease enter a number")
    
    def get_col(self):
      while True:
        try:
          guess = int(input("Column Guess: "))
          if guess in range(1, self.col_size + 1):
            return guess - 1
          else:
            print("\nOops, that's not even in the ocean.")
        except ValueError:
          print("\nPlease enter a number")         
  
#Functions
def print_board(board_array):
    print("\n  " + " ".join(str(x) for x in range(1, col_size + 1)))
    for r in range(row_size):
        print(str(r + 1) + " " + " ".join(str(c) for c in board_array[r]))
        print()
        
def create_players():
    players = []
    while True:    
        print("Choose opponent type:")
        print("1. Hooman")
        print("2. Computer")
        print("3. Computer vs Computer")
        player_type = int(input("Type: "))
        if player_type == 1:
            players.append(Player(1, 1, row_size, col_size))
            players.append(Player(2, 1, row_size, col_size))
            break
        elif player_type == 2:
            players.append(Player(1, 1, row_size, col_size))
            players.append(Player(2, 2, row_size, col_size))
            break
        elif player_type == 3:
            players.append(Player(1, 2, row_size, col_size))
            players.append(Player(2, 2, row_size, col_size))
            break
        else:
            print("Wrong type number")
            os.system('clear')
    return players

# Play Game
os.system('clear')
previous_player = None
current_player = None

players_list = create_players()

current_player = random.choice(players_list)

for turn in range(num_turns):  
  
  if current_player == players_list[0]:
      previous_player = players_list[1]
  else:
      previous_player = players_list[0]
      
  print("Player", current_player.index, "turn")
  print("Turn:", turn + 1, "of", num_turns)
  print("Ships left:", len(current_player.ship_list))
  print()  
  
  current_player.guessLocation(previous_player)
  
  os.system('clear')

  ship_hit = False
  for ship in previous_player.ship_list:
    if ship.contains(previous_player.guess_coords):
      print("Hit!")
      ship_hit = True
      previous_player.setLastImpact(previous_player.guess_coords)
      previous_player.board_display[current_player.guess_coords['row']][current_player.guess_coords['col']] = 'X'
      if ship.destroyed():
        print("Ship Destroyed!")
        previous_player.ship_list.remove(ship)
      break
  if not ship_hit:
    previous_player.board_display[current_player.guess_coords['row']][current_player.guess_coords['col']] = '*'
    print("You missed!")

  print("Player's", previous_player.index, "table")
  print_board(previous_player.board_display)
  
  
  if not previous_player.ship_list:
    break
  
  current_player = previous_player

# End Game
if current_player.ship_list:
  print("You lose!")
else:
  print("All the ships are sunk. You win!")