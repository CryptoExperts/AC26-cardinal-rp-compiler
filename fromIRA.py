#!/usr/bin/env python3

################################################################################
# This file enables the parsing of the output produced by the IronMask tool
# using the 'cardRPC' and 'genRPC' options, which compute the cardinal RPC
# envelope and the general RPC envelope of a gadget, respectively. The parsed
# results are returned as NumPy arrays or Python dictionaries, depending on the
# type of envelope.
################################################################################

import re
import itertools
import numpy as np
import ast
import os
import math
from math import log, fsum
from scipy.special import comb
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def all_subsets(n):
  """
  Generate all subsets of {0, ..., n - 1}.

  Args:
    n (int): Size of the ground set.

  Returns:
    list[list[int]]: All subsets of {0, ..., n - 1}, represented as lists and
      ordered by increasing cardinality.
  """
  s = range(n)
  return [list(c) for r in range(len(s)+1) for c in itertools.combinations(s, r)]

def nb_from_str(line, index_start) :
  """
  Extract the first integer appearing in a string from a given position. Useful
  to parse the results coming from IronMask to compute the cardinal RPC
  envelope. 

  Args:
    line (str): String from which the integer is extracted.
    index_start (int): Index from which to start searching for an integer.

  Returns:
    tuple[int, int]:
      The extracted integer and the index immediately following its last digit.

  Notes:
    This function assumes that at least one integer occurs in `line` at or
    after `index_start`.
  """ 
  index = index_start
  while (not line[index].isdigit()) : 
    index += 1
  
  index2 = index
  while(line[index2].isdigit()) : 
    index2 += 1
    
  return int(line[index:index2]), index2

def compute_proba(c, p, coeff_max_f) :
  r"""
  Compute the following probability : 
  $$
  f(p) = \sum_{i = 0}^{s} c_i p^i (1 - p)^{s - i} 
  $$
  where s is the total number of wires of the circuit.

  Args : 
    c(list[int]) : List of coefficients of Failures tuples.
    p(float) : Leakage rate.
    coeff_max_f(int) : Maximal exact coefficients computed thanks to IronMask in
    c. 
  
  Returns : 
    float : The sum above.
  """ 
  s = len(c) - 1

  terms = []

  for i in range (len(c)) :
    coeff = c[i] 
    if (i > coeff_max_f) : 
      coeff = comb(s, i) 

    terms.append(coeff * p**i * (1 - p)**(s - i)) 
  return fsum(terms)

def parse_enveloppes(filename, n, p, coeff_max_f) :
  """
  Parse a cardinal RPC envelope of a 1-to-1 gadget computed by IronMask (i.e. a
  refresh gadget).

  Args:
      filename (str): Path to the file containing the output of the IronMask
      computation.
      n (int): Number of shares.
      p (float): Leakage rate.
      coeff_max_f (int): Largest coefficient computed exactly by IronMask.

  Returns:
      np.ndarray: The cardinal RPC envelope of the gadget obtained from the
      IronMask output.
  """
  f = open(filename, 'r')
  tin = 0
  tout = 0
  env = np.zeros((n + 1, n + 1))
  line =  f.readline()
  while(line != '') : 
    if (len(line) > 3 and line[:3] == "tin") : 
      tin, index_start_tout = nb_from_str(line, 0)
      tout, index_start_tout = nb_from_str(line, index_start_tout)
      line = f.readline()
      index = 0
      while(line[index] != '[') :
        index += 1
      line = line[index:]
      
      fp = ast.literal_eval(line)
      if (coeff_max_f == - 1) :
          coeff_max_f = len(fp) + 1

      #Cardinal RPC envelopes : 
      env[tin, tout] = max(compute_proba(fp, p, coeff_max_f), env[tin, tout])
    line = f.readline()
  return env


def parse_enveloppes_UOcPRC(filename, n, p, coeff_max_f) :
  """
  Parse a UO cardinal RPC envelopeof a 1-to-1 gadget computed by IronMask (i.e.
  a refresh gadget).

  Args:
      filename (str): Path to the file containing the output of the IronMask
      computation.
      n (int): Number of shares.
      p (float): Leakage rate.
      coeff_max_f (int): Largest coefficient computed exactly by IronMask.

  Returns:
      np.ndarray: The UO cardinal RPC envelope of the gadget obtained from the
      IronMask output.
  """
  f = open(filename, 'r')
  tin = 0
  tout = 0
  env = np.zeros((n + 1, n + 1))
  line =  f.readline()
  while(line != '') : 
    if (len(line) > 3 and line[:3] == "tin") : 
      tin, index_start_tout = nb_from_str(line, 0)
      tout, index_start_tout = nb_from_str(line, index_start_tout)
      line = f.readline()
      index = 0
      while(line[index] != '[') :
        index += 1
      line = line[index:]
      
      fp = ast.literal_eval(line)
      if (coeff_max_f == - 1) :
          coeff_max_f = len(fp) + 1

      #UO Cardinal RPC envelopes : 
      env[tin, tout]  += (compute_proba(fp, p, coeff_max_f) / (math.comb(n, tout)))
    line = f.readline()
  return env



def parse_enveloppes_gen(filename, n, p, coeff_max_f) :
  """
  Parse a general RPC envelope of a 1-to-1 gadget computed by IronMask (i.e. a
  refresh gadget).

  Args:
      filename (str): Path to the file containing the output of the IronMask
      computation.
      n (int): Number of shares.
      p (float): Leakage rate.
      coeff_max_f (int): Largest coefficient computed exactly by IronMask.

  Returns:
      np.ndarray: The general RPC envelope of the gadget obtained from the
      IronMask output.
  """
  f = open(filename, 'r')
  env_gen = {}
  line =  f.readline()
  index_output = -1
  while(line != '') : 
    if (len(line) > 7 and line[:7] == "I_set =") :
      I_raw = re.search(r"I_set\s*=\s*\[(.*?)\]", line).group(1)
      O_raw = re.search(r"O_set\s*=\s*\[(.*?)\]", line).group(1)

      I_set = [int(x) for x in re.findall(r"-?\d+", I_raw)]
      O_nums = [int(x) for x in re.findall(r"-?\d+", O_raw)]

      if index_output == -1 and O_nums:
        index_output = O_nums[0]

      if index_output == -1:
        O_set = O_nums  
      else:
        O_set = [x - index_output for x in O_nums]

      line = f.readline()
      index = 0
      while(line[index] != '[') :
        index += 1
      line = line[index:]
      
      fp = ast.literal_eval(line)
      if (coeff_max_f == - 1) :
          coeff_max_f = len(fp) + 1

      I = sum(1 << i for i in I_set)
      J = sum(1 << i for i in O_set)
      env_gen[(I, J)] = compute_proba(fp, p, coeff_max_f)
    line = f.readline()
  return env_gen



def print_env(n, env) :
  """
  Print cardinal RPC envelope of a refresh gadget. Use mainly for debugging
  purpose.

  Args:
      n (int): Number of shares.
      env (np.ndarray): Cardinal RPC envelope of a 1-to-1 gadget (i.e. refresh
      gadget).
  """
  
  sm = np.zeros((n + 1))
  
  first_row = "         |"
  for j in range (n + 1) :
    if (j < 10) :
      first_row += "|J| = 0" + str(j) + "     |"
      
    else :
      first_row += "|J| = " + str(j) + "     |"
  print(first_row) 
  for i in range (n + 1) :
    row = ""
    if (i < 10) :
      row += "|I| = 0"+ str(i) + " |"
    else :  
      row += "|I| = "+ str(i) + " |"
    for j in range (n + 1) :
      sm[j] += env[i,j]
      var = "%e" % env[i,j]
      row += var + " |"
    print(row)
  print("\n\n")
  last_row = "         |"
  for j in range (n + 1) :
      var = "%e" % sm[j]
      last_row += var + " |"    
  print(last_row)











