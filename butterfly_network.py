#!/usr/bin/env python3

################################################################################
# This file enables the computation of the cardinal RPC envelopes for the
# 1-stage Butterfly Network. Then it enables to compute the threshold RPC
# security of the 1-stage butterfly network derived from the different envelopes
# computed.
# 
# This file is largely inspired by the "butterfly_network.py" file available on
# the GitHub repository:
#
#   https://github.com/CryptoExperts/AC25-cardinal-rp-compiler
#  
################################################################################


################################################################################
################################## Packages ####################################

from scipy.special import comb
import numpy as np

from refresh_env import (cRPC_pgadd, cRPC_pgcmult)


################################################################################
################################################################################
########################## Butterfly Stage 1 ###################################



################################################################################
############################## Cardinal RPC ####################################


def env_ref_add(n, env_ref, env_add):
  r"""
  Compute the enveloppes of the following scheme : 

  --- R ---
           \
            + ---> 
           /
  --- R ---

  Args : 
    n : The number of shares.
    env_ref : The cardinal-RPC enveloppes of the refresh gadget.
    env_add : The cardinal-RPC enveloppes of the addition gadget.
  
  Returns :
    The cardinal-RPC enveloppes of the above scheme.  
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout in range (n + 1) :
        for l1 in range (n + 1) :
          for l2 in range (n + 1) :
            env[tin1, tin2, tout] += env_ref[tin1, l1] * env_ref[tin2, l2] * env_add[l1, l2, tout]
  return env 
                    
def cRPC_copy(n, p) :
  r"""
  Compute the cardinal RPC enveloppes of the copy gadget. Useful for version v2 
  of the first stage butterfly network for example.

  Copy gadget (no refresh gadget) : 

          ---------->
         /
  -------
         \
          ---------->

  Args : 
    n : The number of shares.
    p : The leakage rate probability.

  Returns : 
    The cardinal-RPC enveloppes of the copy gadget described above.

  Notes : 
    In the following description of the scheme, it is noted as C.    
  """
  env = np.zeros((n + 1, n + 1, n + 1))
  for tin in range (n + 1) :
    for tout1 in range (tin + 1) : 
      for tout2 in range (tin + 1) :
        for l in range (tin + 1) :
          if (tin <= (tin - l + tout1  + tout2)) :
            env[tin, tout1, tout2] += (comb(n - l, tin - l) * p**(tin - l) * 
                                       (1 - p)**(n - tin))
        env[tin, tout1, tout2] = min(1, env[tin, tout1, tout2])
  return env
          
def cRPC_prec1 (n, env_add, env_copy) :
  r"""
  Compute the cardinal RPC enveloppes of the following scheme :

  ------ C ------- + -------->
          \       /
  ---------\-----         
            ----------------->

  Args : 
    n : The number of shares.
    env_add : The cardinal RPC enveloppe of the addition gadget (noted + above).
    env_copy : The cardinal RPC enveloppe of the copy gadget (noted C above).

  Returns :
    The cardinal RPC enveloppe of the scheme described above.
  """
  env = np.zeros((n + 1,  n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (tin1 + 1) :
            env[tin1, tin2, tout1, tout2] += env_add[l, tin2, tout1] * env_copy[tin1, l, tout2]
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env
  
def cRPC_prec2 (n , env_prec1, env_copy) :
  r"""
  Compute the cardinal RPC enveloppes of the following scheme :

  ------ C ------- + -------->
          \       /
         --\-----         
        /   ------------------>
  ---- C
        \
         --------------------->
  
  Args : 
    n : The number of shares.
    env_prec1 : The cardinal RPC enveloppe of the previous scheme.
    env_copy : The cardinal RPC enveloppe of the copy gadget (noted C above).

  Returns :
    The cardinal RPC enveloppe of the scheme described above.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout3 in range (n + 1) :
        for tout1 in range (n + 1) :
          for tout2 in range (n + 1) :
            for l in range (tin2 + 1) :
              env[tin1, tin2, tout1, tout2, tout3] += env_prec1[tin1, l, tout1, tout2] * env_copy[tin2, l, tout3]
            env[tin1, tin2, tout1, tout2, tout3] = min(1, env[tin1, tin2, tout1, tout2, tout3])
  return env

def cRPC_prec3 (n, env_prec2, env_add) :
  r"""
  Compute the cardinal RPC enveloppes of the following scheme :

  ------ C ------- + --------->
          \       /
         --\-----         
        /   -------------
  ---- C                 \
        \                 + -->
         \               /
          ---------------
  
  Args : 
    n : The number of shares.
    env_prec2 : The cardinal RPC enveloppe of the previous scheme.
    env_add : The cardinal RPC enveloppe of the add gadget (noted + above).

  Returns :
    The cardinal RPC enveloppe of the scheme described above.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) : 
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l1 in range (n + 1) :
            for l2 in range (n + 1) :
              env[tin1, tin2, tout1, tout2] += env_prec2[tin1, tin2, tout1, l1, l2] * env_add[l1, l2, tout2]
          env[tin1, tin2, tout1, tout2] = min (1, env[tin1, tin2, tout1, tout2])
  return env

def cRPC_prec4 (n, env_prec3, env_cmult) :
  r"""
  Compute the cardinal RPC enveloppes of the following scheme :

  -------------- C ------- + --------->
                  \       /
                 --\-----         
                /   -------------
  ---- C* ---- C                 \
                \                 + -->
                 \               /
                  ---------------
  
  Args : 
    n : The number of shares.
    env_prec3 : The cardinal RPC enveloppe of the previous scheme.
    env_cmult : The cardinal RPC enveloppe of the multiplication by a constant 
    gadget (noted C* above).

  Returns :
    The cardinal RPC enveloppe of the scheme described above.
  """
  env = np.zeros((n + 1, n + 1, n + 1, n + 1))
  for tin1 in range (n + 1) :
    for tin2 in range (n + 1) :
      for tout1 in range (n + 1) :
        for tout2 in range (n + 1) :
          for l in range (n + 1) :
            env[tin1, tin2, tout1, tout2] += env_cmult[tin2, l] * env_prec3[tin1, l, tout1, tout2]
          env[tin1, tin2, tout1, tout2] = min(env[tin1, tin2, tout1, tout2], 1)
  return env

def ref_cRPC_butterfly_stage1 (n, p, pgref):
  """
  Compute the cardinal RPC envelope of the first stage of the butterfly network.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the butterfly network.

  Returns:
      np.ndarray: Cardinal RPC envelope of the 1-stage butterfly network.
  """

  env_add = cRPC_pgadd (n, p, pgref)
  env_cmult = cRPC_pgcmult(n, p, pgref)
  env_copy = cRPC_copy(n, p)
  
  env_prec1 = cRPC_prec1 (n, env_add, env_copy)
  env_prec2 = cRPC_prec2 (n , env_prec1, env_copy)
  env_prec3 = cRPC_prec3 (n, env_prec2, env_add)
  env_butterfly = cRPC_prec4 (n, env_prec3, env_cmult)
  
  return env_butterfly

def tRPC_butterfly (n, env_butterfly, t) :
  """
  Compute the threshold Random Probing Composability of the first stage of the 
  butterfly network.

  Args : 
    n : The number of shares.
    env_butterfly : The cardinal RPC enveloppe of the first stage of the
    butterfly network.
    t : The threshold parameter.

  Returns :
    Advantage ε of the threshold Random Probing Composability derived from the 
    cardinal RPC enveloppes `env_butterfly` of the first stage of 
    the butterfly network. 
  """  
  eps = 0
  for tout1 in range (t + 1) :
    for tout2 in range (t + 1) :  
      tmp = 0
      for tin1 in range (n + 1) :
        for tin2 in range (n + 1) :
          if (tin1 > t or tin2 > t) :
            tmp += env_butterfly[tin1, tin2, tout1, tout2]
      eps = max(eps, min(tmp, 1))
  return eps

def new_tRPC_butterfly (n, env_butterfly) :
  """
  Compute the threshold Random Probing Composability of the first stage of the 
  butterfly network.

  Args : 
    n : The number of shares.
    env_butterfly : The cardinal RPC enveloppe of the first stage of the
    butterfly network.
    t : The threshold parameter.

  Returns :
    Advantage ε of the threshold Random Probing Composability derived from the 
    cardinal RPC enveloppes `env_butterfly` of the first stage of 
    the butterfly network. 
  """  
  eps = 0
  for t in range (1, n) :
    eps_t = tRPC_butterfly(n, env_butterfly, t)
    eps = max(eps, eps_t)
  return eps

