#!/usr/bin/env python3

################################################################################
# This file contains the core routines used to evaluate the random probing
# security of the schemes considered in our paper. It computes and manipulates
# general, cardinal, and UO cardinal RPP/RPC envelopes, including their
# compositions and the corresponding threshold RPC security, for every refresh
# gadget considered in the paper. It also provides implementations of the
# different refresh gadgets, together with analytical formulas and validation
# thanks to IronMask.
################################################################################

from collections import Counter, defaultdict
import itertools
from itertools import combinations, product
import numpy as np
import math
from scipy.special import comb
from sympy import  simplify
import pickle
import tempfile
import time
import os
from multiprocessing import Pool, cpu_count

import sys

from fromIRA import parse_enveloppes, parse_enveloppes_UOcPRC, parse_enveloppes_gen



## -----------------------------------------------------------------------------
##                  Useful Precomputations 
## -----------------------------------------------------------------------------

def precompute_binomial_coefficients(max_n):
  """
  Precompute the binomial coefficient until max_n.

  Args : 
      max_n(int) : Maximale number of population.
  
  Returns : 
      dict[tuple[int, int], float]: Dictionary containing the precomputed
      binomial coefficients.  
  """  
  binom_coeffs = {}
  for n in range(max_n + 1):
    for k in range(n + 1):
      binom_coeffs[(n, k)] = comb(n, k, exact=True)
  return binom_coeffs

def precompute_p_powers (p, max_power) :
  """
  Precompute powers of p.

  Args:
      p (float): leakage rate.
      max_power (int): Maximale power of p precomputed.

  Returns:
      dict[[int], float]: Dictionnary containing the precomputed power of p.
  """
  p_powers = {}
  res = 1
  for i in range (1, max_power + 2) :
    p_powers[i - 1] = res
    res *= p
  return p_powers

## -----------------------------------------------------------------------------
##                  Useful Notions
## -----------------------------------------------------------------------------



############################ gRPC from gRPP ####################################
def part_to_block_masks(part: bytes) -> list[int]:
  """
  Exemple of what do this function. Suppose you have a partition : 

  Part = {{0, 1}, {2, 3, 4}, {5}}
  which is represent in our modelisation by : 
    
  part = b'\x00\x00\x01\x01\x01\x02'

  Then we will return 

  res := [0b000011, 0b011100, 0b100000] = [3, 28, 32]

  corresponding to the 3 blocks and the index contained it modelized as a
  bitset. 

  Args : 
      part(bytes) : A partition of the ZeroEncoding gadget.
    
  Returns :
      list[int] : the partition in a bitset format.
  """
  if not part:
    return []

  k = max(part) + 1         
  block_masks = [0] * k

  for i, lbl in enumerate(part):
    block_masks[lbl] |= 1 << i

  return block_masks



def gRPC_from_gRPP (n, p, env_gRPP) :
  """
  Obtain the general RPC enveloppes of a refresh gadget based on a Zero Encoding
  gadget, which is proven general RPP, by applying Theorem 1 of our paper.
  The computation of the general RPC enveloppes is based on bitsets. For
  exemple, if we want to compute the general RPC enveloppes for a set I = {0, 2,
  4} and a set J = {3, 4}. Then we represents I and J as follows : 

  I = 0b10101
  J = 0b11000 
  
  i.e. if the i-th bit is set to 1, then i is in the set, otherwise i is not in
  set. 
  
  Args : 
      n(int) : Number of shares.
      p(float) : Probability Leakage rate.
      env_gRPP (np.ndarray): Object containing the partitions and the
      probability associated to these partitions that can be obtained in output
      of the zero encoding gadget.
  
  Returns : 
      np.ndarray : General RPC envelope of a refresh gadget, derived from the 
      general RPP envelope.
  """

  def submasks(I):
    """
    Generate all the subsets of I. Starting from I1 = I, which is include in I.
    Then we subtract 1 to I1, and then, the next subsets which is a subset of I
    is I1 & I. We do this until I1 = 0, which finish the function. 
    
    Args : 
        I(int) : Bitsets of I. 
    """

    I1 = I
    while True:
        yield I1
        if I1 == 0:
            break
        I1 = (I1 - 1) & I
  
  def build_q_by_J(n, env_gRPP):
    """
    Return q(I1, J) for all bitsets I1 and J in {0, ..., n - 1}, the function
    from Theorem 1 of our paper.
    
    Args : 
        n (int): Number of shares.
        env_gRPP(np.ndarray) : The general RPP envelope of the refresh gadget.
    
    Returns : 
        list[dict[[int], float]] : The function q(I1, J) for all bitsets I1 and
        J, where the bitsets of J is the index of the list.
    """
    parts = [(part_to_block_masks(part), p) for part, p in env_gRPP.items()]
    q_by_J = [defaultdict(float) for _ in range(1 << n)]

    for J in range(1 << n):
      notJ = ~J
      qJ = q_by_J[J]
      for blocks, p in parts:
        U = 0
        for M in blocks:
          if (M & notJ) == 0:  
            U |= M
        qJ[U] += p              
    return q_by_J
  
  ##############################################################################
  ######################### Core of the function ###############################
  
  p_power_cache = precompute_p_powers(p, n)
  one_minus_p_power_cache = precompute_p_powers(1 - p, n)

  q_by_J = build_q_by_J(n, env_gRPP)
  env_gen = {}
  for I in range (1 << n) :
    card_I = I.bit_count()
    p2 = one_minus_p_power_cache[n - card_I]
    for J in range(1 << n) :
      p_I_J = 0
      qJ = q_by_J[J]

      for I1 in submasks(I) :
        card_I1 = I1.bit_count()
        p1 = p_power_cache[card_I - card_I1]
        p_I_J += p1 * qJ.get(I1, 0.0)
      
      p_I_J *= p2
      env_gen[(I, J)] = p_I_J
  
  return env_gen
################################################################################


############################ cRPC from gRPP ####################################
  
def build_q_tin_J(n, env_gRPP):
  """
  Copy of the function above, instead the return type os different. TODO :
  Factorize the two functions.
  """
  parts = [(part_to_block_masks(part), p) for part, p in env_gRPP.items()]
  q = np.zeros((n + 1, (1 << n + 1)))

  for J in range(1 << n):
    tout = J.bit_count()

    q_J = np.zeros((n + 1))

    notJ = ~J
    #qJ = q_by_J[J]
    for blocks, p in parts:
      I = 0
      for M in blocks:
        if (M & notJ) == 0:  
          I |= M
      tin = I.bit_count()
      q_J[tin] += p

    for tin in range (n + 1) : 
      q[tin, J] = q_J[tin]
    
  return q


def cRPC_from_gRPP (n, p, env_gRPP) :
  """
  Derived cardinal RPC envelope from general RPP envelope of a refresh gadget
  using Corollary 1 of our article.

  Args :
      n (int) : Number of shares.
      p (float) : Leakage rate.
      env_gRPP (np.ndarray) : General RPP envelope of the refresh gadget.
  
  Returns : 
      np.ndarray : Cardinal RPC envelope of the refresh gadget.
  """  
  env_card = np.zeros((n + 1, n + 1))
  q = build_q_tin_J(n, env_gRPP)
  
  p_power_cache = precompute_p_powers(p, n)
  one_minus_p_power_cache = precompute_p_powers(1 - p, n)
  binom_coeffs = precompute_binomial_coefficients(n)


  for tin in range (n + 1) :
    for J in range (1 << n) :
      env_card_J = np.zeros((n + 1))
      tout = J.bit_count()

      for q_tin in range (tin + 1) :
        env_card_J[tin] += (q[q_tin, J] * p_power_cache[tin - q_tin] * 
          one_minus_p_power_cache[n - tin] * binom_coeffs[n - q_tin, tin - q_tin])

      #Cardinal RPC formula.
      env_card[tin, tout] = max(env_card[tin, tout], env_card_J[tin])
    
  return env_card

def UOcRPC_from_gRPP (n, p, env_gRPP) :
  """
  Derived UO cardinal RPC envelope from general RPP envelope of a refresh 
  gadget.

  Args :
      n (int) : Number of shares.
      p (float) : Leakage rate.
      env_gRPP (np.ndarray) : General RPP envelope of the refresh gadget.
  
  Returns : 
      np.ndarray : UO Cardinal RPC envelope of the refresh gadget.
  """    
  env_card = np.zeros((n + 1, n + 1))
  q = build_q_tin_J(n, env_gRPP)
  
  p_power_cache = precompute_p_powers(p, n)
  one_minus_p_power_cache = precompute_p_powers(1 - p, n)
  binom_coeffs = precompute_binomial_coefficients(n)


  for tin in range (n + 1) :
    for J in range (1 << n) :
      env_card_J = np.zeros((n + 1))
      tout = J.bit_count()

      for q_tin in range (tin + 1) :
        env_card_J[tin] += (q[q_tin, J] * p_power_cache[tin - q_tin] * 
          one_minus_p_power_cache[n - tin] * binom_coeffs[n - q_tin, tin - q_tin])

      env_card[tin, tout] += env_card_J[tin]

  #UO cardinal RPC formula.
  for tin in range (n + 1) :
    for tout in range (n + 1) :
      env_card[tin, tout] = (env_card[tin, tout] / binom_coeffs[(n, tout)])
    
  return env_card
################################################################################

############################ cRPC from gRPC ####################################
def cRPC_from_gRPC(n, env_gen):
  """
  Derived Cardinal RPC envelope from general RPC envelope of a refresh
  gadget. 

  Args:
      n (int): Number of shares.
      env_gen (np.ndarray): General RPC envelope of a refresh gadget.

  Returns:
      np.ndarray: Cardinal RPC envelope.
  """
  env_card = np.zeros((n + 1, n + 1))
  
  for tout in range (n + 1) :
    for tin in range (n + 1) :
      res = 0
      for J in range(1 << n) :
        if(not(J.bit_count() == tout)) :
          continue
        res_J = 0
        for I in range(1 << n) :
          if (not(I.bit_count() == tin)) :
            continue
          res_J += env_gen[(I, J)]
        res = max(res, res_J)
      env_card[tin, tout] = res
  return env_card


def UOcRPC_from_gRPC(n, env_gen):
  """
  Derived UO Cardinal RPC envelope from general RPC envelope of a refresh
  gadget. 

  Args:
      n (int): Number of shares.
      env_gen (np.ndarray): General RPC envelope of a refresh gadget.

  Returns:
      np.ndarray: UO Cardinal RPC envelope.
  """
  env_card = np.zeros((n + 1, n + 1))
  binom_coeffs = precompute_binomial_coefficients(n)
  
  for J in range (1 << n) :
    tout = J.bit_count()
    for I in range (1 << n) : 
      tin = I.bit_count()
      env_card[tin, tout] += env_gen[(I, J)]

  #UO Cardinal RPC formula.
  for tin in range (n + 1) :
    for tout in range (n + 1) :
      env_card[tin, tout] = (env_card[tin,tout] / binom_coeffs[(n, tout)])

  return env_card
################################################################################

############################## cRPC from cRPP ##################################
def cRPP_output_manipulation(int_partitions_list, env_cRPP, binom_coeffs, p, n):
  """
  Update the probabilities for each cardinal random probing partition with an
  additional leakage on the outputs of the zero encoding (zi) before they are
  added to the inputs shares. This corresponds to Lemma 7 in [BelRivRos25].

  Args:
      int_partitions_list (list[tuple[int, ...]]): The list of possible
      partitions in ZeroEnc.
      env_cRPP (np.ndarray): cRPP envelopes of ZeroEnc.
      binom_coeffs (dict[tuple[int, int], float]): Precomputed binomial coefficients.
      p (float): Leakage rate.
      n (int): Number of shares.

  Returns:
      list[float]: cRPP envelopes of ZeroEnc^+.
  """

    
  initial_probabilities = list(env_cRPP)
  env_cRPP_out = [0] * len(env_cRPP)

  # Convert int_partitions_list to a dictionary for O(1) lookups
  int_partitions_dict = {tuple(cs): idx for idx, cs in enumerate(int_partitions_list)}

  # Precompute the powers of p and (1-p) for efficiency
  p_powers = [p**i for i in range(n+1)]
  one_minus_p_powers = [(1-p)**i for i in range(n+1)]

  for ind_cs, cs in enumerate(int_partitions_list): 
    # (k1, k2, ..., kl) with ki the number of elements leaking in the set of cardinal ci
    all_combinations = itertools.product(*[range(c + 1) for c in cs])
    for combo in all_combinations:
      # we build the corresponding partition cs_2
      s = sum(combo)
      # we remove all the leaking elements from the cardinals
      updated_partition = [ci - ki for ci, ki in zip(cs, combo) if ci - ki > 0]
      # we add ones for all the leaking elements
      updated_partition.extend([1] * s)
      index_updated_set = find_cardinal_set_index(int_partitions_dict, updated_partition)
            
      # we compute the probability of the leakage
      pleak = 1
      for ci, ki in zip(cs, combo):
        pleak *= binom_coeffs[(ci, ki)] * p_powers[ki] * one_minus_p_powers[ci-ki]
      env_cRPP_out[index_updated_set] += initial_probabilities[ind_cs]*pleak

  return env_cRPP_out

def compute_probas_x_outputs(n, t_out, int_partitions_list, env_cRPP, 
                             binom_coeffs):
  """
  Probabilities that the internal leakage and the given outputs make it possible
  to recover x input shares.
  
  Args:
      n (int): Number of shares.
      t_out (int): Number of output shares of the refresh gadget leaked.
      int_partitions_list (list[tuple[int, ...]]): The list of possible
      partitions in ZeroEnc^+.
      env_cRPP (list[float]): cRPP envelope of ZeroEnc^+.
      binom_coeffs (dict[tuple[int, int], float]): Precomputed binomial
      coefficients. 

  Returns:
      list[float]: Probabilities that the internal leakage and the given outputs
      make it possible to recover x input shares.
  """

  probas_x_outputs = [0] * (t_out+1)
  binom_n_t_out = binom_coeffs[(n, t_out)]

  for ind_cs, cs in enumerate(int_partitions_list):    
    if min(cs) > t_out:
      probas_x_outputs[0] += env_cRPP[ind_cs]

    else:
      # Multihypergeometric law
      # 1. We enumerate all the (k1,k2,k_ell) with ell the number of cardinals cs
      # ki = number of known outputs in the set of cardinal ci
      valid_combinations = [
        combo for combo in itertools.product(*[range(c + 1) for c in cs])
        if sum(combo) == t_out
      ]
      for combo in valid_combinations:
        # 2. for each of them, we compute s_k = sum(k_i, tq k_i = c_i)
        s_k = sum(k for k, c in zip(combo, cs) if k == c)
                
        # 3. we update probas_x_outputs[s_k] with probas[ind_cs]*(see multihypergeometric law)
        proba_this_combo = 1
        for i, k_i in enumerate(combo):    
          proba_this_combo *= binom_coeffs[(cs[i],k_i)]

        proba_this_combo = proba_this_combo/binom_n_t_out
        probas_x_outputs[s_k] += env_cRPP[ind_cs]*proba_this_combo

  return probas_x_outputs

def pb_leaking_input_shares_cache(n, p, binom_coeffs):
  r"""
  Precompute the leakage of i input shares among n input shares, for all i in
  {0, ..., n}.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      binom_coeffs (dict[tuple[int, int], float]): Precomputed binomial
      coefficients. 
  
  Returns:
      np.ndarray: Probabilities of leaking i input shares among n, for all nt
      \in {0, \ldots, n} and i \in {0, \ldots, nt}.
  """
  proba_leaking_input_shares = np.zeros((n+1, n+1))
  p_powers = [p**i for i in range(n+1)]
  one_minus_p_powers = [(1-p)**i for i in range(n+1)]
  for nt in range(0,n+1):
    for it in range(0,nt+1):
      proba_leaking_input_shares[nt,it] = (binom_coeffs[(nt, it)] * p_powers[it] 
                                           * one_minus_p_powers[nt-it])
  return proba_leaking_input_shares

def cRPC_tin_tout(n, t_in, t_out, p, probas_x_outputs, 
                           cache_pr_leaking_input_shares):
  """
  Compute Cardinal RPC envelope of a Zero Encoding refresh gadget, based on
  Theorem 1 of [BelRivRos25].

  Args:
      n (int): Number of shares.
      t_in (int): Number of input shares needed for the simulation of the
      leakage. 
      t_out (int): Number of output shares leaked that must be simulated.
      p (float): Leakage rate.
      probas_x_outputs (list[float]): Probabilities of needing x input shares to
      simulate tout outputs shares.
      cache_pr_leaking_input_shares (np.ndarray): Probabilities of leaking input
      shares.

  Returns:
      float : Probabilities to need tin input shares to simulate the internal
      leakages as well as the tout leaked output shares.
  """
               
  pb_cRPC_tin_tout = 0
  for tt in range(0,min(t_in+1,t_out+1)):
    # tt input shares are recovered from the internal leakage and the outputs
    # j input shares are still missing among the remaining ones
    j= t_in - tt
    pb_cRPC_tin_tout += (probas_x_outputs[tt] * 
                         cache_pr_leaking_input_shares[n - tt, j])
  return pb_cRPC_tin_tout

def cRPC_from_cRPP (n, p, env_cRPP) :
  """
  Compute a cardinal RPC envelope of a refresh gadget from a cardinal RPP
  envelope of a Zero encoding. Based on Theorem 1 in [BelRivRos25]. 

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      env_cRPP (np.ndarray): Cardinal RPP envelope of the zero encoding.

  Returns:
      np.ndarray: Cardinal RPC envelope of the refresh gadget.
  """
  binom_coeffs = precompute_binomial_coefficients(n)
  int_partitions_list = generate_cardinal_set(n)
  cache_pb_leaking_input_shares = pb_leaking_input_shares_cache(n,p,binom_coeffs)
  pgref = np.zeros((n+1, n+1))
  for j in range(n + 1):
    probas_x_outputs = compute_probas_x_outputs(n, j, int_partitions_list, 
                                                env_cRPP, binom_coeffs)
    for i in range(n+1):
      pgref[i, j] = cRPC_tin_tout(n, i, j, p, probas_x_outputs, 
                                           cache_pb_leaking_input_shares)
  return pgref

################################################################################
def tRPC_from_gRPC_t (n, t, env) :
  """
  Compute the t-threshold RPC advantage from the general RPC envelope of a
  refresh gadget.

  Args:
      n (int): Number of shares.
      t (int): Threshold number.
      env (np.ndarray): General RPC envelope of the refresh gadget.

  Returns:
      float: Advantage of the t-threshold RPC property.
  """
  eps = 0
  for J in range(1 << n) :
    if J.bit_count() >= t :
      continue
    eps_J = 0
    for I in range(1 << n) :
      if I.bit_count() <= t :
        continue
      eps_J += env[(I, J)]
    eps = max(eps, eps_J)
  return eps

def tRPC_from_gRPC (n, env) :
  """
  Compute the best advantage of t-threshold RPC we can obtain among all the t in
  {1, ..., n - 1}, derived from the general RPC envelope of a refresh gadget.

  Args:
      n (int): Number of shares.
      env (np.ndarray): General RPC envelope of the refresh gadget.

  Returns:
      float: Best advantage of t-threshold RPC we can obtain among all the t in
      {1, ..., n - 1}
  """
  eps = 1 
  for t in range (1, n) :
    eps = min(eps, tRPC_from_gRPC_t(n, t, env))
  return eps

def tRPC_from_cRPC_t (n, env, t) :
  """
  Compute the t-threshold RPC advantage from the cardinal RPC envelope of a
  refresh gadget.

  Args:
      n (int): Number of shares.
      t (int): Threshold number.
      env (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      float: Advantage of the t-threshold RPC property.
  """
  eps = 0
  for tout in range (t + 1) :
    eps_tout = 0
    for tin in range (t + 1, n + 1) :
      eps_tout += env[tin, tout]
    eps = max(eps, eps_tout)
  return eps

def tRPC_from_cRPC (n, env) :
  """
  Compute the best advantage of t-threshold RPC we can obtain among all the t in
  {1, ..., n - 1}, derived from the cardinal RPC envelope of a refresh gadget.

  Args:
      n (int): Number of shares.
      env (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      float: Best advantage of t-threshold RPC we can obtain among all the t in
      {1, ..., n - 1}
  """
  eps = 1 
  for t in range (1, n) :
    eps = min(eps, tRPC_from_cRPC_t(n, t, env))
  return eps



## -----------------------------------------------------------------------------
##                        General RPP.
## -----------------------------------------------------------------------------
## Partitions are represented as byte sequences. Two indices belong to the same
## part if and only if their corresponding bytes are equal. Labels are assigned
## canonically in order of first occurrence (starting from 0).
##
## General RPP envelopes are represented as dictionaries where the keys are byte 
## sequences (i.e., partitions) and the values are probabilities.

def init_env_gRPP (n) :
  """
  Initialize the general RPP envelope.

  The initial partition consists of n singleton parts, meaning that all indices
  belong to distinct parts.

  Args:
      n (int): Number of elements in the partition.

  Returns:
      dict[bytes, float]: Initial general RPP envelope, mapping the initial
      partition  to probability 1.
  """
  env_gRPP = {}
  part_init = bytes(range(n))
  env_gRPP[part_init] = 1.0
  return env_gRPP

def __canonize_partitions__(rgs):
  """
  Canonicalize a partition represented as a restricted growth string (RGS).

  The labels are reassigned to consecutive integers starting from 0, following
  their order of first occurrence.

  Args:
      rgs (bytes | bytearray): Partition represented as a sequence of integer
      labels in [0, 255].

  Returns:
      bytes: Canonical representation of the partition.
  """  
  # Fast mapping table for byte values
  map = [-1] * 256
  out = bytearray(len(rgs))
  nxt = 0
  for i, b in enumerate(rgs):
    m = map[b]
    if m == -1:
      m = nxt
      map[b] = m
      nxt += 1
    out[i] = m
  return bytes(out)
    
def to_blocks(rgs_b: bytes):
  blocks = {}
  for i, b in enumerate(rgs_b):
    blocks.setdefault(b, []).append(i)
  return list(blocks.values())

def __update_items_chunk__(items, i, j, p):
  """
  Update a list of general RPP partition/probability pairs according to the
  leakage of the random value r involved in the updates of z_i and z_j.

  Args:
      items (list[tuple[bytes, float]]): List of pairs consisting of a partition
      and its associated probability.
      i (int): First updated index, corresponding to z_i = z_i + r.
      j (int): Second updated index, corresponding to z_j = z_j - r.
      p (float): Leakage probability.

  Returns:
      dict[bytes, float]: Updated general RPP envelope, mapping each partition
      to its associated probability.
  """
  new_env_gRPP = {}
  p = float(p)
  q = 1.0 - p

  # Precompute powers
  q2 = q * q                     # (1-p)^2
  q3 = q2 * q                    # (1-p)^3
  q4 = q3 * q                    # (1-p)^4
  q5 = q4 * q                    # (1-p)^5

  one_minus_q3 = 1.0 - q3        # 1 - (1-p)^3
  p2 = p * p                     # p^2

  def add(part_b: bytes, pb: float, coeff: float):
    if coeff == 0.0:
      return
    new_env_gRPP[part_b] = new_env_gRPP.get(part_b, 0.0) + pb * coeff

  # Helpers: we use temporary labels 254/255; after canonization labels are 
  # 0..<=31 (for n<=32)
  def isolate(pi_b: bytes, idxs, label=255) -> bytes:
    lst = bytearray(pi_b)
    for t in idxs:
      lst[t] = label
    return __canonize_partitions__(lst)

  for pi_b, pb in items:
    idx_i = pi_b[i]
    idx_j = pi_b[j]

    if idx_i == idx_j:
      # Table 1

      # Case 1: partition stays the same
      coeff = (one_minus_q3) * q2 + 2.0 * p * q4 + q5
      add(pi_b, pb, coeff)

      # Case 2: {i,j} isolated together
      part = isolate(pi_b, (i, j), label=255)
      coeff = q3 * p2
      add(part, pb, coeff)

      # Case 3: {j} isolated
      part = isolate(pi_b, (j,), label=255)
      coeff = p * q * one_minus_q3
      add(part, pb, coeff)

      # Case 4: {i} isolated
      part = isolate(pi_b, (i,), label=255)
      coeff = p * q * one_minus_q3
      add(part, pb, coeff)

      # Case 5: {i} isolated and {j} isolated (two singletons)
      lst = bytearray(pi_b)
      lst[i] = 254
      lst[j] = 255
      part = __canonize_partitions__(lst)
      coeff = p2 * one_minus_q3
      add(part, pb, coeff)

    else:
      # Table 2

      # Case 1: same partition
      coeff = one_minus_q3 * q2
      add(pi_b, pb, coeff)

      # Case 2: i and j both isolated
      lst = bytearray(pi_b)
      lst[i] = 254
      lst[j] = 255
      part = __canonize_partitions__(lst)
      coeff = p2 * one_minus_q3
      add(part, pb, coeff)

      # Case 3: j isolated
      part = isolate(pi_b, (j,), label=255)
      coeff = p * q * one_minus_q3
      add(part, pb, coeff)

      # Case 4: i isolated
      part = isolate(pi_b, (i,), label=255)
      coeff = p * q * one_minus_q3
      add(part, pb, coeff)

      # Case 5: {i,j} isolated together
      part = isolate(pi_b, (i, j), label=255)
      coeff = p2 * q3
      add(part, pb, coeff)

      # Case 6: move j into i's block
      lst = bytearray(pi_b)
      lst[j] = idx_i
      part = __canonize_partitions__(lst)
      coeff = q4 * p
      add(part, pb, coeff)

      # Case 7: move i into j's block
      lst = bytearray(pi_b)
      lst[i] = idx_j
      part = __canonize_partitions__(lst)
      coeff = q4 * p
      add(part, pb, coeff)

      # Case 8: union blocks labeled idx_i and idx_j
      lst = bytearray(pi_b)
      for t, lab in enumerate(lst):
        if lab == idx_j:
          lst[t] = idx_i
      part = __canonize_partitions__(lst)
      coeff = q5
      add(part, pb, coeff)
  return new_env_gRPP

def update(p, i, j, env_gRPP, random_added_by_index):
  """
  Update the general RPP envelope after adding the same random value to indices
  i and j, and update the corresponding randomness counters.

  Assumes i < j.

  Args:
      p (float): Leakage probability.
      i (int): First updated index.
      j (int): Second updated index.
      env_gRPP (dict[bytes, float]): Current general RPP envelope.
      random_added_by_index (list[int]): Number of random values added to each
      index.

  Returns:
      dict[bytes, float]: Updated general RPP envelope.
  """
  items = list(env_gRPP.items())
  new_env_gRPP = __update_items_chunk__(items, i, j, p)
  random_added_by_index[i] += 1
  random_added_by_index[j] += 1
  return new_env_gRPP

def gRPP_output_manipulation(n, p, env_gRPP, random_added_by_index):
  """
  Update the general RPP envelope with an additional manipulation of the output
  of the zero encoding.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      env_gRPP (dict[bytes, float]): Current general RPP envelope.
      random_added_by_index (list[int]): Number of random values added to each
      index.

  Returns:
      dict[bytes, float]: Updated general RPP envelope.
  """
  active = [i for i, x in enumerate(random_added_by_index[:n]) if x >= 2]
  k = len(active)

  q = 1.0 - p
  new_envgRPP = {}

  mask_probs = []
  for mask in range(1 << k):
    r = mask.bit_count()
    mask_probs.append((p ** r) * (q ** (k - r)))

  for part_b, proba_part in env_gRPP.items():
    base = bytearray(part_b)

    for mask in range(1 << k):
      lst = bytearray(base)
      cpt = 255

      for j, i in enumerate(active):
        if mask & (1 << j):
          lst[i] = cpt
          cpt -= 1

      new_part = __canonize_partitions__(lst)
      new_envgRPP[new_part] = new_envgRPP.get(new_part, 0.0) + proba_part * mask_probs[mask]

  return new_envgRPP

def print_env_gRPP(env_gRPP):
  """
  Print the general RPP envelope.

  Args:
      env_gRPP (dict[bytes, float]): Current general RPP envelope.
  """
  str_res = ""
  final_proba = 0.0

  for part_b, prob in env_gRPP.items():
    blocks = to_blocks(part_b)

    str_res += "{"
    str_res += ", ".join("{" + ",".join(map(str, B)) + "}" for B in blocks)
    str_res += "} \n"

    str_res += f"Probability = {prob}\n\n"
    final_proba += prob

  print(str_res)
  print("Sum of probabilities =", final_proba)

def _join_rgs(P: bytes, Q: bytes) -> bytes:
  """
  Compute the join of two partitions P and Q, i.e., the finest partition
  that is coarser than both P and Q.

  Args:
      P (bytes): First partition, represented as a byte sequence of length n.
      Q (bytes): Second partition, represented as a byte sequence of length n.

  Returns:
      bytes: Canonical representation of the join of P and Q.
  """
  n = len(P)
  assert len(Q) == n

  parent = list(range(n))
  rank = [0] * n

  def find(x):
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb:
      return
    if rank[ra] < rank[rb]:
      parent[ra] = rb
    else:
      parent[rb] = ra
      if rank[ra] == rank[rb]:
        rank[ra] += 1

  # union via labels of P
  first = [-1] * 256
  for i, lab in enumerate(P):
    j = first[lab]
    if j == -1:
      first[lab] = i
    else:
      union(i, j)

  # union via labels of Q
  first = [-1] * 256
  for i, lab in enumerate(Q):
    j = first[lab]
    if j == -1:
      first[lab] = i
    else:
      union(i, j)

  r = bytearray(n)
  for i in range(n):
    r[i] = find(i) & 0xFF
  return __canonize_partitions__(r)


def treeAdd_ze(n, p, env1_gRPP, random_added_by_index1, env2_gRPP, random_added_by_index2) :
  """
  Compute the general RPP envelope of the composition of two zero encodings with
  known general RPP envelope, using Lemma 2 of our paper.

  Args:
      n (int): Number of shares.
      p (float)): Leakage rate.
      env1_gRPP (dict[bytes, float]): General RPP envelope of the first zero 
      encoding.
      random_added_by_index1 (list[int]): Number of random values added to each
      index for the first zero encoding.
      env2_gRPP (dict[bytes, float]): General RPP envelope of the second zero 
      encoding.
      random_added_by_index2 (list[int]): Number of random values added to each
      index for the second zero encoding.

  Returns:
      dict[bytes, float]: General RPP envelope of the composed zero encoding.
  """
  env_gRPP_compo = {}
  for part_P, proba_part_P in env1_gRPP.items():
    for part_Q, proba_part_Q in env2_gRPP.items():
      proba_part_W = proba_part_P * proba_part_Q
      part_W = _join_rgs(part_P, part_Q)
      env_gRPP_compo[part_W] = env_gRPP_compo.get(part_W, 0.0) + proba_part_W

  random_added_by_index = [random_added_by_index1[i] + random_added_by_index2[i] for i in range (n)] 

  env_gRPP_compo = gRPP_output_manipulation(n, p, env_gRPP_compo, random_added_by_index)
  return env_gRPP_compo, random_added_by_index


## -----------------------------------------------------------------------------
##                        General RPC.
## -----------------------------------------------------------------------------
## Input set and output set are represented as bitsets.

def env_gRPC_atomic (n, p, i, j) :
  """
  Compute the general RPC envelope of the atomic function on indices i and j
  (Definition 11 of our paper), using Lemma 7 of our paper.

  Args:
      n (int): Number of shares.
      p (float): Leakage probability.
      i (int): First index.
      j (int): Second index.

  Returns:
      dict[tuple[int, int], float]: General RPC envelope.
  """
  env = np.zeros((4, 4))
  
  env[0, 0] =  (1 - p)**2
  env[1, 0] = p * (1 - p)
  env[2, 0] = p * (1 - p)
  env[3, 0] = p**2

  env[0, 1] = (1 - p)**5
  env[1, 1] = 1 - p - (1 - p)**5
  env[2, 1] = p * (1 - p)**4
  env[3, 1] = p * (1 - (1 - p)**4)

  env[0, 2] = (1 - p)**5
  env[1, 2] = p * (1 - p)**4
  env[2, 2] = 1 - p - (1 - p)**5
  env[3, 2] = p * (1 - (1 - p)**4)

  env[0, 3] = 0.0
  env[1, 3] = 0.0
  env[2, 3] = 0.0
  env[3, 3] = 1.0


  env_gen = {}
  for J in range (1 << n) :
    case_J = 0
    if (J & (1 << i)) : 
      case_J += 1
    
    if (J & (1 << j)) :
      case_J += 2
    
    for I in range (1 << n) :
      env_gen[(I, J)] = 0.0

      if (I == (J & ~((1 << i) | (1 << j)))) :         
        env_gen[(I, J)] += env[0, case_J]
      
      if (I == ((J & ~(1 << j)) | (1 << i))) :         
        env_gen[(I, J)] += env[1, case_J]
      
      if (I == ((J & ~(1 << i)) | (1 << j))) :      
        env_gen[(I, J)] += env[2, case_J]
      
      if (I == (J | (1 << i) | (1 << j))) : 
        env_gen[(I, J)] += env[3, case_J]
      
  return env_gen 

def compo_env_gRPC(n, first_env, second_env) :
  """
  Compose general RPC envelopes of two refresh gadgets.

  Args:
      n (int): Number of shares.
      first_env (dict[tuple[int, int], float]): General RPC envelope of the
      first refresh gadget.
      second_env (dict[tuple[int, int], float]): General RPC envelope of the
      second refresh gadget.

  Returns:
      dict[tuple[int, int], float]: General RPC envelope of the composed refresh
      gadget.
  """
  new_env_gen = {}
  for I in range (1 << n) :
    for J in range (1 << n) :
      for L in range (1 << n) :
        new_env_gen[(I, J)] = new_env_gen.get((I, J), 0.0) + first_env[(I, L)] * second_env[(L, J)]
  return new_env_gen

## -----------------------------------------------------------------------------
##                        Cardinal RPC.
## -----------------------------------------------------------------------------

def env_cRPC_init (n) :
  """
  Initialisation of cardinal RPC envelope.

  Args:
      n (int): Number of shares.

  Returns:
      np.ndarray: Cardinal RPC envelope initialised.
  """
  pgref = np.zeros((n + 1, n + 1))
  for tin in range (n + 1) :
    pgref[tin, tin] = 1.0
  return pgref

def env_cRPC_atomic (n, p, i, j) :
  """
  Cardinal RPC envelope of atomic refresh (Definition 11 of our paper.) 

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      i (int): First index.
      j (int): Second index.

  Returns:
      np.ndarray: Cardinal RPC envelope of the atomic refresh.
  """
  env_cRPC = np.zeros((n + 1, n + 1))


  tmp1 = p**2
  tmp2 = max(p * (1 - p), p * (1 - (1 - p)**4))
  tmp3 = max((1 - p)**2, p * (1 - p)**4, 1 - p - (1 - p)**5, 1.0)
  tmp4 = max((1 - p)**5, 0.0)
  tmp5 = 0.0

  for tin in range (n + 1) :

    if (tin - 2 >= 0) :
      env_cRPC[tin, tin - 2] = tmp1
    
    if (tin - 1 >= 0) : 
      env_cRPC[tin, tin - 1] = tmp2
    
    env_cRPC[tin, tin] = tmp3

    if (tin + 1 <= n) : 
      env_cRPC[tin, tin + 1] = tmp4
    
    if (tin + 2 <= n) :
      env_cRPC[tin, tin + 2] = tmp5
  
  return env_cRPC

def compo_env_cRPC(n, first_env, second_env) :
  """
  Compose cardinal RPC envelopes of two refresh gadgets.

  Args:
      n (int): Number of shares.
      first_env (np.ndarray): Cardinal RPC envelope of the first refresh gadget.
      second_env (np.ndarray): Cardinal RPC envelope of the second refresh 
      gadget.

  Returns:
      np.ndarray: Cardinal RPC envelope of the composed refresh gadget.
  """
  new_env = np.zeros((n + 1, n + 1))
  for tin in range (n  + 1) :
    for tout in range (n + 1) :
      for l in range (n + 1) :
        new_env[tin, tout] += first_env[tin, l] * second_env[l, tout]
  return new_env

## -----------------------------------------------------------------------------
##                1.Gadget RPRefresh (cardinal RPC)
## -----------------------------------------------------------------------------

############################## Direct Version ##################################

def env_cRPC_di_RandomAtomic(n, p):
  """
  Cardinal RPC envelope of atomic refresh with random index. 

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns:
      np.ndarray: Cardinal RPC envelope of the atomic refresh with random index.
  """

  env = np.zeros((n + 1, n + 1))

  for tin in range (n + 1) :
    if (tin - 2 >= 0) :
      env[tin, tin - 2] = ((n - tin + 2) * (n - tin + 1) * p**2) / (n * (n - 1))
    if (tin - 1 >= 0) :
      env[tin, tin - 1] = ((n - tin + 1) * (n - tin) * 2 * p * (1 - p)) / (n * (n - 1))
      env[tin, tin - 1] += (2 * (tin - 1) * (n - tin + 1) * p * (1 - (1 - p)**4)) / (n * (n - 1))

    env[tin, tin] = ((n - tin) * (n - tin - 1) * (1 - p)**2) / (n * (n - 1))
    env[tin, tin] += (2 * tin * (n - tin) * (1 - p - (1 - p)**5 + p * (1 - p)**4)) / (n * (n - 1))
    env[tin, tin] += (tin * (tin - 1)) / (n * (n - 1))

    if (tin + 1 <= n) :
      env[tin, tin + 1] = (2 * (tin + 1) * (n - tin - 1) * (1 - p)**5) / (n * (n - 1))

  return env


def env_cRPC_di_RPRefresh (n, p, nb_iter):
  """
  Cardinal RPC envelope of the direct version of the RPRefresh gadget.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      nb_iter (int): Number of iterations of the RPRefresh.

  Returns:
      np.ndarray: Cardinal RPC envelope of the direct RPRefersh.
  """
  if (n == 1) :
    return None 
  
  pgref_RandomAtomic = env_cRPC_di_RandomAtomic(n, p)
  pgref = env_cRPC_init(n)
  for _ in range (nb_iter) :
    pgref = compo_env_cRPC(n, pgref, pgref_RandomAtomic)
  return pgref

########################### Zero Encoding Version ##############################

def generate_cardinal_set(n):
  """
  Generate all integer partitions of n.

  Args:
      n (int): Number of shares.

  Returns:
      list[tuple[int, ...]]: List of all cardinal partitions of n.
  """
  stack = [(n, 1, [])]  # (remaining_n, start_value, current_comp)
  cardinal_sets = set()
    
  while stack:
    remaining_n, start_value, current_comp = stack.pop()

    for i in range(start_value, remaining_n + 1):
      new_comp = current_comp + [i]

      if remaining_n - i == 0:
        cardinal_sets.add(tuple(sorted(new_comp)))
      else:
        stack.append((remaining_n - i, i, new_comp))
    
  return sorted(list(cardinal_sets))

def initial_probability(cardinal_set):
  """
  Return the initial probability associated with a cardinal partition.

  Args:
      cardinal_set (tuple[int, ...]): Cardinal partition.

  Returns:
      np.float64: Initial probability of the partition.
  """

  if max(cardinal_set)==1:
    return np.float64(1.0)
  else:
    return np.float64(0.0)

def new_probabilities(int_partitions_list, probabilities, binom_coeffs, p, n):

  """
  Compute the probabilities of all cardinal partitions after one iteration.

  Args:
      int_partitions_list (list[tuple[int, ...]]): List of cardinal partitions.
      probabilities (list[float]): Current partition probabilities.
      binom_coeffs (dict[tuple[int, int], float]): Precomputed binomial coefficients.
      p (float): Leakage probability.
      n (int): Number of shares.

  Returns:
      list[float]: Updated partition probabilities.
  """    

  initial_probabilities = list(probabilities)
  new_probabilities = [np.float64(0.0)] * len(probabilities)

  cardinal_sets_dict = {tuple(cs): idx for idx, cs in enumerate(int_partitions_list)}

  z1z2_leak = p*p
  z1z2_dont_leak = 1-2*p+p*p
  r_doesnot_leak = 1 - (3*p - 3*p*p + p*p*p)
  r_leaks = 3*p - 3*p*p + p*p*p
  one_zi_leaks = p-p*p


  for ind_cs, cs in enumerate(int_partitions_list):
        
        if initial_probabilities[ind_cs] == 0.0:
            continue

        cs_count = Counter(cs)

        
        #Both drawn indices belong to the same set           
        for cardinal_i, freq in cs_count.items():

            if cardinal_i < 2:
                continue
     
            proba_same_set_i = np.float64(cardinal_i)*np.float64(cardinal_i-1)/(np.float64(n)*np.float64(n-1))
            proba_for_update = initial_probabilities[ind_cs]*proba_same_set_i*freq

            i = cs.index(cardinal_i)

            # 1. i1 and i2 are in the same set, zi1, zi2 and r leak 
            #   => [1,1,c1,...,c{i-1},c{i}-2,c{i+1},...,cn]
            updated_set_1 = list(cs)
            if updated_set_1[i]==2:
                del updated_set_1[i]
            else:
                updated_set_1[i] -= 2
            updated_set_1.extend([1, 1])
            updated_proba_1 = z1z2_leak*r_leaks
            index_1 = find_cardinal_set_index(cardinal_sets_dict,updated_set_1)
            new_probabilities[index_1] += proba_for_update*updated_proba_1

            # 2. i1 and i2 are in the same set, zi1, zi2 leak, r does not leak
            #   => [2,c1,...,c{i-1},c{i}-2,c{i+1},...,cn]
            updated_set_2 = list(cs)
            if updated_set_2[i]>2:
                updated_set_2[i] -= 2
                updated_set_2.append(2)
                index_2 = find_cardinal_set_index(cardinal_sets_dict,updated_set_2)
            else:
                index_2 = ind_cs
            updated_proba_2 = z1z2_leak*r_doesnot_leak
            new_probabilities[index_2] += proba_for_update*updated_proba_2

            # 3/5. i1 and i2 are in the same set, exactly 1 of zi1/zi1 leaks, r leaks
            #   => [1,c1,...,c{i-1},c{i}-1,c{i+1},...,cn]
            updated_set_3 = list(cs)
            updated_set_3[i] -= 1
            updated_set_3.append(1)
            updated_proba_3 = 2*one_zi_leaks*r_leaks
            index_3 = find_cardinal_set_index(cardinal_sets_dict,updated_set_3)
            new_probabilities[index_3] += proba_for_update*updated_proba_3

            # 4/6. i1 and i2 are in the same set, exactly 1 of zi1/zi1 leaks, r does not leak
            #   => initial partition
            updated_proba_4 = 2*one_zi_leaks*r_doesnot_leak

            # 7/8. i1 and i2 are in the same set, zi1 does not leak, z2 does not leak
            #   => initial partition
            updated_proba_7 = z1z2_dont_leak
            new_probabilities[ind_cs] += proba_for_update*(updated_proba_4+updated_proba_7)

        
        #The two drawn indices belong to different sets. 
        for i1 in range(len(cs)):
            for i2 in range(i1+1,len(cs)):

                proba_sets_i1_i2 = np.float64(2.0)*np.float64(cs[i1])*np.float64(cs[i2])/(np.float64(n)*np.float64(n-1))
                proba_for_update = initial_probabilities[ind_cs]*proba_sets_i1_i2

                # If n_{i2} = 1, then n_{i1}=1 because they are ordered,
                if cs[i2]==1:
                    # 9/11/13/15. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r leaks
                    #   => [1,1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                    updated_proba_9 = z1z2_leak*r_leaks
                    new_probabilities[ind_cs] += proba_for_update*r_leaks

                    # 10/12/14/16. i1 and i2 are not in the same set, r does not leak
                    #   => [2,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                    updated_set_10 = list(cs) 
                    updated_set_10[i2] +=1
                    del updated_set_10[i1]
                    index_10 = find_cardinal_set_index(cardinal_sets_dict,updated_set_10)
                    new_probabilities[index_10] += proba_for_update*r_doesnot_leak

                else:
                    # cs[i2]>1

                    if cs[i1]==1:
                        # 9. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r leaks
                        #   => [1,1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_9 = list(cs)
                        updated_set_9[i2] -= 1
                        updated_set_9.append(1)
                        updated_proba_9 = z1z2_leak*r_leaks
                        index_9 = find_cardinal_set_index(cardinal_sets_dict,updated_set_9)

                        # 10. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r does not leak
                        #   => [2,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_10 = list(cs) 
                        updated_set_10[i2] -= 1
                        updated_set_10[i1] += 1
                        updated_proba_10 = z1z2_leak*r_doesnot_leak
                        index_10 = find_cardinal_set_index(cardinal_sets_dict,updated_set_10)

                        # 11. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2},c{i2+1},...,cn]
                        updated_proba_11 = one_zi_leaks*r_leaks

                        # 12. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}+1,c{i2+1},...,cn]
                        updated_set_12 = list(cs) 
                        updated_set_12[i2] +=1
                        del updated_set_12[i1]
                        updated_proba_12 = one_zi_leaks*r_doesnot_leak
                        index_12 = find_cardinal_set_index(cardinal_sets_dict,updated_set_12)

                        # 13. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1},c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_proba_13 = one_zi_leaks*r_leaks

                        # 14. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}+1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_proba_14 = one_zi_leaks*r_doesnot_leak

                        # 15. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r leaks
                        #   => ensemble initial
                        updated_proba_15 = z1z2_dont_leak*r_leaks

                        # 16. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1+1},...,c{i2-1},c{i2+1},...,cn,c{i1}+c{i2}]
                        updated_proba_16 = z1z2_dont_leak*r_doesnot_leak

                        new_probabilities[ind_cs] += proba_for_update*(updated_proba_11+updated_proba_15)
                        new_probabilities[index_9] += proba_for_update*(updated_proba_9+updated_proba_13)
                        new_probabilities[index_10] += proba_for_update*(updated_proba_10+updated_proba_14)
                        new_probabilities[index_12] += proba_for_update*(updated_proba_12+updated_proba_16)

                    else:   
                        # cs[i1]>1

                        # 9. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r leaks
                        #   => [1,1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_9 = list(cs)
                        updated_set_9[i1] -= 1
                        updated_set_9.append(1)
                        updated_set_9[i2] -= 1
                        updated_set_9.append(1)
                        updated_proba_9 = z1z2_leak*r_leaks
                        index_9 = find_cardinal_set_index(cardinal_sets_dict,updated_set_9)
                        new_probabilities[index_9] += proba_for_update*updated_proba_9

                        # 10. i1 and i2 are not in the same set, zi1 leaks, z2 leaks, r does not leak
                        #   => [2,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_10 = list(cs) 
                        updated_set_10[i2] -= 1
                        updated_set_10[i1] -= 1
                        updated_set_10.append(2)
                        updated_proba_10 = z1z2_leak*r_doesnot_leak
                        index_10 = find_cardinal_set_index(cardinal_sets_dict,updated_set_10)
                        new_probabilities[index_10] += proba_for_update*updated_proba_10

                        # 11. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2},c{i2+1},...,cn]
                        updated_set_11 = list(cs)
                        updated_set_11[i1] -= 1
                        updated_set_11.append(1)
                        updated_proba_11 = one_zi_leaks*r_leaks
                        index_11 = find_cardinal_set_index(cardinal_sets_dict,updated_set_11)
                        new_probabilities[index_11] += proba_for_update*updated_proba_11

                        # 12. i1 and i2 are not in the same set, zi1 leaks, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}-1,c{i1+1},...,c{i2-1},c{i2}+1,c{i2+1},...,cn]
                        updated_set_12 = list(cs) 
                        updated_set_12[i2] +=1
                        updated_set_12[i1] -=1
                        updated_proba_12 = one_zi_leaks*r_doesnot_leak
                        index_12 = find_cardinal_set_index(cardinal_sets_dict,updated_set_12)
                        new_probabilities[index_12] += proba_for_update*updated_proba_12

                        # 13. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r leaks
                        #   => [1,c1,...,c{i1-1},c{i1},c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_13 = list(cs)
                        updated_set_13[i2] -= 1
                        updated_set_13.append(1)
                        updated_proba_13 = one_zi_leaks*r_leaks
                        index_13 = find_cardinal_set_index(cardinal_sets_dict,updated_set_13)
                        new_probabilities[index_13] += proba_for_update*updated_proba_13

                        # 14. i1 and i2 are not in the same set, zi1 does not leak, z2 leaks, r does not leak
                        #   => [c1,...,c{i1-1},c{i1}+1,c{i1+1},...,c{i2-1},c{i2}-1,c{i2+1},...,cn]
                        updated_set_14 = list(cs)
                        updated_set_14[i2] -=1
                        updated_set_14[i1] +=1
                        updated_proba_14 = one_zi_leaks*r_doesnot_leak
                        index_14 = find_cardinal_set_index(cardinal_sets_dict,updated_set_14)
                        new_probabilities[index_14] += proba_for_update*updated_proba_14

                        # 15. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r leaks
                        #   => initial set
                        updated_proba_15 = z1z2_dont_leak*r_leaks
                        new_probabilities[ind_cs] += proba_for_update*updated_proba_15

                        # 16. i1 and i2 are not in the same set, zi1 does not leak, z2 does not leak, r does not leak
                        #   => [c1,...,c{i1-1},c{i1+1},...,c{i2-1},c{i2+1},...,cn,c{i1}+c{i2}]
                        updated_set_16 = list(cs)
                        updated_set_16[i2] += updated_set_16[i1]
                        del updated_set_16[i1]
                        updated_proba_16 = z1z2_dont_leak*r_doesnot_leak
                        index_16 = find_cardinal_set_index(cardinal_sets_dict,updated_set_16)
                        new_probabilities[index_16] += proba_for_update*updated_proba_16

  return new_probabilities

def find_cardinal_set_index(int_partitions_dict, target_partition):
  """
  Return the index associated with a target cardinal partition.

  Args:
      int_partitions_dict (dict[tuple[int, ...], int]): Mapping from cardinal
      partitions to their indices.
      target_partition (list[int]): Target cardinal partition.

  Returns:
      int | None: Index of the target partition, or None if it is not found.
  """

  target_partition = tuple(sorted(target_partition))
  return int_partitions_dict.get(target_partition)


def cRPP_RPRefreh(n,nb_iter,p, binom_coeffs, int_partitions_list):
  """
  Compute the cardinal RPP envelope of the zero encoding ZeroEnc used in
  RPRefresh.

  Args:
      n (int): Number of shares.
      nb_iter (int): Number of iterations.
      p (float): Leakage rate.
      binom_coeffs (dict[tuple[int, int], float]): Precomputed binomial coefficients.
      int_partitions_list (list[tuple[int, ...]]): List of cardinal partitions.

  Returns:
      list[float]: Cardinal RPP envelope of ZeroEnc^+.
  """
    
  initial_probabilities = [initial_probability(cs) for cs in int_partitions_list]
  env_cRPP = initial_probabilities
  for i in range(nb_iter):
    new_env_cRPP = new_probabilities(int_partitions_list, env_cRPP, binom_coeffs, p, n)
    env_cRPP = new_env_cRPP

  # leakage on the shares (z_i) during the addition with the inputs
  env_cRPP_out = cRPP_output_manipulation (int_partitions_list, env_cRPP, binom_coeffs, p, n)
  return env_cRPP_out

def env_cRPC_ze_RPRefresh(n, p, nb_iter):
  """
  Cardinal RPC envelope of the zero encoding RPRefresh.

  Args:
      n (int): Number of shares.
      p (float): Leakage Rate.
      nb_iter (int): Number of iterations.

  Returns:
      np.ndarray: Cardinal RPC envelope of the zero encoding version of the
      RPRefresh. 
  """
  binom_coeffs = precompute_binomial_coefficients(n)
  int_partitions_list = generate_cardinal_set(n)
  env_cRPP = cRPP_RPRefreh(n, nb_iter, p, binom_coeffs, 
                                int_partitions_list)
  pgref = cRPC_from_cRPP(n, p, env_cRPP)
  return pgref

def env_gRPC_ze_RPRefresh(n, p, nb_iter):
  """
  Compute the general RPC envelope of the zero-encoding version of RPRefresh.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      nb_iter (int): Number of iterations of RPRefresh.

  Returns:
      dict[tuple[int, int], float]:
          General RPC envelope. Input and output sets are represented as
          bitsets: env_gRPC[(I, J)] = E^g_J(I).
  """

  binom_coeffs = precompute_binomial_coefficients(n)
  int_partitions_list = generate_cardinal_set(n)

  env_cRPP = cRPP_RPRefreh(
      n,
      nb_iter,
      p,
      binom_coeffs,
      int_partitions_list
  )

  # q[k, j] is the probability that k input shares are recovered from
  # the internal leakage and a set J of j output shares.
  q = np.zeros((n + 1, n + 1))

  for j in range(n + 1):
    probas_x_outputs = compute_probas_x_outputs(
        n,
        j,
        int_partitions_list,
        env_cRPP,
        binom_coeffs
    )

    for k in range(j + 1):
      q[k, j] = probas_x_outputs[k]

  # Since the expression only depends on |I| and |J|, first compute one
  # value for each pair of cardinalities.
  env_by_card = np.zeros((n + 1, n + 1))

  for card_I in range(n + 1):
    for card_J in range(n + 1):

      proba = 0.0

      for card_K in range(min(card_I, card_J) + 1):

        # There are binom(|I|, |K|) subsets K of I of cardinality |K|.
        proba += (
            binom_coeffs[(card_I, card_K)]
            * p**(card_I - card_K)
            * (1 - p)**(n - card_I)
            * q[card_K, card_J]
            / binom_coeffs[(n, card_K)]
        )

      env_by_card[card_I, card_J] = proba

  env_gRPC = {}

  for I in range(1 << n):
    card_I = I.bit_count()

    for J in range(1 << n):
      card_J = J.bit_count()
      card_IJ = (I & J).bit_count()

      proba = 0.0

      for card_K in range(card_IJ + 1):
        proba += (
          math.comb(card_IJ, card_K)
          * p**(card_I - card_K)
          * (1 - p)**(n - card_I)
          * q[card_K, card_J]
          / math.comb(card_J, card_K)
        )

      env_gRPC[(I, J)] = proba

  return env_gRPC

## -----------------------------------------------------------------------------
##                2. Gadget Ideal Refresh (gen RPC & card RPC)
## -----------------------------------------------------------------------------

############################### Direct Version #################################


def env_cRPC_di_ideal (n, p) :
  """
  Cardinal RPC envelope of the ideal Direct Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the ideal Direct Refresh gadget.
  """ 
  env = np.zeros((n + 1, n + 1))
  for tout in range (n) :
    for tin in range (n) :
      env[tin, tout] = math.comb(n, tin) * p**tin * (1 - p)**(n - tin) * (1 - p**(2 * (n - tout)))
    
    env[n, tout] = p**n + (1 - p**n) *  p**(2 * (n - tout))

  env[n, n] = 1
  return env

def env_gRPC_di_ideal (n, p) :
  """
  General RPC envelope of the ideal Direct Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the ideal Direct
      Refresh gadget.
  """ 
  env_gen = {}
  lim = (1 << n) - 1 
  for I in range (lim) : 
    card_I = I.bit_count()
    comp = p**card_I * (1 - p)**(n - card_I)
    for J in range (lim) :
      card_J = J.bit_count()
      env_gen[(I, J)] = comp * (1 - p**(2 * (n - card_J)))

    env_gen[(I, lim)] = 0

  for J in range (lim) :
    env_gen[(lim, J)] = p**n + p**(2 * (n - J.bit_count())) * (1 - p**n)

  env_gen[(lim, lim)] = 1 
  return env_gen
  
########################### Zero Encoding Version ##############################

def env_cRPC_ze_ideal (n, p) :
  """
  Cardinal RPC envelope of the ideal Zero Encoding Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the ideal Zero Encoding Refresh 
      gadget.
  """ 
  env = np.zeros((n + 1, n + 1))
  p1 = p + (1 - p) * p**2
  
  tout = 0
  for tin in range (n + 1) :
    env[tin, tout] += math.comb(n, tin) * p**(tin) * (1 - p)**(n - tin)
  
  for tout in range (1, n) :
    for tin in range (tout) :
      for i in range (tin + 1) :
        env[tin, tout] += math.comb(tout, i) * p1**i * (1 - p1)**(tout - i) * math.comb(n - i, tin - i) * p**(tin - i) * (1 - p)**(n - tin)
      env[tin, tout] *= (1 - p1**(n - tout))

    for tin in range (tout, n + 1) :
      for i in range (tout + 1) :
        env[tin, tout] += math.comb(tout, i) * p1**i * (1 - p1)**(tout - i) * math.comb(n - i, tin - i) * p**(tin - i) * (1 - p)**(n - tin)
      env[tin, tout] *= (1 - p1**(n - tout))

      env[tin, tout] += p1**(n - tout) * math.comb(n - tout, tin - tout) * p**(tin - tout) * (1 - p)**(n - tin)
      
  tout = n    
  tin = n
  env[tin, tout] = 1.0          
  return env

def env_gRPC_ze_ideal(n, p) :
  """
  General RPC envelope of the ideal Zero Encoding Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the ideal Zero 
      Encoding Refresh gadget.
  """ 
  p1 = p + (1 -p)* p**2
  env_gen = {}
  lim = (1 << n) - 1 
  for I in range (lim + 1) : 
    card_I = I.bit_count()
    for J in range (lim) :
      card_J = J.bit_count()
      IandJ = I & J
      card_IandJ = IandJ.bit_count()
      sm = 0
      for i in range (card_IandJ + 1) :
        sm += (math.comb(card_IandJ, i) * p1**i * (1 - p1)**(card_J - i) * 
               p**(card_I - i))
      sm *= (1 - p1**(n - card_J))

      if ((J | I) == I) :
        sm += p1**(n - card_J) * p**(card_I - card_J)

      env_gen[(I, J)] = sm * (1 - p)**(n -card_I)
    env_gen[(I, lim)] = 0


  env_gen[(lim, lim)] = 1 
  return env_gen

def env_gRPP_ze_ideal(n, p):
  """
  General RPP envelope of the ideal Zero Encoding gadget.

  Initially, all n shares belong to the same block. Each share independently
  leaves this block and becomes a singleton with probability

      p1 = p + (1 - p) * p**2.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns:
      dict[bytes, float]: General RPP envelope.
  """

  p1 = p + (1 - p) * p**2
  env_gRPP = defaultdict(float)

  # I is the set of elements leaving the main block.
  for I in range(1 << n):

    card_I = I.bit_count()

    proba = (
      p1**card_I
      * (1 - p1)**(n - card_I)
    )

    # Initially, every element belongs to the same block.
    part = bytearray(n)

    # Every element of I leaves the main block and gets its own singleton.
    label = 1
    for i in range(n):
      if I & (1 << i):
        part[i] = label
        label += 1

    # Put the partition in your canonical RGS representation.
    part = __canonize_partitions__(part)

    # Several leakage patterns can yield the same partition.
    env_gRPP[part] += proba

  return dict(env_gRPP)

def env_gRPC_ze_ideal_v2(n, p) :
  env_gRPP = env_gRPP_ze_ideal(n, p)
  env_gRPC = gRPC_from_gRPP(n, p, env_gRPP)
  return env_gRPC


def test_env_gRPC_ze_ideal(n, p, c=100):
    """
    Check that env_gRPC_ze_ideal and env_gRPC_ze_ideal_v2 are numerically
    equal up to machine precision.

    Args:
        n (int): Number of shares.
        p (float): Leakage rate.
        c (float): Multiplicative tolerance applied to machine precision.

    Returns:
        bool: True if all entries are numerically indistinguishable.
    """

    env1 = env_gRPC_ze_ideal(n, p)
    env2 = env_gRPC_ze_ideal_v2(n, p)

    eps = sys.float_info.epsilon
    tol = c * eps

    max_abs_error = 0.0
    max_rel_error = 0.0
    worst_key = None

    for key in env1:

        v1 = env1[key]
        v2 = env2[key]

        abs_error = abs(v1 - v2)

        # Relative error, except when both values are essentially zero.
        if max(abs(v1), abs(v2)) > 0:
            rel_error = abs_error / max(abs(v1), abs(v2))
        else:
            rel_error = 0.0

        if rel_error > max_rel_error:
            max_rel_error = rel_error
            max_abs_error = abs_error
            worst_key = key

        if abs_error > tol * max(1.0, abs(v1), abs(v2)):
            print(f"Mismatch for {key}:")
            print(f"  ideal    = {v1:.17e}")
            print(f"  ideal_v2 = {v2:.17e}")
            print(f"  abs err  = {abs_error:.3e}")
            print(f"  rel err  = {rel_error:.3e}")
            print(f"  tolerance = {tol:.3e}")
            return False

    print("Envelopes are numerically equal.")
    print(f"Maximum relative error: {max_rel_error:.3e}")
    print(f"Maximum absolute error: {max_abs_error:.3e}")
    print(f"Worst entry: {worst_key}")
    print(f"Machine precision: {eps:.3e}")

    return True

## -----------------------------------------------------------------------------
##                3. Gadget Linear (gen RPC & card RPC) 
## -----------------------------------------------------------------------------

########################### Zero Encoding Version ##############################

def gRPP_ze_linear (n, p) :
  """
  General RPP envelope of the Zero Encoding of the Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[bytes, float]: General RPC envelope of the Zero Encoding of the
      Linear Refresh gadget.
  """  
  env_init = init_env_gRPP(n)
  
  env_gRPP_random_added_by_index = [0] * n
  env_gRPP = update(p, 0, n - 1, env_init, env_gRPP_random_added_by_index)

  for i in range(1, n - 1) :
    atom_random_added_by_index = [0] * n
    env_atom = update(p, i, n - 1, env_init, atom_random_added_by_index)
    env_gRPP, env_gRPP_random_added_by_index = treeAdd_ze(n, p, env_gRPP, env_gRPP_random_added_by_index, env_atom, atom_random_added_by_index)

  return env_gRPP

def env_gRPC_ze_linear (n, p) :
  """
  General RPC envelope of the Zero Encoding Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero 
      Encoding Linear Refresh gadget.
  """
  env_gRPP = gRPP_ze_linear(n, p)
  return gRPC_from_gRPP(n, p, env_gRPP)

def env_cRPC_ze_linear (n, p) :
  """
  Cardinal RPC envelope of the Zero Encoding Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding Linear Refresh 
      gadget.
  """
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_linear_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_linear_n{n}_p{logp}.npy"

  if os.path.exists(filename_prec):
    return np.load(filename_prec)

  if (n == 1) :
    return None
  
  env_gRPP = gRPP_ze_linear(n, p)
  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)
  
  #Saving also this to be faster than do it again in env_UOcRPC_ze_linear
  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)


  return env_cRPC

def env_UOcRPC_ze_linear (n, p) :
  """
  UO Cardinal RPC envelope of the Zero Encoding Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Zero Encoding Linear Refresh 
      gadget.
  """
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_linear_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_linear_n{n}_p{logp}.npy"

  if os.path.exists(filename_UO_prec):
    return np.load(filename_UO_prec)

  if (n == 1) :
    return None
  
  env_gRPP = gRPP_ze_linear(n, p)
  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)

  return env_UOcRPC 

########################### Direct Version ##############################

def env_gRPC_di_linear (n, p) :
  """
  General RPC envelope of the Direct Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPP envelope of the Direct Linear 
      Refresh gadget.
  """
  env_gRPC = env_gRPC_atomic(n, p, 0, n - 1)
  for i in range (1, n - 1) :
    env_atom = env_gRPC_atomic (n, p, i, n - 1)
    env_gRPC = compo_env_gRPC(n, env_gRPC, env_atom)
  
  return env_gRPC

def env_cRPC_di_linear (n, p) :
  """
  Cardinal RPC envelope of the Direct Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Direct Linear Refresh gadget.
  """
  if (n == 1) :
    return None
  env_gRPC = env_gRPC_di_linear(n, p)
  return cRPC_from_gRPC(n, env_gRPC)

def env_UOcRPC_di_linear (n, p) :
  """
  UO Cardinal RPC envelope of the Direct Linear Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Direct Linear Refresh gadget.
  """
  if (n == 1) :
    return None
  env_gRPC = env_gRPC_di_linear(n, p)
  return UOcRPC_from_gRPC(n, env_gRPC)

########################## IronMask Verif ######################################

def IRA_env_gRPC_ze_linear (n, p) :
  """
  Compute the general RPC envelope of the Zero Encoding Linear Refresh gadget
  using IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero 
      Encoding Linear Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_gen("env_IRA/linear/gRPC/ze_linear_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_gRPC_di_linear (n, p) :
  """
  Compute the general RPC envelope of the Direct Linear Refresh gadget
  using IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Direct Linear 
      Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_gen("env_IRA/linear/gRPC/di_linear_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_cRPC_ze_linear (n, p) :
  """
  Compute the cardinal RPC envelope of the Zero Encoding Linear Refresh gadget
  using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding Linear Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/linear/cRPC/ze_linear_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_ze_linear (n, p) :
  """
  Compute the UO cardinal RPC envelope of the Zero Encoding Linear Refresh 
  gadget using IronMask.
  
  Warning: UO Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Zero Encoding Linear Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/linear/cRPC/ze_linear_"+str(n)+"shares.txt", n, p, -1)
  return pgref


def test_envcRPC_ze_linear(n, p) :
  tol = 100 * np.finfo(float).eps


  env1 = env_cRPC_ze_linear(n, p)
  env2 = IRA_env_cRPC_ze_linear (n, p)

  for tin in range (n + 1) :
    for tout in range (n + 1) :
      a = env1[tin, tout]
      b = env2[tin, tout]

      rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)

      if rel_err > tol:
        print("tin  = ", tin)
        print("tout = ", tout)
        print("rel_err = ", rel_err)
        print()
  

def IRA_env_cRPC_di_linear (n, p) :
  """
  Compute the cardinal RPC envelope of the Direct Linear Refresh gadget
  using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Direct Linear Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/linear/cRPC/di_linear_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_di_linear (n, p) :
  """
  Compute the UO cardinal RPC envelope of the Direct Linear Refresh gadget
  using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO cardinal RPC envelope of the Direct Linear Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/linear/cRPC/di_linear_"+str(n)+"shares.txt", n, p, -1)
  return pgref

## -----------------------------------------------------------------------------
##                4. Gadget nlogn Refresh (gen RPC & card RPC) :
## -----------------------------------------------------------------------------

########################### Zero Encoding Version ##############################

def gRPP_ze_nlogn_rec (n, p, env_gRPP, rabi, start_index, end_index) :
  """
  Recursively compute the general RPP envelope of the Zero Encoding of the
  O(n log n) Refresh gadget.

  Args:
      n (int): Number of shares.
      p (float): Leakage probability.
      env_gRPP (dict[bytes, float]): Current general RPP envelope.
      rabi (list[int]): Number of random values added to each index.
      start_index (int): First index of the current recursive block.
      end_index (int): Last index of the current recursive block.

  Returns:
      tuple[dict[bytes, float], list[int]]: Updated general RPP envelope and
      randomness counters.
  """ 
  env_init = init_env_gRPP(n)
  n_rec = (end_index - start_index) + 1

  if (n_rec == 2) :
    atom_rabi = [0] * n
    env_atom = update(p, start_index, end_index, env_init, atom_rabi)
    new_env, new_rabi = treeAdd_ze(n, p, env_gRPP, rabi, env_atom, atom_rabi)
    return new_env, new_rabi
  
  if (n_rec == 3) :
    atom1_rabi = [0] * n
    env_atom1 = update(p, start_index, end_index, env_init, atom1_rabi)
    
    atom2_rabi = [0] * n
    env_atom2 = update(p, start_index + 1, end_index, env_init, atom2_rabi)
    
    new_env, new_rabi = treeAdd_ze(n, p, env_gRPP, rabi, env_atom1, atom1_rabi)
    new_env, new_rabi = treeAdd_ze(n, p, new_env, new_rabi, env_atom2, atom2_rabi)

    return new_env, new_rabi 

  nh_rec = n_rec // 2

  env_start, start_rabi = gRPP_ze_nlogn_rec(n, p, env_gRPP, rabi, start_index, start_index + nh_rec - 1)
  
  end_rabi = [0] * n
  env_end,  end_rabi = gRPP_ze_nlogn_rec(n, p, env_init, end_rabi, start_index + nh_rec, end_index)
  
  new_env, new_rabi = treeAdd_ze(n, p, env_start, start_rabi, env_end, end_rabi)

  for i in range(nh_rec) :
    atom_rabi = [0] * n
    env_atom = update(p, start_index + i, start_index + nh_rec + i, env_init, atom_rabi)
    new_env, new_rabi = treeAdd_ze(n, p, new_env, new_rabi, env_atom, atom_rabi)

  return new_env, new_rabi


def gRPP_ze_nlogn (n, p) :
  """
  General RPP envelope of the Zero Encoding of the O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[bytes, float]: General RPP envelope of the Zero Encoding of the
      O(nlogn) Refresh gadget.
  """  
  env_init = init_env_gRPP(n)
  init_rabi = [0] * n

  env_gRPP, rabi = gRPP_ze_nlogn_rec(n, p, env_init, init_rabi , 0, n - 1)

  return env_gRPP

def env_gRPC_ze_nlogn (n, p) :
  """
  General RPC envelope of the Zero Encoding O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero 
      Encoding O(nlogn) Refresh gadget.
  """
  env_gRPP = gRPP_ze_nlogn(n, p)
  return gRPC_from_gRPP(n, p, env_gRPP)

def env_cRPC_ze_nlogn (n, p) :
  """
  Cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh 
      gadget.
  """
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_nlogn_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_nlogn_n{n}_p{logp}.npy"


  if os.path.exists(filename_prec):
    return np.load(filename_prec)
  
  if (n == 1) :
    return None

  env_gRPP = gRPP_ze_nlogn(n, p)
  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)

  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  return env_cRPC

def env_UOcRPC_ze_nlogn (n, p) :
  """
  UO Cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh 
      gadget.
  """
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_nlogn_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_nlogn_n{n}_p{logp}.npy"

  if os.path.exists(filename_UO_prec):
    return np.load(filename_UO_prec)
  
  if (n == 1) :
    return None

  env_gRPP = gRPP_ze_nlogn(n, p)
  env_UOcRPC =  UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)

  return env_UOcRPC


########################### Direct Version #####################################

def gRPC_di_nlogn_rec (n, p, env_gRPC, start_index, end_index) : 
  """
  Recursively compute the general RPC envelope of the Direct O(n log n) Refresh
  gadget.

  Args:
      n (int): Number of shares.
      p (float): Leakage probability.
      env_gRPC (dict[tuple[int, int], float]): Current general RPC envelope.
      start_index (int): First index of the current recursive block.
      end_index (int): Last index of the current recursive block.

  Returns:
      dict[tuple[int, int], float]: Updated general RPC envelope.
  """
  n_rec = (end_index - start_index) + 1
  if (n_rec == 2) :
    env_atom_gRPC = env_gRPC_atomic(n, p, start_index, end_index)
    new_env_gRPC = compo_env_gRPC(n, env_gRPC, env_atom_gRPC)
    return new_env_gRPC
  
  if (n_rec == 3) :

    first_env_atom_gRPC = env_gRPC_atomic(n, p, start_index, end_index)
    new_env_gRPC = compo_env_gRPC(n, env_gRPC, first_env_atom_gRPC)

    second_env_atom_gRPC = env_gRPC_atomic(n, p, start_index + 1, end_index)
    new_env_gRPC = compo_env_gRPC(n, new_env_gRPC, second_env_atom_gRPC)

    return new_env_gRPC

  nh_rec = n_rec // 2

  new_env_gRPC = gRPC_di_nlogn_rec(n, p, env_gRPC, start_index, start_index + nh_rec - 1)
  new_env_gRPC = gRPC_di_nlogn_rec(n, p, new_env_gRPC, start_index + nh_rec, end_index)

  for i in range(nh_rec) :
    env_atom = env_gRPC_atomic(n, p, start_index + i, start_index + i + nh_rec)
    new_env_gRPC = compo_env_gRPC(n, new_env_gRPC, env_atom)

  return new_env_gRPC

def env_gRPC_di_nlogn (n, p) :
  """
  General RPC envelope of the Direct O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Direct O(nlogn) 
      Refresh gadget.
  """
  env_gRPC = {}
  for I in range (1 << n) :
    for J in range (1 << n) :
      
      if (I == J) :
        env_gRPC[(I, J)] = 1.0
      
      else :
        env_gRPC[(I, J)] = 0.0

  new_env_gRPC = gRPC_di_nlogn_rec(n, p, env_gRPC, 0, n - 1)
  return new_env_gRPC

def env_cRPC_di_nlogn (n, p) :
  """
  Cardinal RPC envelope of the Direct O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Direct O(nlogn) Refresh 
      gadget.
  """ 
  if (n == 1) :
    return None
  
  env_gRPC = env_gRPC_di_nlogn(n, p)
  return cRPC_from_gRPC(n, env_gRPC)

def env_UOcRPC_di_nlogn (n, p) : 
  """
  UO Cardinal RPC envelope of the Direct O(nlogn) Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Direct O(nlogn) Refresh 
      gadget.
  """ 
  if (n == 1) :
    return None
  
  env_gRPC = env_gRPC_di_nlogn(n, p)
  return UOcRPC_from_gRPC(n, env_gRPC)

########################## IronMask Verif ######################################

def IRA_env_cRPC_ze_nlogn (n, p) :
  """
  Compute the cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh gadget
  using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/nlogn_ze_victor/cRPC/nlogn_ze_victor_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_ze_nlogn (n, p) :
  """
  Compute the UO cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh gadget
  using IronMask.
  
  Warning: UO Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Zero Encoding O(nlogn) Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/nlogn_ze_victor/cRPC/nlogn_ze_victor_"+str(n)+"shares.txt", n, p, -1)
  return pgref


def IRA_env_cRPC_di_nlogn(n, p) :
  """
  Compute the cardinal RPC envelope of the Direct O(nlogn) Refresh gadget
  using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Direct O(nlogn) Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/nlogn_in/cRPC/nlogn_in_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_di_nlogn(n, p) :
  """
  Compute the UO cardinal RPC envelope of the Direct O(nlogn) Refresh gadget
  using IronMask.
  
  Warning: UO Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Direct O(nlogn) Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/nlogn_in/cRPC/nlogn_in_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_gRPC_di_nlogn (n, p) :
  """
  Compute the general RPC envelope of the Direct O(nlogn) Refresh gadget
  using IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Direct O(nlogn) 
      Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_gen("env_IRA/nlogn_in/gRPC/nlogn_in_"+str(n)+"shares.txt", n, p, -1)
  return pgref

def IRA_env_gRPC_ze_nlogn (n, p) :
  """
  Compute the general RPC envelope of the Zero Encoding O(nlogn) Refresh gadget
  using IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero Encoding 
      O(nlogn) Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 6) : 
    raise Exception("Number of shares must be at most 6!")
  pgref = parse_enveloppes_gen("env_IRA/nlogn_ze_victor/gRPC/nlogn_ze_victor_"+str(n)+"shares.txt", n, p, -1)
  return pgref

## -----------------------------------------------------------------------------
##                5. Gadget Half Refresh (gen RPC & card RPC) :
## -----------------------------------------------------------------------------

########################### Zero Encoding Version ##############################

def gRPP_ze_half (n, p) :
  """
  General RPP envelope of the Zero Encoding of the Half Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[bytes, float]: General RPP envelope of the Zero Encoding of the
      Half Refresh gadget.
  """ 
  env_init = init_env_gRPP(n)
  
  env_gRPP_random_added_by_index = [0] * n
  env_gRPP = update(p, 0, 1, env_init, env_gRPP_random_added_by_index)

  for i in range(2, n - 1, 2) :
    atom_random_added_by_index = [0] * n
    env_atom = update(p, i, i + 1, env_init, atom_random_added_by_index)
    env_gRPP, env_gRPP_random_added_by_index = treeAdd_ze(n, p, env_gRPP, env_gRPP_random_added_by_index, env_atom, atom_random_added_by_index)

  return env_gRPP

def env_gRPC_ze_half (n, p) :
  """
  General RPC envelope of the Half Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Half Refresh 
      gadget.
  """
  env_gRPP = gRPP_ze_half(n, p)
  return gRPC_from_gRPP(n, p, env_gRPP)

def env_cRPC_ze_half (n, p) :
  """
  Cardinal RPC envelope of the Half Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Half Refresh gadget.
  """ 
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_half_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_half_n{n}_p{logp}.npy"

  if os.path.exists(filename_prec):
    return np.load(filename_prec)
  
  if (n == 1) :
    return None
  
  env_gRPP = gRPP_ze_half(n, p)
  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)
  
  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  return env_cRPC

def env_UOcRPC_ze_half (n, p) :
  """
  UO Cardinal RPC envelope of the Half Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Half Refresh gadget.
  """
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_half_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_half_n{n}_p{logp}.npy"

  if os.path.exists(filename_prec):
    return np.load(filename_prec)
  
  if (n == 1) :
    return None
  
  env_gRPP = gRPP_ze_half(n, p)
  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)
  
  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  return env_UOcRPC

############################### Direct Version #################################

#These direct version are the same than the Zero Encoding version, you could
#skip sinc this is not useful.
def env_gRPC_di_half(n, p) :
  env_gRPC = env_gRPC_atomic(n, p, 0, 1)
  for i in range (2, n - 1, 2) :
    env_atom = env_gRPC_atomic(n, p, i, i + 1)
    env_gRPC = compo_env_gRPC(n, env_gRPC, env_atom)
  return env_gRPC

def env_cRPC_di_half (n, p) :
  if (n == 1) :
    return None
  
  env_gRPC = env_gRPC_di_half(n, p)
  return cRPC_from_gRPC(n, env_gRPC)

def env_UOcRPC_di_half (n, p) :
  if (n == 1) :
    return None
  
  env_gRPC = env_gRPC_di_half(n, p)
  return UOcRPC_from_gRPC(n, env_gRPC)


########################## IronMask Verif ######################################

def IRA_env_cRPC_ze_half (n, p) :
  """
  Compute the cardinal RPC envelope of the Half Refresh gadget using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Half Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/half/half_refresh_"+str(n)+"shares_cRPC.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_ze_half (n, p) :
  """
  Compute the UO cardinal RPC envelope of the Half Refresh gadget using IronMask.
  
  Warning: UO cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO cardinal RPC envelope of the Half Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/half/half_refresh_"+str(n)+"shares_cRPC.txt", n, p, -1)
  return pgref

def IRA_env_gRPC_ze_half (n, p) :
  """
  Compute the general RPC envelope of the Half Refresh gadget using IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Half Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_gen("env_IRA/half/half_refresh_"+str(n)+"shares_gRPC.txt", n, p, -1)
  return pgref


############################ Analytical Formula ################################

################################################################################
# Note that for the Half refresh gadget. The zero encoding version is          #
# equivalent to the direct version since there is only one random added by     #
# shares and consequently directly added to the shares by doing x_i = x_i + r. #
#                                                                              #
# Hence computing general RPC or general RPP is equivalent.                    #
################################################################################

def ana_env_gRPC_half (n, p) :
  """
  Compute the general RPC envelope of the Half Refresh gadget using analytical
  formula (i.e. Lemma 9 of our paper).

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Half Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  
  if (n % 2) : 
    raise Exception("Analytical formula for the half refresh gadget only " 
     + "available for n even")


  env_gen = {}
  lim = (1 << n)
  
  one_minus_q = 3*p - 3*p*p + p*p*p
  q = 1 - one_minus_q
  
  for I in range (lim) : 
    k0 = I.bit_count()
    for J in range (lim) :
      
      #Computing C_J from Lemma 9 (here the index starts with 0).
      paired_indices = 0
      for i in range (0, n, 2) :
        pair = (1 << i) + (1 << (i + 1))
        if ((J & pair) == pair) :
           paired_indices += pair
      
      #Test if C_J is include in I.
      if ((paired_indices & I) != paired_indices) :
         env_gen[(I, J)] = 0
         continue
      
      unpaired_indices = J & (~paired_indices)
      k1 = paired_indices.bit_count()
      k2 = (unpaired_indices & I).bit_count()
      k3 = unpaired_indices.bit_count()

      final_proba = 0
      for l in range (k2 + 1) :
        final_proba += (comb(k2, l) * (one_minus_q)**l * q**(k3 - l) * 
                        p**(k0 - k1 - l) * (1 - p)**(n - k0)) 
      env_gen[(I, J)] = final_proba
  return env_gen

def test_gRPC_half_refresh () : 
  """
  Verifying Analytical formula above thanks to IronMask.
  """
  n = 8
  p = 2**-5
  tol = 10 * np.finfo(float).eps

  env_gen1 = ana_env_gRPC_half(n, p)
  env_gen2 = IRA_env_gRPC_ze_half(n, p)

  for I in range (1 << n) :
    for J in range (1 << n) :
      a = env_gen1[(I, J)]
      b = env_gen2[(I, J)]

      rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)

      if rel_err > tol:
        print("I = ", I)
        print("J = ", J)
        print("rel_err = ", rel_err)
        print()
  
def ana_env_cRPC_half (n, p) :
  """
  Compute the cardinal RPC envelope of the Half Refresh gadget using analytical
  formula (i.e. Corollary 2 of our paper).

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: cardinal RPC envelope of the Half Refresh gadget.
  """
  if (n == 1) :
    return None
  
  n1 = n + 1
  nh = n // 2
  env = np.zeros((n1, n1))
  p_powers = precompute_p_powers(p, n)
  p1_powers = precompute_p_powers((1 - p), n)

  one_minus_q = 3*p - 3*p*p + p*p*p
  q = 1 - one_minus_q

  one_minus_q_powers = precompute_p_powers(one_minus_q, n)
  q_powers = precompute_p_powers(q, n)
  bino = precompute_binomial_coefficients(n)
  
  for tin in range (n1) :
    for tout in range (n1) :
      for k1 in range (max(0, tout - nh), (tout // 2) + 1) :
        k11 = 2 * k1
        if (tin < k11) :
          continue
        k3 = tout - k11
        pr = 0
        for l in range (min (tin - k11, k3) + 1) :
          c1 = bino[(k3, l)]
          elt1 = q_powers[k3 - l] * one_minus_q_powers[l]
          elt2 = bino[(n - k11 - l, tin - k11 - l)] * p_powers[tin - k11 - l] * p1_powers[n - tin] 
          pr += c1 * elt1 * elt2
        env[tin, tout] = max (env[tin, tout], pr)
  return env

def test_cRPC_half_refresh () :
  """
  Verifying Analytical formula above thanks to IronMask.
  """ 
  n = 8
  p = 2**-5
  tol = 10 * np.finfo(float).eps


  env1 = ana_env_cRPC_half(n, p)
  env2 = IRA_env_cRPC_ze_half (n, p)

  for tin in range (n + 1) :
    for tout in range (n + 1) :
      a = env1[tin, tout]
      b = env2[tin, tout]

      rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)

      if rel_err > tol:
        print("tin  = ", tin)
        print("tout = ", tout)
        print("rel_err = ", rel_err)
        print()
  

## -----------------------------------------------------------------------------
##                6. Gadget DoubleHalf Refresh (gen RPC & card RPC) 
## -----------------------------------------------------------------------------
##
##  This gadget is also called the Circular 1 Layer gadget, in its ZE version. 
##

########################### Zero Encoding Version ##############################

def gRPP_ze_doublehalf (n, p) :
  """
  General RPP envelope of the Zero Encoding of the Double Half Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[bytes, float]: General RPP envelope of the Zero Encoding of the
      Double Half Refresh gadget.
  """  
  env_init = init_env_gRPP(n)
  
  env_gRPP_random_added_by_index = [0] * n
  env_gRPP = update(p, 0, 1, env_init, env_gRPP_random_added_by_index)

  for i in range(2, n - 1, 2) :
    atom_random_added_by_index = [0] * n
    env_atom = update(p, i, i + 1, env_init, atom_random_added_by_index)
    env_gRPP, env_gRPP_random_added_by_index = treeAdd_ze(n, p, env_gRPP, env_gRPP_random_added_by_index, env_atom, atom_random_added_by_index)

  for i in range (1, n - 2, 2) :
    atom_random_added_by_index = [0] * n
    env_atom = update(p, i, i + 1, env_init, atom_random_added_by_index)
    env_gRPP, env_gRPP_random_added_by_index = treeAdd_ze(n, p, env_gRPP, env_gRPP_random_added_by_index, env_atom, atom_random_added_by_index)

  atom_random_added_by_index = [0] * n
  env_atom = update(p, 0, n - 1, env_init, atom_random_added_by_index)
  env_gRPP, env_gRPP_random_added_by_index = treeAdd_ze(n, p, env_gRPP, env_gRPP_random_added_by_index, env_atom, atom_random_added_by_index)

  return env_gRPP

def env_gRPC_ze_doublehalf (n, p) :
  """
  General RPC envelope of the Zero Encoding DoubleHalf Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero Encoding
      DoubleHalf Refresh gadget.
  """
  env_gRPP = gRPP_ze_doublehalf(n, p)
  return gRPC_from_gRPP(n, p, env_gRPP)

def env_cRPC_ze_doublehalf (n, p) :
  """
  Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh 
      gadget.
  """ 
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_doublehalf_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_doublehalf_n{n}_p{logp}.npy"

  if os.path.exists(filename_prec):
    return np.load(filename_prec)
  
  if (n == 1) :
    return None
  
  env_gRPP = gRPP_ze_doublehalf(n, p)
  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)

  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  return env_cRPC

def env_UOcRPC_ze_doublehalf (n, p) :
  """
  UO Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh 
      gadget.
  """ 
  logp = int(math.log(p, 2))
  filename_prec = f"./prec/refresh_prec/env_cRPC_ze_doublehalf_n{n}_p{logp}.npy"
  filename_UO_prec = f"./prec/refresh_prec/env_UOcRPC_ze_doublehalf_n{n}_p{logp}.npy"

  if os.path.exists(filename_UO_prec):
    return np.load(filename_UO_prec)
  
  if (n == 1) :
    return None
  
  env_gRPP = gRPP_ze_doublehalf(n, p)
  env_cRPC = cRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_prec, env_cRPC)

  env_UOcRPC = UOcRPC_from_gRPP(n, p, env_gRPP)
  np.save(filename_UO_prec, env_UOcRPC)

  return env_UOcRPC

############################### Direct Version #################################

def env_gRPC_di_doublehalf(n, p) :
  """
  General RPC envelope of the Direct DoubleHalf Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Direct
      DoubleHalf Refresh gadget.
  """
  env_gRPC = env_gRPC_atomic(n, p, 0, 1)

  for i in range (2, n - 1, 2) :
    env_atom = env_gRPC_atomic(n, p, i, i + 1)
    env_gRPC = compo_env_gRPC(n, env_gRPC, env_atom)

  for i in range (1, n - 1, 2) :
    env_atom = env_gRPC_atomic(n, p, i, i + 1)
    env_gRPC = compo_env_gRPC(n, env_gRPC, env_atom)

  env_atom = env_gRPC_atomic(n, p, 0, n - 1)
  env_gRPC = compo_env_gRPC(n, env_gRPC, env_atom)
  return env_gRPC

def env_cRPC_di_doublehalf (n, p) :
  """
  Cardinal RPC envelope of the Direct DoubleHalf Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Direct DoubleHalf Refresh 
      gadget.
  """ 
  if (n == 1) :
    return None
  
  env_gRPC = env_gRPC_di_doublehalf(n, p)
  return cRPC_from_gRPC(n, env_gRPC)

def env_UOcRPC_di_doublehalf (n, p) :
  """
  UO Cardinal RPC envelope of the Direct DoubleHalf Refresh gadget.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO Cardinal RPC envelope of the Direct DoubleHalf Refresh 
      gadget.
  """ 
  if (n == 1) :
    return None
  
  env_gRPC = env_gRPC_di_doublehalf(n, p)
  return UOcRPC_from_gRPC(n, env_gRPC)


########################## IronMask Verif ######################################

def IRA_env_cRPC_ze_doublehalf (n, p) :
  """
  Compute the cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh
  gadget using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh 
      gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/doublehalf/ze_doublehalf_"+str(n)+"shares_cRPC.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_ze_doublehalf (n, p) :
  """
  Compute the UO cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh
  gadget using IronMask.
  
  Warning: UO cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO cardinal RPC envelope of the Zero Encoding DoubleHalf 
      Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/doublehalf/ze_doublehalf_"+str(n)+"shares_cRPC.txt", n, p, -1)
  return pgref

def IRA_env_gRPC_ze_doublehalf (n, p) :
  """
  Compute the general RPC envelope of the Zero Encoding DoubleHalf Refresh
  gadget using IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero Encoding
      DoubleHalf Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_gen("env_IRA/doublehalf/ze_doublehalf_"+str(n)+"shares_gRPC.txt", n, p, -1)
  return pgref

def IRA_env_cRPC_di_doublehalf (n, p) :
  """
  Compute the cardinal RPC envelope of the Direct DoubleHalf Refresh gadget
  using IronMask.
  
  Warning: Cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Direct DoubleHalf Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes("env_IRA/doublehalf/di_doublehalf_"+str(n)+"shares_cRPC.txt", n, p, -1)
  return pgref

def IRA_env_UOcRPC_di_doublehalf (n, p) :
  """
  Compute the UO cardinal RPC envelope of the Direct DoubleHalf Refresh gadget
  using IronMask.
  
  Warning: UO cardinal RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: UO cardinal RPC envelope of the Direct DoubleHalf Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_UOcPRC("env_IRA/doublehalf/di_doublehalf_"+str(n)+"shares_cRPC.txt", n, p, -1)
  return pgref

def IRA_env_gRPC_di_doublehalf (n, p) :
  """
  Compute the general RPC envelope of the Direct DoubleHalf Refresh gadget using
  IronMask.

  Warning: General RPC envelopes are only available for up to 8 shares.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Direct
      DoubleHalf Refresh gadget.
  """
  if (n == 1) :
    return None
  
  if (n > 8) : 
    raise Exception("Number of shares must be at most 8!")
  pgref = parse_enveloppes_gen("env_IRA/doublehalf/di_doublehalf_"+str(n)+"shares_gRPC.txt", n, p, -1)
  return pgref



######################### Analytical ZE Version ################################


def all_cyclic_interval_rgs(n):  
  """
  Generate all canonical RGS partitions into contiguous intervals on a cycle.

  Args:
      n (int): Number of elements.

  Yields:
      bytes: Canonical RGS representation of each cyclic interval partition.
  """  
  seen = set()

  # cuts[i] = 1 if there is a cut between i and  i+1 mod n
  for cuts in product((0, 1), repeat=n):
    if sum(cuts) == 0:
      l_part = [0] * n
    else:
      l_part = [0] * n
      cur = 0

      for i in range(1, n):
        if cuts[i - 1]:
          cur += 1
        l_part[i] = cur

      # If no cuts between n-1 and 0, last bloc is merged with the first.
      if cuts[n - 1] == 0:
        last_label = l_part[-1]
        for i in range(n):
          if l_part[i] == last_label:
            l_part[i] = 0

      l_part = list(__canonize_partitions__(l_part))

    part = bytes(l_part)
    if part not in seen:
      seen.add(part)
      yield part

def ana_env_gRPP_ze_doublehalf (n, p) :
  """
  Compute the general RPP envelope of the Zero Encoding DoubleHalf Refresh
  gadget using analytical formula (i.e. Lemma 10 of our paper).

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[bytes, float]: General RPP envelope of the Zero Encoding DoubleHalf 
      Refresh gadget.
  """
  env_gRPP = {}

  one_minus_q = 3*p - 3*p*p + p*p*p
  q = 1 - one_minus_q


  for part_b in all_cyclic_interval_rgs(n):
    if (part_b == bytes([0] * n)) :
      pb = q**n + n * q**(n - 1) * one_minus_q      
      env_gRPP[part_b] = pb
      continue
    
    k = max(part_b) + 1
    pb = one_minus_q**k * q**(n - k)
    env_gRPP[part_b] = pb
  
  return env_gRPP


def ana_env_gRPC_ze_doublehalf (n, p) : 
  """
  Compute the general RPC envelope of the Zero Encoding DoubleHalf Refresh
  gadget using analytical formula.

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      dict[tuple[int, int], float]: General RPC envelope of the Zero Encoding 
      DoubleHalf Refresh gadget.
  """ 
  if (n == 1) :
    return None
  
  env_gRPP = ana_env_gRPP_ze_doublehalf(n, p)

  #All index have been refreshed two times, so they have to be output
  #manipulated before getting added to general RPC envelope.
  random_added_by_index = [2] * n
  env_gRPP = gRPP_output_manipulation(n, p, env_gRPP, random_added_by_index)

  return gRPC_from_gRPP(n, p, env_gRPP)

def test_gRPC_doublehalf_refresh () :
  """
  Verify Analytical formula above thanks to IronMask.
  """ 
  n = 8
  p = 2**-10
  tol = 100 * np.finfo(float).eps

  env_gen1 = ana_env_gRPC_ze_doublehalf(n, p)
  env_gen2 = parse_enveloppes_gen("env_IRA/doublehalf/ze_doublehalf_"+str(n)+"shares_gRPC.txt", n, p, -1)

  for I in range (1 << n) :
    for J in range (1 << n) :
      a = env_gen1[(I, J)]
      b = env_gen2[(I, J)]

      rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)

      if rel_err > tol:
        print("I = ", I)
        print("J = ", J)
        print("rel_err = ", rel_err)
        print()

def ana_env_cRPC_ze_doublehalf (n, p) :
  """
  Compute the cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh
  gadget using analytical formula.

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh 
      gadget.
  """   
  if (n == 1) :
    return None
  
  env_gRPP = ana_env_gRPP_ze_doublehalf(n, p)

  #All index have been refreshed two times, so they have to be output
  #manipulated before getting added to general RPC envelope.
  random_added_by_index = [2] * n
  env_gRPP = gRPP_output_manipulation(n, p, env_gRPP, random_added_by_index)

  return cRPC_from_gRPP(n, p, env_gRPP)

def test_cRPC_doublehalf_refresh () : 
  """
  Verify Analytical formula above thanks to IronMask.
  """
  n = 8
  p = 2**-10
  tol = 100 * np.finfo(float).eps

  env1 = ana_env_cRPC_ze_doublehalf(n, p)
  env2 = IRA_env_cRPC_ze_doublehalf (n, p)

  for tin in range (n + 1) :
    for tout in range (n + 1) :
      a = env1[tin, tout]
      b = env2[tin, tout]

      rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)

      if rel_err > tol:
        print("tin = ", tin)
        print("tout = ", tout)
        print("a = ", a)
        print("b = ", b)
        print("rel_err = ", rel_err)
        print()


def ana_env_cRPP_ze_doublehalf (n, p) :
  """
  Compute the cardinal RPP envelope of the Zero Encoding DoubleHalf Refresh
  gadget using analytical formula.

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      list[float]: Cardinal RPP envelope of the Zero Encoding DoubleHalf Refresh 
      gadget.
  """  
  int_partitions_list = generate_cardinal_set(n)
  env_cRPP = [0] * len(int_partitions_list)

  factorial_cache = [math.factorial(k) for k in range (n + 1)]
  
  one_minus_q = 3*p - 3*p*p + p*p*p
  q = 1 - one_minus_q

  for ind_cs, cs in enumerate(int_partitions_list):
    k = len(cs)
    if (k == 1) :
      env_cRPP[ind_cs] = q**n + n * q**(n - 1) * one_minus_q
      continue

    # Comptage des multiplicités
    counts = Counter(cs)  # ex: {1:2, 2:1}
    
    # Produit des m_s!
    multiplicity_prod = 1
    for m in counts.values():
      multiplicity_prod *= factorial_cache[m]

    env_cRPP[ind_cs] = ((n / k) * (factorial_cache[k] / multiplicity_prod)) * one_minus_q**k * q**(n - k)
  return env_cRPP

def ana_env_cRPC_ze_doublehalf_v2 (n, p) :
  """
  Compute the cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh
  gadget using analytical formula. The Cardinal RPC envelope are derived from
  the cardinal RPP formula, and not the general RPP formula as previously done
  in |ana_env_cRPC_ze_doublehalf|.

  Warning: Analytical are only available for an even number of shares n.

  Args: 
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns: 
      np.ndarray: Cardinal RPC envelope of the Zero Encoding DoubleHalf Refresh 
      gadget.
  """   
  int_partitions_list = generate_cardinal_set(n)
  binom_coeffs = precompute_binomial_coefficients(n)
  env_cRPP = ana_env_cRPP_ze_doublehalf(n, p)
  env_cRPP = cRPP_output_manipulation(int_partitions_list, env_cRPP, 
                                      binom_coeffs, p, n)
  
  return cRPC_from_cRPP(n, p, env_cRPP)



## -----------------------------------------------------------------------------
##                Gadget RP Add from refresh
## -----------------------------------------------------------------------------

def cRPC_pgadd(n, p, pgref):
  """
  Compute the cardinal RPC envelope of the addition gadget according to the
  cardinal RPC envelope of the refresh gadget.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      np.ndarray: Cardinal RPC envelope of the addition gadget.
  """
  pgadd = np.zeros((n+1, n+1, n+1))
  for i_1 in range(n+1):
    for i_2 in range(n+1):
      for j in range(n+1):
        pgadd[i_1, i_2, j] = 0
        for l1 in range(0,n-j+1):
          for l2 in range(0,n-j+1):
            pgadd[i_1, i_2, j] += (comb(n-j, l1) * comb(n-j, l2) * (p**(l1+l2)) * 
            (1-p)**(2*n-2*j-l1-l2) * pgref[i_1, j+l1] * pgref[i_2, j+l2])
  return pgadd


def cRPC_pgcopy_prec(n,pgref):
  """
  Precomputation to compute the cardinal RPC envelope of the copy gadget.

  Args:
      n (int): Number of shares.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      np.ndarray: Precomputation for the cardinal RPC envelope of the copy 
      gadget.
  """
  binom_coeffs = precompute_binomial_coefficients(n)
  pgcopy = np.zeros((n+1, n+1, n+1))
  for tin in range(0,n+1):
    for i in range(0,tin+1):
      for j in range(tin-i,tin+1):
        ptemp = binom_coeffs[(i, i+j-tin)]*binom_coeffs[(n-i, tin-i)]/binom_coeffs[(n, j)]
        for tout_1 in range(0,n+1):
          for tout_2 in range(0,n+1):
            pgcopy[tin,tout_1,tout_2] += pgref[i,tout_1]*pgref[j,tout_2]*ptemp
  return pgcopy

def cRPC_pgcopy(n, p, pgref):
  """
  Compute the cardinal RPC envelope of the copy gadget according to the
  cardinal RPC envelope of the refresh gadget.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      np.ndarray: Cardinal RPC envelope of the copy gadget.
  """
  prec = cRPC_pgcopy_prec(n,pgref)
  pgcopy = np.zeros((n + 1, n + 1, n + 1))
  for tin in range (n + 1) :
    for tout1 in range (n + 1) :
      for tout2 in range (n + 1) :
        for l in range (tin + 1) :
          pgcopy[tin, tout1, tout2] += (comb(n - l, tin - l) * p**(tin - l) * 
                                    (1 - p)**(n - tin) * prec[l, tout1, tout2]) 
        pgcopy[tin, tout1, tout2] = min(1, pgcopy[tin, tout1, tout2])
  return pgcopy


def cRPC_pgcmult(n, p, pgref):
  """
  Compute the cardinal RPC envelope of the multiplication by a constant gadget 
  according to the cardinal RPC envelope of the refresh gadget.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      np.ndarray: Cardinal RPC envelope of the multiplication by a constant 
      gadget.
  """
  binom_coeffs = precompute_binomial_coefficients(n)
  pgcmult = np.zeros((n + 1, n + 1))
  for tin in range(0, n + 1):
    for tout in range(0, n + 1):
      for i in range(0,n - tout + 1):
        pgcmult[tin,tout] += binom_coeffs[(n-tout, i)]*(p**i)*((1-p)**(n-tout-i))*pgref[tin,tout+i]
  return pgcmult


## -----------------------------------------------------------------------------
##                Printing Refresh envelope
## -----------------------------------------------------------------------------

def print_env(n, env) :
  """
  Print cardinal RPC envelope of a refresh gadget.

  Args:
      n (int): Number of shares.
      env (np.ndarray): Cardinal RPC envelope of a refresh gadget.
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


def print_env_gen(n, env_gen) :
  """
  Print general RPC envelope of a refresh gadget.

  Args:
      n (int): Number of shares.
      env (dict[tuple[int, int], float]): General RPC envelope of a refresh gadget.
  """
  for J in range (1 << n) :
    J_line = "{"
    for i in range (n + 1) :
      if((1 << i) & J) :
        J_line += " " + str(i) + ","
    J_line += "}"
      
    for I in range (1 << n) :
      I_line = "{"
      for i in range (n + 1) :
        if((1 << i) & I) :
          I_line += " " + str(i) + ","
      I_line += "}"


      res = env_gen[(I, J)]
         
      first_line = "J = " + J_line + ", I = " + I_line + f", pr = {res:.3g}"
      print(first_line)
      print()

################################################################################
#                                                                              #
#                       Testing Precision via IRA                              #
#                                                                              #
################################################################################

def test_precision_cRPC (n, p) :
  """
  Test the numerical precision of different procedures computing the same 
  cardinal RPC envelope.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
  """

  print("Testing cardinal RPC compositions properties...")
  def compare_methods (method1, method2, name_ref) :
    print("Testing " + name_ref + "...")
    env1 = method1(n, p)
    env2 = method2(n, p)

    tol = 100 * np.finfo(float).eps
    max_rel_err = 0
    is_fail = False

    for tin in range (n + 1) :
      for tout in range (n + 1) :
        a = env1[tin, tout]
        b = env2[tin, tout]

        rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)
        max_rel_err = max(max_rel_err, rel_err)

        if rel_err > tol:
          is_fail = True


    if (is_fail) :
      print("Maximale Relative Error : ", max_rel_err)
      print("*** Failed ***")
      print()
          
    else : 
      print("Maximale Relative Error : ", max_rel_err)
      print("*** Passed ***")
      print()    

  """
  name_ref = "Atomic Composition for Direct Encoding Linear Refresh"
  method1 = env_cRPC_di_linear
  method2 = IRA_env_cRPC_di_linear
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Direct Encoding nlogn Refresh"
  method1 = env_cRPC_di_nlogn
  method2 = IRA_env_cRPC_di_nlogn
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Direct Encoding Double Half Refresh"
  method1 = env_cRPC_di_doublehalf
  method2 = IRA_env_cRPC_di_doublehalf
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding Linear Refresh"
  method1 = env_cRPC_ze_linear
  method2 = IRA_env_cRPC_ze_linear
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding nlogn Refresh"
  method1 = env_cRPC_ze_nlogn
  method2 = IRA_env_cRPC_ze_nlogn
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding Half Refresh"
  method1 = env_cRPC_ze_half
  method2 = IRA_env_cRPC_ze_half
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding Double Half Refresh"
  method1 = env_cRPC_ze_doublehalf
  method2 = IRA_env_cRPC_ze_doublehalf
  compare_methods(method1, method2, name_ref)
  """

  name_ref = "Analytical Formula for Zero Encoding Half Refresh"
  method1 = ana_env_cRPC_half
  method2 = IRA_env_cRPC_ze_half
  compare_methods(method1, method2, name_ref)

  name_ref = "Analytical Formula for Zero Encoding Double Half Refresh"
  method1 = ana_env_cRPC_ze_doublehalf
  method2 = IRA_env_cRPC_ze_doublehalf
  compare_methods(method1, method2, name_ref)


def test_precision_gRPC (n, p) :
  """
  Test the numerical precision of different procedures computing the same 
  general RPC envelope.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
  """
  print("Testing general RPC compositions properties...")

  def compare_methods (method1, method2, name_ref) :
    print("Testing " + name_ref + "...")
    env1 = method1(n, p)
    env2 = method2(n, p)

    tol = 100 * np.finfo(float).eps
    max_rel_err = 0
    is_fail = False

    for I in range (1 << n) :
      for J in range (1 << n) :
        a = env1[(I, J)]
        b = env2[(I, J)]

        rel_err = abs(a - b) / max(abs(a), abs(b), 1e-300)
        max_rel_err = max(max_rel_err, rel_err)

        if rel_err > tol:
          is_fail = True


    if (is_fail) :
      print("Maximale Relative Error : ", max_rel_err)
      print("*** Failed ***")
      print()
          
    else : 
      print("Maximale Relative Error : ", max_rel_err)
      print("*** Passed ***")
      print()    

  """
  name_ref = "Atomic Composition for Direct Encoding Linear Refresh"
  method1 = env_gRPC_di_linear
  method2 = IRA_env_gRPC_di_linear
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Direct Encoding nlogn Refresh"
  method1 = env_gRPC_di_nlogn
  method2 = IRA_env_gRPC_di_nlogn
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Direct Encoding Double Half Refresh"
  method1 = env_gRPC_di_doublehalf
  method2 = IRA_env_gRPC_di_doublehalf
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding Linear Refresh"
  method1 = env_gRPC_ze_linear
  method2 = IRA_env_gRPC_ze_linear
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding nlogn Refresh"
  method1 = env_gRPC_ze_nlogn
  method2 = IRA_env_gRPC_ze_nlogn
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding Half Refresh"
  method1 = env_gRPC_ze_half
  method2 = IRA_env_gRPC_ze_half
  compare_methods(method1, method2, name_ref)

  name_ref = "Atomic Composition for Zero Encoding Double Half Refresh"
  method1 = env_gRPC_ze_doublehalf
  method2 = IRA_env_gRPC_ze_doublehalf
  compare_methods(method1, method2, name_ref)
  """

  name_ref = "Analytical Formula for Zero Encoding Half Refresh"
  method1 = ana_env_gRPC_half
  method2 = IRA_env_gRPC_ze_half
  compare_methods(method1, method2, name_ref)

  name_ref = "Analytical Formula for Zero Encoding Double Half Refresh"
  method1 = ana_env_gRPC_ze_doublehalf
  method2 = IRA_env_gRPC_ze_doublehalf
  compare_methods(method1, method2, name_ref)




if __name__ == "__main__" :
  print("main")

  #Responsible of the last sentence of section 5.3.

  #test_precision_cRPC(n, p)
  #print()
  #test_precision_gRPC(n, p)

  n = 6
  p = 2**-20
  env = env_cRPC_di_ideal (n, p)

  print_env(n, env)






















