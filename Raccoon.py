#!/usr/bin/env python3

################################################################################
# This file is primarily based on the file available at: 
#
#   https://github.com/CryptoExperts/EC25-random-probing-Raccoon
#
# with a few minor modifications to suit our purposes.
# In the following, we will note [BelRivRos25] the paper we found here : 
#
#   https://eprint.iacr.org/2025/278
#
# It enables the computation of the threshold random probing composability of
# our masked implementation of Raccoon, built using the compiler described in
# Appendix E.1 of our paper. In addition, it enables to optimize automatically
# the gamma taken (i.e. the number of iteration in RPRefresh) for each block in
# Raccon-Sign and Raccoon-Keygen.
################################################################################

import itertools
from collections import defaultdict
import math
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import os

from refresh_env import (env_cRPC_ze_RPRefresh, cRPC_pgadd, cRPC_pgcopy, cRPC_pgcmult, 
                         precompute_binomial_coefficients)


## -------------------------------------------------------------
##              Gadget Gsum with chain structure
## -------------------------------------------------------------
def cardinal_rpc_gaddpchain_pgadd(n, ell, tin, tout, pgadd):
  r"""
  Compute the case (*tin, tout) of the $\ell + 1$-dimensionnal cardinal RPC 
  envelope of Gsum-chain.

  Args:
      n (int): Number of shares.
      ell (int): Number of $n$-sharing.
      tin (tuple[int, ...]): Tuple of input shares leaked by each $n$-sharings.
      tout (int): Number of output shares leaked.
      pgadd (np.ndarray): Cardinal RPC envelope of addition gadget.

  Returns : 
      float : Probability of the case (*tin, tout) of the cardinal RPC envelope
      of Gsum-tree.
  """
  def recursive_sum(index, i_prev):
    if index == ell-1:  # Last sum (index ell-1)
      return pgadd[i_prev, tin[index], tout]
    else:
      total_sum = 0.0
      for i_curr in range(n+1):
        prob = pgadd[i_prev, tin[index], i_curr] * recursive_sum(index + 1, i_curr)
        total_sum += prob
      return total_sum

  total_probability = 0.0
  for i_3 in range(n+1):
    total_probability += pgadd[tin[0], tin[1], i_3] * recursive_sum(2, i_3)
  return total_probability

def cardinal_rpc_gaddpchain_envelope_pgadd(n, ell, pgadd):
  r"""
  Compute the cardinal RPC envelope of the gadget Gsum-chain (algorithm 10 of
  [BelRivRos25]) using Lemma 12 of [BelRivRos25]. 

  Args:
      n (int): Number of shares.
      ell (int): Number of $n$-sharings.
      pgadd (np.ndarray): Cardinal RPC envelope of an addition gadget, i.e. a
      three-dimensionnal array of shape (n, n, n).

  Returns:
      np.ndarray: Cardinal RPC envelope of the gadget Gsum-chain, i.e. a
      $\ell + 1$-dimensionnal array of shape $(n, \ldots, n)$.
  """

  pgaddpchain = np.zeros([n+1] * (ell+1))
  all_tin = list(itertools.product(range(n+1), repeat=ell))
  for tin in all_tin:
    for j in range(n+1):
      pgaddpchain[(*tin, j)] = cardinal_rpc_gaddpchain_pgadd(n,ell,tin,j,pgadd)
  return pgaddpchain

## -------------------------------------------------------------
##              Gadget Gsum with binary tree structure
## -------------------------------------------------------------
def cardinal_rpc_gaddptree_pgadd(n,ell,tin,tout,pgadd, memo=None):
  r"""
  Compute the case (*tin, tout) of the $\ell + 1$-dimensionnal cardinal RPC 
  envelope of Gsum-tree.

  Args:
      n (int): Number of shares.
      ell (int): Number of $n$-sharing.
      tin (tuple[int, ...]): Tuple of input shares leaked by each $n$-sharings.
      tout (int): Number of output shares leaked.
      pgadd (np.ndarray): Cardinal RPC envelope of addition gadget.
      memo (dict, optionnal) : Dictionnary of the results, if we have already
      computed it in the past.
  
  Returns : 
      float : Probability of the case (*tin, tout) of the cardinal RPC envelope
      of Gsum-tree.
  """
  if memo is None:
    memo = {}

  key = (ell, tuple(tin), tout)
  if key in memo:
    return memo[key]

  if ell==2:
    result = pgadd[tin[0], tin[1], tout]
    memo[key] = result 
    return result

  if ell==3:
    p = 0
    for i in range(n + 1):
      p += pgadd[tin[0], tin[1], i] * pgadd[i, tin[2], tout]
    memo[key] = p 
    return p

  p = 0
  half_ell = ell // 2
  for i1 in range(n+1):
    for i2 in range(n+1):
      part1 = cardinal_rpc_gaddptree_pgadd(n, half_ell, tin[:half_ell], i1, pgadd, memo)
      part2 = cardinal_rpc_gaddptree_pgadd(n, ell - half_ell, tin[half_ell:], i2, pgadd, memo)
      p += part1 * part2 * pgadd[i1, i2, tout]

  memo[key] = p
  return p

def cardinal_rpc_gaddptree_envelope_pgadd(n,ell,pgadd):
  r"""
  Compute the cardinal RPC envelope of the gadget Gsum-tree (algorithm 11 of
  [BelRivRos25]) using Lemma 13 of [BelRivRos25]. 

  Args:
      n (int): Number of shares.
      ell (int): Number of $n$-sharings.
      pgadd (np.ndarray): Cardinal RPC envelope of an addition gadget, i.e. a
      three-dimensionnal array of shape (n, n, n).

  Returns:
      np.ndarray: Cardinal RPC envelope of the gadget Gsum-tree, i.e. a
      $\ell + 1$-dimensionnal array of shape $(n, \ldots, n)$.
  """
  pgaddptree = np.zeros([n+1] * ell+ [n+1])
  all_tin = list(itertools.product(range(n+1), repeat=ell))
  for tin in all_tin:
    for j in range(n+1):
      pgaddptree[(*tin, j)] = cardinal_rpc_gaddptree_pgadd(n,ell,tin,j,pgadd)

  return pgaddptree

def rpc_gsumtree(n,t,ell,pgadd):
  r"""
  Compute the advantage $\varepsilon$ of the t-threshold RPC of the gadget
  Gsum-tree.

  Args:
      n (int): Number of shares.
      t (int): Threshold of the threshold RPC property.
      ell (int): Number of $n$-sharing.
      pgadd (np.ndarray): Cardinal RPC envelope of addition gadget.

  Returns:
      float: Advantage $\varepsilon$ of the t-threshold RPC of the gadget 
      Gsum-tree.
  """
  rpc_gaddp=0
  for tin in itertools.product(range(n+1), repeat=ell):
    if any(x > t for x in tin):
      rpc_gaddp+= cardinal_rpc_gaddptree_pgadd(n,ell,tin,t,pgadd)
  return rpc_gaddp

## -------------------------------------------------------------
##              Gadget AddNoiseTo
## -------------------------------------------------------------

def cardinal_rpc_addnoiseto_envelope_pgaddp_pgadd(n,ell,pgaddp,pgadd):  
  r"""
  Compute the cardinal RPC envelope of the gadget AddNoiseTo 
  (Algorithm 8 of [BelRivRos25]), following Lemma 11 of [BelRivRos25]. Note that it is not
  exactly the same than in Lemma 11 because we does not sum the $t_{in, *}^a$.

  Args:
      n (int): _description_
      ell (int): _description_
      pgaddp (np.ndarray): Cardinal RPC envelope of the gadget Gsum, either
      Gsum-chain or Gsum-tree. In the following, it will be Gsum-tree.
      pgadd (np.ndarray): Cardinal RPC envelope of the addition gadget.

  Returns:
      np.ndarray: Cardinal RPC envelope of the gadget AddNoiseTo, i.e. a
      $\ell + 2$-dimensionnal array of shape $(n, \ldots, n)$.
  """
  paddnoiseto = np.zeros([n+1]+[n+1]*ell+[n+1])
  all_tin_a = list(itertools.product(range(n+1), repeat=ell))
  for tin_a in all_tin_a:
    for tin_x in range(0,n+1):
      for tout in range(0,n+1):
        for i in range(n+1):
          paddnoiseto[(tin_x,*tin_a,tout)] += pgaddp[(*tin_a, i)] * pgadd[tin_x,i,tout]
  return paddnoiseto

## -------------------------------------------------------------
##              Gadget Gcmult
## -------------------------------------------------------------

def rpc_gcmult_pgref(n,t,p,pgref):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the gadget Gcmult.

  Args:
      n (int): Number of shares.
      t (int): Threshold of the threshold RPC property..
      p (float): Leakage rate.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      float : t-threshold RPC advantage $\varepsilon$ of the gadget Gcmult.
  """
  binom_coeffs = precompute_binomial_coefficients(n)
  p_rpc_cmult = 0
  for tin in range(t+1,n+1):
    for i in range(0,n-t+1):
      p_rpc_cmult += binom_coeffs[(n-t, i)]*(p**i)*((1-p)**(n-t-i))*pgref[tin,t+i]
  return p_rpc_cmult

## -------------------------------------------------------------
##              Gadget Gmmult
## -------------------------------------------------------------
def cardinal_rpc_gcopytree_pgcopy(n,ell,tin,tout,pgcopy, memo=None):
  r"""
  Compute partially the cardinal RPC envelope of Gcopy-tree, where *tout* is the
  same for all output sharings.

  Args:
      n (int): Number of shares.
      ell (int): Number of $n$-sharing inputs.
      tin (int): Number of input shares leaked.
      tout (int): Number of output shares leaked.
      pgcopy (np.ndarray): Cardinal RPC envelope of copy gadget.
      memo (dict, optional): Dictionnary of the results, if we have already
      computed it in the past. Defaults to None.

  Returns:
      float: Probability of the case $[tin, tout, \ldots, tout]$ of the cardinal
      RPC envelope of Gcopy-tree. 
  """
  if memo is None:
    memo = {}
  key = (ell, tin)
  if key in memo:
    return memo[key]

  if ell==2:
    result = pgcopy[tin, tout, tout]
    memo[key] = result
    return result
  if ell==3: 
    p = 0
    for i in range(0,n+1):
      p += pgcopy[tin,i,tout]*pgcopy[i,tout,tout]
    result = p
    memo[key] = result
    return result
  p = 0
  half_ell = ell // 2
  for o1 in range(0,n+1):
    for o2 in range(0,n+1):
      part1 = cardinal_rpc_gcopytree_pgcopy(n, half_ell, o1, tout, pgcopy, memo)
      part2 = cardinal_rpc_gcopytree_pgcopy(n, ell - half_ell, o2, tout, pgcopy, memo)
      p += part1 * part2 * pgcopy[tin, o1, o2]

  memo[key] = p
  return p

def rpc_gcopytree(n,t,ell,pgcopy):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the gadget Gcopy-tree.

  Args:
      n (int): Number of shares.
      t (int): Threshold number.
      ell (int): Number of $\ell$ $n$-output sharings.
      pgcopy (np.ndarray): Cardinal RPC envelope of copy gadget.

  Returns:
      float: Advantage $\varepsilon$ of the t-threshold RPC property for the
      gadget Gcopy-tree.
  """
  rpc_gcopytree=0
  for tin in range(t+1,n+1):
    rpc_gcopytree+= cardinal_rpc_gcopytree_pgcopy(n,ell,tin,t,pgcopy)
  return rpc_gcopytree

def rpc_gpolymult(n,t,p,raccoon_n,pgcopy,pgadd,pgref):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the gadget Gcpmult,
  see section 6.1 of [BelRivRos25] for more details.


  Args:
      n (int): Number of shares.
      t (int): Threshold Number.
      p (float): Leakage Rate.
      raccoon_n (int): Dimension of the lattice in Raccoon.
      pgcopy (np.ndarray): Cardinal RPC envelope of the copy gadget.
      pgadd (np.ndarray): Cardinal RPC envelope of the addition gadget.
      pgref (np.ndarray): Cardinal RPC envelope of the refresh gadget.

  Returns:
      float: t-threshold RPC advantage $\varepsilon$ of the gadget Gcpmult.
  """
    
  # First copy step
  p_rpc_gcopytree = rpc_gcopytree(n,t,raccoon_n,pgcopy)*raccoon_n
  # Linear multiplication step
  p_rpc_gpolymult = rpc_gcmult_pgref(n,t,p,pgref)*raccoon_n*raccoon_n
  # Addition step
  p_rpc_gsumtree_4 = rpc_gsumtree(n,t,4,pgadd)
  p_rpc_gsumtree = p_rpc_gsumtree_4*(128+32+8+4+1)*raccoon_n
  return p_rpc_gcopytree+p_rpc_gpolymult+p_rpc_gsumtree

## -------------------------------------------------------------
##              Gadget Decode
## -------------------------------------------------------------

def compute_pmf_Dn(n, p, memo):
  r"""
  Function to recursively compute the PMF of D^{(n)}, used in Theorem 2 of
  [BelRivRos25].

  Args:
      n (int): Number of shares
      p (float): Leakage rate.
      memo (dict[int, dict[tuple[int, int], float]]): Dictionary storing
      previously computed PMFs, indexed by the number of shares.

  Returns:
      dict[tuple[int, int], float]: PMF of D^{(n)}, mapping each pair (r, s) to
      its probability. 
  """
  # Base case for n=1
  if n == 1:
    return {(1, 1): p, (0, 0): 1 - p}
    
  # Check if already computed
  if n in memo:
    return memo[n]

  # Recursive case: compute PMF for smaller n
  pmf_D_half_floor = compute_pmf_Dn(math.floor(n / 2), p, memo)
  pmf_D_half_ceil = compute_pmf_Dn(math.ceil(n / 2), p, memo)

  # Dictionary to hold PMF for current n
  pmf_Dn = defaultdict(float)

  # Combine PMFs of D^{(floor(n/2))} and D^{(ceil(n/2))}
  for (r1, s1), prob1 in pmf_D_half_floor.items():
    for (r2, s2), prob2 in pmf_D_half_ceil.items():
      prob_combined = prob1 * prob2

      s = min(r1 + r2, s1 + math.ceil(n / 2), s2 + math.floor(n / 2))

      # First case: probability p
      r = n
      pmf_Dn[(r, s)] += p * prob_combined

      # Second case: probability (1 - p)
      r = r1 + r2
      pmf_Dn[(r, s)] += (1 - p) * prob_combined

  # Memorize the result
  memo[n] = dict(pmf_Dn)
  return memo[n]

def compute_pmf_min_rs(n,p):
  r"""
  Function to compute the PMF of min(r, s) based on the PMF of D^{(n)}.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.

  Returns:
      dict[int, float]: PMF of min(r, s), mapping each possible value to its
      probability.
  """
  memo = {}
  pmf_Dn = compute_pmf_Dn(n, p, memo)

  # Dictionary to hold the PMF of min(r, s)
  pmf_min_rs = defaultdict(float)

  # Sum up probabilities for min(r, s)
  for (r, s), prob in pmf_Dn.items():
    pmf_min_rs[min(r, s)] += prob

  return dict(pmf_min_rs)

def decode_crpcwo_envelope(n,p,pgref):
  r"""
  Compute the cardinal RPC-O envelope of the gadget Decode, se section 4.3 of
  [BelRivRos25]. 

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      pgref (_type_): Cardinal RPC envelope of the refresh gadget.

  Returns:
      list[np.longdouble]: Cardinal RPC-O envelope of the gadget Decode.
  """
  pmf_Tn = compute_pmf_min_rs(n,p)
    
  dec_enc = [np.longdouble(0.0)]*(n+1)
    
  for t_in in range(n+1):    
    for t_out in range(n+1):    
      dec_enc[t_in] += pgref[t_in][t_out]*pmf_Tn[t_out]
            
  return dec_enc

## -------------------------------------------------------------
##              Raccoon-128 - KeyGen
## -------------------------------------------------------------

def rpc_ai_keygen_block1(n,t,tri,pgaddp,pgcopy):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the block 1 of the
  KeyGen function of the Raccoon scheme, see Figure 6 of [BelRivRos25] to identify the
  block 1.

  Args:
      n (int): Number of shares.
      t (int): Threshold number.
      tri (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      pgaddp (np.ndarray): Cardinal RPC envelope of Gsum-tree.
      pgcopy (np.ndarray): Cardinal RPC envelope of Gcopy.

  Returns:
      float:  t-threshold RPC advantage $\varepsilon$ of the block 1 of the
      KeyGen function.
  """
  p_rpc_ai_keygen_block1 = 0
  filtered_tin = [(i, j) for i, j in itertools.product(range(n + 1), repeat=2) if i + j > tri]
  for tm in range(n+1):
    for tin in filtered_tin:
      p_rpc_ai_keygen_block1 += pgaddp[tin[0],tin[1],tm] * pgcopy[tm,t,t]
  return p_rpc_ai_keygen_block1

def rpc_ai_keygen_block2(n,t,tri,paddnoiseto,pgdecode):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the block 2 of the
  KeyGen function of the Raccoon scheme, see Figure 6 of [BelRivRos25] to identify the
  block 2.

  Args:
      n (int): Number of shares.
      t (int): Threshold number.
      tri (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      pgaddnoiseto (np.ndarray): Cardinal RPC envelope of AddNoiseTo.
      pgdecode (np.ndarray): Cardinal RPC envelope of Gdecode.

  Returns:
      float:  t-threshold RPC advantage $\varepsilon$ of the block 2 of the
      KeyGen function.
  """
  p_rpc_ai_keygen_block2 = 0
  filtered_tin = [(i, j) for i, j in itertools.product(range(n + 1), repeat=2) if i + j > tri]
  for tm in range(n+1):
    for tin in filtered_tin:
      for tin_block2 in range(t+1,n+1):
        p_rpc_ai_keygen_block2 += paddnoiseto[tin_block2,tin[0],tin[1],tm] * pgdecode[tm]
  return p_rpc_ai_keygen_block2

def rpc_keygen(n, t, tri, p, pgref1, pgref2, pgref3):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the KeyGen function of
  the Raccoon scheme, by Figure 6 of [BelRivRos25], we add the three block together to
  get the t-threshold RPC property of KeyGen.

  Args:
      n (int): Number of shares.
      t (int): Threshold Number.
      tri (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      p (float): Leakage rate.
      pgref1 (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the block 1.
      pgref2 (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the block 2.
      pgref3 (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the Gcmatmult part.

  Returns:
      float: t-threshold RPC advantage $\varepsilon$ of the KeyGen function.
  """

  raccoon_l = 4
  raccoon_k = 5
  raccoon_n = 512 

  # block 1
  pgadd1 = cRPC_pgadd(n,p,pgref1)
  pgaddp1 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd1)
  pgcopy1 = cRPC_pgcopy(n,p,pgref1)
  p_rpc_ai_keygen_block1 = rpc_ai_keygen_block1(n,t,tri,pgaddp1,pgcopy1)*raccoon_l*raccoon_n

  # block 2
  pgadd2 = cRPC_pgadd(n,p,pgref2)
  pgaddp2 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd2)
  paddnoiseto = cardinal_rpc_addnoiseto_envelope_pgaddp_pgadd(n,2,pgaddp2,pgadd2)
  pgdecode = decode_crpcwo_envelope(n,p,pgref2)
  p_rpc_ai_keygen_block2 = rpc_ai_keygen_block2(n,t,tri,paddnoiseto,pgdecode)*raccoon_k*raccoon_n

  # Gcmatmult
  pgadd3 = cRPC_pgadd(n,p,pgref3)
  pgcopy3 = cRPC_pgcopy(n,p,pgref3)
  p_rpc_gcopytree = rpc_gcopytree(n,t,raccoon_k,pgcopy3) 
  p_rpc_gpolymult = rpc_gpolymult(n,t,p,raccoon_n,pgcopy3,pgadd3,pgref3)
  p_rpc_gsumtree = rpc_gsumtree(n,t,raccoon_l,pgadd3)
  p_rpc_gcmatmult = p_rpc_gcopytree*raccoon_l * raccoon_n + p_rpc_gpolymult*raccoon_k *raccoon_l + p_rpc_gsumtree*raccoon_k*raccoon_n

  p_keygen = p_rpc_ai_keygen_block1+p_rpc_gcmatmult+p_rpc_ai_keygen_block2
  return p_keygen


#In the case of the RPRefresh, we optimize the number of iterations of the three
#RPRefresh used in each block, which gives the three following function.

def optimize_keygen_gamma1 (n, t, tri, p, eps_witness, thr, compute_pgref_RPRefresh) :

  gamma2 = 500
  pgref2 = compute_pgref_RPRefresh(n, p, gamma2)

  gamma3 = 500
  pgref3 = compute_pgref_RPRefresh(n, p, gamma3)

  #We start with 50.
  gamma1_0 = 50
  pgref1_0 = compute_pgref_RPRefresh(n, p, gamma1_0)

  eps0 = rpc_keygen(n, t, tri, p, pgref1_0, pgref2, pgref3)
  
  gamma1_1 = gamma1_0
  eps1 = eps0

  if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
    while (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr and not(gamma1_0 == 0)) : 
      eps1 = eps0
      gamma1_1 = gamma1_0
      gamma1_0 = gamma1_1 // 2

      pgref1_0 = compute_pgref_RPRefresh(n, p, gamma1_0)
      eps0 = rpc_keygen(n, t, tri, p, pgref1_0, pgref2, pgref3)

    if (gamma1_0 == 0) :
      if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
        return eps0, gamma1_0
      else :
        return eps1, gamma1_0 + 1
    
  else :
    while (np.abs(math.log(eps1, 2) - math.log(eps_witness, 2)) >= thr) :
      eps0 = eps1
      gamma1_1 = 2 * gamma1_1
      pgref1_1 = compute_pgref_RPRefresh(n, p, gamma1_1)
      eps1 = rpc_keygen(n, t, tri, p, pgref1_1, pgref2, pgref3)
      
    gamma1_0 = gamma1_1 // 2

  #We have gamma1_0 which is upper the threshold and gamma1_1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma1_0 == gamma1_1) and not (gamma1_0 == (gamma1_1 - 1))) :
    gamma1_2 = (gamma1_0 + gamma1_1) // 2
    pgref1_2 = compute_pgref_RPRefresh(n, p, gamma1_2)
    eps2 = rpc_keygen(n, t, tri, p, pgref1_2, pgref2, pgref3)

    if (np.abs(math.log(eps2, 2) - math.log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma1_1 = gamma1_2
    else :
      eps = eps2
      gamma1_0 = gamma1_2
    
  if (gamma1_0 == gamma1_1) :
    return eps0, gamma1_0

  if (gamma1_0 == (gamma1_1 - 1)):
    return eps1, gamma1_1


def optimize_keygen_gamma2 (n, t, tri, p, eps_witness, thr, gamma1, compute_pgref_RPRefresh) :

  logp = int(math.log(p, 2))
  
  pgref1 = compute_pgref_RPRefresh(n, p, gamma1)

  gamma3 = 500
  pgref3 = compute_pgref_RPRefresh(n, p, gamma3)

  #We start with 50.
  gamma2_0 = 50
  pgref2_0 = compute_pgref_RPRefresh(n, p, gamma2_0)

  eps0 = rpc_keygen(n, t, tri, p, pgref1, pgref2_0, pgref3)
  
  gamma2_1 = gamma2_0
  eps1 = eps0

  if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
    while (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr and not(gamma2_0 == 0)) : 
      eps1 = eps0
      gamma2_1 = gamma2_0
      gamma2_0 = gamma2_1 // 2

      pgref2_0 = compute_pgref_RPRefresh(n, p, gamma2_0)
      eps0 = rpc_keygen(n, t, tri, p, pgref1, pgref2_0, pgref3)

    if (gamma2_0 == 0) :
      if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
        return eps0, gamma2_0
      else :
        return eps1, gamma2_0 + 1
    
  else :
    while (np.abs(math.log(eps1, 2) - math.log(eps_witness, 2)) >= thr) :
      eps0 = eps1
      gamma2_1 = 2 * gamma2_1
      pgref2_1 = compute_pgref_RPRefresh(n, p, gamma2_1)
      eps1 = rpc_keygen(n, t, tri, p, pgref1, pgref2_1, pgref3)
      
    gamma2_0 = gamma2_1 // 2

  #We have gamma2_0 which is upper the threshold and gamma2_1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma2_0 == gamma2_1) and not (gamma2_0 == (gamma2_1 - 1))) :
    gamma2_2 = (gamma2_0 + gamma2_1) // 2
    pgref2_2 = compute_pgref_RPRefresh(n, p, gamma2_2)
    eps2 = rpc_keygen(n, t, tri, p, pgref1, pgref2_2, pgref3)

    if (np.abs(math.log(eps2, 2) - math.log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma2_1 = gamma2_2
    else :
      eps = eps2
      gamma2_0 = gamma2_2
    
  if (gamma2_0 == gamma2_1) :
    return eps0, gamma2_0

  if (gamma2_0 == (gamma2_1 - 1)):
    return eps1, gamma2_1

def optimize_keygen_gamma3 (n, t, tri, p, eps_witness, thr, gamma1, gamma2, compute_pgref_RPRefresh) :

  logp = int(math.log(p, 2))
  
  pgref1 = compute_pgref_RPRefresh(n, p, gamma1)
  pgref2 = compute_pgref_RPRefresh(n, p, gamma2)

  #We start with 50.
  gamma3_0 = 50
  pgref3_0 = compute_pgref_RPRefresh(n, p, gamma3_0)

  eps0 = rpc_keygen(n, t, tri, p, pgref1, pgref2, pgref3_0)
  
  gamma3_1 = gamma3_0
  eps1 = eps0

  if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
    while (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr and not(gamma3_0 == 0)) : 
      eps1 = eps0
      gamma3_1 = gamma3_0
      gamma3_0 = gamma3_1 // 2

      pgref3_0 = compute_pgref_RPRefresh(n, p, gamma3_0)
      eps0 = rpc_keygen(n, t, tri, p, pgref1, pgref2, pgref3_0)

    if (gamma3_0 == 0) :
      if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
        return eps0, gamma3_0
      else :
        return eps1, gamma3_0 + 1
    
  else :
    while (np.abs(math.log(eps1, 2) - math.log(eps_witness, 2)) >= thr) :
      eps0 = eps1
      gamma3_1 = 2 * gamma3_1
      pgref3_1 = compute_pgref_RPRefresh(n, p, gamma3_1)
      eps1 = rpc_keygen(n, t, tri, p, pgref1, pgref2, pgref3_1)
      
    gamma3_0 = gamma3_1 // 2

  #We have gamma3_0 which is upper the threshold and gamma3_1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma3_0 == gamma3_1) and not (gamma3_0 == (gamma3_1 - 1))) :
    gamma3_2 = (gamma3_0 + gamma3_1) // 2
    pgref3_2 = compute_pgref_RPRefresh(n, p, gamma3_2)
    eps2 = rpc_keygen(n, t, tri, p, pgref1, pgref2, pgref3_2)

    if (np.abs(math.log(eps2, 2) - math.log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma3_1 = gamma3_2
    else :
      eps = eps2
      gamma3_0 = gamma3_2
    
  if (gamma3_0 == gamma3_1) :
    return eps0, gamma3_0

  if (gamma3_0 == (gamma3_1 - 1)):
    return eps1, gamma3_1


def rpc_keygen_optimized_RPRefresh (n, p, thr, compute_pgref_RPRefresh, is_ze) :  
  r"""
  Optimize the number of iterations made by the refresh gadget RPRefresh in
  each block of Keygen and return the advantage of the t-threshold RPC advantage
  $\varepsilon$ whenever the number of iterations are optimized, as well as
  these optimal number of iterations.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      thr (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      compute_pgref_RPRefresh (Callable[[int, float, int], np.ndarray]):
      Function to compute the cardinal RPC envelope of the RPRefresh gadget,
      according to the number of shares, the leakage rate and the number of
      iterations.  
      is_ze (bool): Indicate if we use RPRefresh in its Zero encoding version or
      its direct version.

  Returns:
      tuple[float, int, int, int]: A tuple containing the optimized RPC
      advantage $\varepsilon$ and the optimal numbers of iterations gamma1,
      gamma2, and gamma3.   
  """
  logp = int(math.log(p, 2))
  if(is_ze) :
    filename_prec = f"./prec/opti_gamma/Raccoon_KeyGen_ZE_opti_gamma_n{n}_p{logp}.npy"
  else : 
    filename_prec = f"./prec/opti_gamma/Raccoon_KeyGen_DI_opti_gamma_n{n}_p{logp}.npy"

  if os.path.exists(filename_prec):
    eps, gamma1, gamma2, gamma3 = np.load(filename_prec, allow_pickle=True)
    return eps, gamma1, gamma2, gamma3

  gamma1 = 500
  gamma2 = 500
  gamma3 = 500
  t = n // 2
  tri = n - 1

  pgref1 = compute_pgref_RPRefresh(n, p, gamma1)
  pgref2 = compute_pgref_RPRefresh(n, p, gamma2)
  pgref3 = compute_pgref_RPRefresh(n, p, gamma3)

  eps = rpc_keygen(n, t, tri, p, pgref1, pgref2, pgref3)

  eps, gamma1 = optimize_keygen_gamma1(n, t, tri, p, eps, thr, compute_pgref_RPRefresh)
  eps, gamma2 = optimize_keygen_gamma2 (n, t, tri, p,  eps, thr, gamma1, compute_pgref_RPRefresh)  
  eps, gamma3 = optimize_keygen_gamma3(n, t, tri, p, eps, thr, gamma1, gamma2, compute_pgref_RPRefresh)

  to_save = np.array(
        [eps, gamma1, gamma2, gamma3],
        dtype=object
  )
  np.save(filename_prec, to_save)
  
  return eps, gamma1, gamma2, gamma3

## -------------------------------------------------------------
##              Raccoon-128 - Sign
## -------------------------------------------------------------

## Signature 128-16 
def rpc_ai_sign_block1(n,t,tri,pgaddp,pgcopy,pgcpolymult,pgadd,pgdecode,raccoon_l,raccoon_n):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the block 1 of the
  Sign function of the Raccoon scheme, see Figure 7 of [BelRivRos25] to identify the
  block 1.

  Args:
      n (int): Number of shares.
      t (int): Threshold Number.
      tri (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      pgaddp (np.ndarray): Cardinal RPC envelope of Gsum-tree.
      pgcopy (np.ndarray): Cardinal RPC envelope of Gcopy.
      pgcpolymult (float): t-threshold RPC advantage $\varepsilon$ of the gadget
      Gcpmult.  
      pgadd (np.ndarray): Cardinal RPC envelope of the addition gadget.
      pgdecode (np.ndarray): Cardinal RPC-O envelope of the decode gadget.
      raccoon_l (int): Dimension of the secret vector in the Raccoon scheme.
      raccoon_n (int): Dimension of the lattice used in Raccoon scheme.

  Returns:
      float : t-threshold RPC advantage $\varepsilon$ of the block 1 of the
      KeyGen function.
  """
  p_rpc_ai_sign_sub_block1 = np.zeros((n+1,n+1,n+1))
  p_rpc_ai_sign_sub_block2 = np.zeros((n+1,n+1))
  p_rpc_ai_sign_block1 = 0
  filtered_tin = [(i, j) for i, j in itertools.product(range(n + 1), repeat=2) if i + j > tri]
  # Sub-block 1
  for tin in filtered_tin:
    for tm1 in range(0,n+1):
      for tm2 in range(0,n+1):
        p_rpc_ai_sign_sub_block1[tin[0],tin[1],tm2] += pgaddp[tin[0],tin[1],tm1] * pgcopy[tm1,tm2,t]
  # Sub-block 2
  for tin_s in range(t+1,n+1):
    for tm1 in range(0,n+1):
      for tm2 in range(0,n+1):
        for tm3 in range(0,n+1):
          p_rpc_ai_sign_sub_block2[tin_s,tm1] += pgadd[tm2,tm1,tm3]*pgdecode[tm3]

  for tin in filtered_tin:
    for tin_s in range(t+1,n+1):
      for tm in range(0,n+1):
        p_rpc_ai_sign_block1 += p_rpc_ai_sign_sub_block1[tin[0],tin[1],tm]*p_rpc_ai_sign_sub_block2[tin_s,tm]

  p_rpc_ai_sign_block1 = p_rpc_ai_sign_block1*raccoon_l*raccoon_n+pgcpolymult*raccoon_l
  return p_rpc_ai_sign_block1

def rpc_sign(n,t,tri,p,pgref1,pgref2,pgref3):
  r"""
  Compute the t-threshold RPC advantage $\varepsilon$ of the Sign function of
  the Raccoon scheme, by Figure 7 of [BelRivRos25], we add the three block together to
  get the t-threshold RPC property of Sign.

  Args:
      n (int): Number of shares.
      t (int): Threshold Number.
      tri (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      p (float): Leakage rate.
      pgref1 (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the block 1.
      pgref2 (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the block 2.
      pgref3 (np.ndarray): Cardinal RPC envelope of the refresh gadget used in
      the Gcmatmult part.

  Returns:
      float: t-threshold RPC advantage $\varepsilon$ of the Sign function.
  """

  raccoon_l = 4
  raccoon_k = 5
  raccoon_n = 512 

  # block 1
  pgadd1 = cRPC_pgadd(n,p,pgref1)
  pgaddp1 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd1)
  pgcopy1 = cRPC_pgcopy(n,p,pgref1)
  pgcpolymult = rpc_gpolymult(n,t,p,raccoon_n,pgcopy1,pgadd1,pgref1)
  pgdecode = decode_crpcwo_envelope(n,p,pgref1)
  p_rpc_ai_sign_block1 = rpc_ai_sign_block1(n,t,tri,pgaddp1,pgcopy1,pgcpolymult,pgadd1,pgdecode,raccoon_l,raccoon_n)

  # block 2
  pgadd2 = cRPC_pgadd(n,p,pgref2)
  pgaddp2 = cardinal_rpc_gaddptree_envelope_pgadd(n,2,pgadd2)
  paddnoiseto = cardinal_rpc_addnoiseto_envelope_pgaddp_pgadd(n,2,pgaddp2,pgadd2)
  pgdecode = decode_crpcwo_envelope(n,p,pgref2)
  p_rpc_ai_sign_block2 = rpc_ai_keygen_block2(n,t,tri,paddnoiseto,pgdecode)*raccoon_k*raccoon_n

  # Gcmatmult
  pgadd3 = cRPC_pgadd(n,p,pgref3)
  pgcopy3 = cRPC_pgcopy(n,p,pgref3)
  p_rpc_gcopytree = rpc_gcopytree(n,t,raccoon_k,pgcopy3) 
  p_rpc_gpolymult = rpc_gpolymult(n,t,p,raccoon_n,pgcopy3,pgadd3,pgref3)
  p_rpc_gsumtree = rpc_gsumtree(n,t,raccoon_l,pgadd3)
  p_rpc_gcmatmult = p_rpc_gcopytree*raccoon_l * raccoon_n + p_rpc_gpolymult*raccoon_k *raccoon_l + p_rpc_gsumtree*raccoon_k*raccoon_n

  p_keygen = p_rpc_ai_sign_block1+p_rpc_gcmatmult+p_rpc_ai_sign_block2
  return p_keygen

#In the case of the RPRefresh, we optimize the number of iterations of the three
#RPRefresh used in each block, which gives the three following function.

def optimize_sign_gamma1 (n, t, tri, p, eps_witness, thr, compute_pgref_RPRefresh) :

  logp = int(math.log(p, 2))
  
  gamma2 = 500
  pgref2 = compute_pgref_RPRefresh(n, p, gamma2)

  gamma3 = 500
  pgref3 = compute_pgref_RPRefresh(n, p, gamma3)

  #We start with 50.
  gamma1_0 = 50
  pgref1_0 = compute_pgref_RPRefresh(n, p, gamma1_0)

  eps0 = rpc_sign(n, t, tri, p, pgref1_0, pgref2, pgref3)
  
  gamma1_1 = gamma1_0
  eps1 = eps0

  if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
    while (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr and not(gamma1_0 == 0)) : 
      eps1 = eps0
      gamma1_1 = gamma1_0
      gamma1_0 = gamma1_1 // 2

      pgref1_0 = compute_pgref_RPRefresh(n, p, gamma1_0)
      eps0 = rpc_sign(n, t, tri, p, pgref1_0, pgref2, pgref3)

    if (gamma1_0 == 0) :
      if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
        return eps0, gamma1_0
      else :
        return eps1, gamma1_0 + 1
    
  else :
    while (np.abs(math.log(eps1, 2) - math.log(eps_witness, 2)) >= thr) :
      eps0 = eps1
      gamma1_1 = 2 * gamma1_1
      pgref1_1 = compute_pgref_RPRefresh(n, p, gamma1_1)
      eps1 = rpc_sign(n, t, tri, p, pgref1_1, pgref2, pgref3)
      
    gamma1_0 = gamma1_1 // 2

  #We have gamma1_0 which is upper the threshold and gamma1_1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma1_0 == gamma1_1) and not (gamma1_0 == (gamma1_1 - 1))) :
    gamma1_2 = (gamma1_0 + gamma1_1) // 2
    pgref1_2 = compute_pgref_RPRefresh(n, p, gamma1_2)
    eps2 = rpc_sign(n, t, tri, p, pgref1_2, pgref2, pgref3)

    if (np.abs(math.log(eps2, 2) - math.log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma1_1 = gamma1_2
    else :
      eps = eps2
      gamma1_0 = gamma1_2
    
  if (gamma1_0 == gamma1_1) :
    return eps0, gamma1_0

  if (gamma1_0 == (gamma1_1 - 1)):
    return eps1, gamma1_1


def optimize_sign_gamma2 (n, t, tri, p, eps_witness, thr, gamma1, compute_pgref_RPRefresh) :

  logp = int(math.log(p, 2))
  
  pgref1 = compute_pgref_RPRefresh(n, p, gamma1)

  gamma3 = 500
  pgref3 = compute_pgref_RPRefresh(n, p, gamma3)

  #We start with 50.
  gamma2_0 = 50
  pgref2_0 = compute_pgref_RPRefresh(n, p, gamma2_0)

  eps0 = rpc_sign(n, t, tri, p, pgref1, pgref2_0, pgref3)
  
  gamma2_1 = gamma2_0
  eps1 = eps0

  if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
    while (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr and not(gamma2_0 == 0)) : 
      eps1 = eps0
      gamma2_1 = gamma2_0
      gamma2_0 = gamma2_1 // 2

      pgref2_0 = compute_pgref_RPRefresh(n, p, gamma2_0)
      eps0 = rpc_sign(n, t, tri, p, pgref1, pgref2_0, pgref3)

    if (gamma2_0 == 0) :
      if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
        return eps0, gamma2_0
      else :
        return eps1, gamma2_0 + 1
    
  else :
    while (np.abs(math.log(eps1, 2) - math.log(eps_witness, 2)) >= thr) :
      eps0 = eps1
      gamma2_1 = 2 * gamma2_1
      pgref2_1 = compute_pgref_RPRefresh(n, p, gamma2_1)
      eps1 = rpc_sign(n, t, tri, p, pgref1, pgref2_1, pgref3)
      
    gamma2_0 = gamma2_1 // 2

  #We have gamma2_0 which is upper the threshold and gamma2_1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma2_0 == gamma2_1) and not (gamma2_0 == (gamma2_1 - 1))) :
    gamma2_2 = (gamma2_0 + gamma2_1) // 2
    pgref2_2 = compute_pgref_RPRefresh(n, p, gamma2_2)
    eps2 = rpc_sign(n, t, tri, p, pgref1, pgref2_2, pgref3)

    if (np.abs(math.log(eps2, 2) - math.log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma2_1 = gamma2_2
    else :
      eps = eps2
      gamma2_0 = gamma2_2
    
  if (gamma2_0 == gamma2_1) :
    return eps0, gamma2_0

  if (gamma2_0 == (gamma2_1 - 1)):
    return eps1, gamma2_1

def optimize_sign_gamma3 (n, t, tri, p, eps_witness, thr, gamma1, gamma2, compute_pgref_RPRefresh) :

  logp = int(math.log(p, 2))
  
  pgref1 = compute_pgref_RPRefresh(n, p, gamma1)
  pgref2 = compute_pgref_RPRefresh(n, p, gamma2)

  #We start with 50.
  gamma3_0 = 50
  pgref3_0 = compute_pgref_RPRefresh(n, p, gamma3_0)

  eps0 = rpc_sign(n, t, tri, p, pgref1, pgref2, pgref3_0)
  
  gamma3_1 = gamma3_0
  eps1 = eps0

  if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
    while (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr and not(gamma3_0 == 0)) : 
      eps1 = eps0
      gamma3_1 = gamma3_0
      gamma3_0 = gamma3_1 // 2

      pgref3_0 = compute_pgref_RPRefresh(n, p, gamma3_0)
      eps0 = rpc_sign(n, t, tri, p, pgref1, pgref2, pgref3_0)

    if (gamma3_0 == 0) :
      if (np.abs(math.log(eps0, 2) - math.log(eps_witness, 2)) < thr) :
        return eps0, gamma3_0
      else :
        return eps1, gamma3_0 + 1
    
  else :
    while (np.abs(math.log(eps1, 2) - math.log(eps_witness, 2)) >= thr) :
      eps0 = eps1
      gamma3_1 = 2 * gamma3_1
      pgref3_1 = compute_pgref_RPRefresh(n, p, gamma3_1)
      eps1 = rpc_sign(n, t, tri, p, pgref1, pgref2, pgref3_1)
      
    gamma3_0 = gamma3_1 // 2

  #We have gamma3_0 which is upper the threshold and gamma3_1 which is under 
  #the threshold, we apply dichotomy.
  while (not (gamma3_0 == gamma3_1) and not (gamma3_0 == (gamma3_1 - 1))) :
    gamma3_2 = (gamma3_0 + gamma3_1) // 2
    pgref3_2 = compute_pgref_RPRefresh(n, p, gamma3_2)
    eps2 = rpc_sign(n, t, tri, p, pgref1, pgref2, pgref3_2)

    if (np.abs(math.log(eps2, 2) - math.log(eps_witness, 2))< thr) :
      eps1 = eps2
      gamma3_1 = gamma3_2
    else :
      eps = eps2
      gamma3_0 = gamma3_2
    
  if (gamma3_0 == gamma3_1) :
    return eps0, gamma3_0

  if (gamma3_0 == (gamma3_1 - 1)):
    return eps1, gamma3_1

def rpc_sign_optimized_RPRefresh (n, p, thr, compute_pgref_RPRefresh, is_ze) :
  r"""
  Optimize the number of iterations made by the refresh gadget RPRefresh in
  each block of Sign and return the advantage of the t-threshold RPC advantage
  $\varepsilon$ whenever the number of iterations are optimized, as well as
  these optimal number of iterations.

  Args:
      n (int): Number of shares.
      p (float): Leakage rate.
      thr (int): Maximum number of auxiliary input leaked (i.e. the $r_i$ and
      $r'_i$). 
      compute_pgref_RPRefresh (Callable[[int, float, int], np.ndarray]):
      Function to compute the cardinal RPC envelope of the RPRefresh gadget,
      according to the number of shares, the leakage rate and the number of
      iterations.  
      is_ze (bool): Indicate if we use RPRefresh in its Zero encoding version or
      its direct version.

  Returns:
      tuple[float, int, int, int]: A tuple containing the optimized RPC
      advantage $\varepsilon$ and the optimal numbers of iterations gamma1,
      gamma2, and gamma3.   
  """
    
  logp = int(math.log(p, 2))

  if(is_ze) :
    filename_prec = f"./prec/opti_gamma/Raccoon_Sign_ZE_opti_gamma_n{n}_p{logp}.npy"
  else : 
    filename_prec = f"./prec/opti_gamma/Raccoon_Sign_DI_opti_gamma_n{n}_p{logp}.npy"


  if os.path.exists(filename_prec):
    eps, gamma1, gamma2, gamma3 = np.load(filename_prec, allow_pickle=True)
    return eps, gamma1, gamma2, gamma3


  gamma1 = 500
  gamma2 = 500
  gamma3 = 500
  t = n // 2
  tri = n - 1

  pgref1 = compute_pgref_RPRefresh(n, p, gamma1)
  pgref2 = compute_pgref_RPRefresh(n, p, gamma2)
  pgref3 = compute_pgref_RPRefresh(n, p, gamma3)

  eps = rpc_sign(n, t, tri, p, pgref1, pgref2, pgref3)

  eps, gamma1 = optimize_sign_gamma1(n, t, tri, p, eps, thr, compute_pgref_RPRefresh)
  eps, gamma2 = optimize_sign_gamma2 (n, t, tri, p,  eps, thr, gamma1, compute_pgref_RPRefresh)  
  eps, gamma3 = optimize_sign_gamma3(n, t, tri, p, eps, thr, gamma1, gamma2, compute_pgref_RPRefresh)

  to_save = np.array(
    [eps, gamma1, gamma2, gamma3],dtype=object
  )
  np.save(filename_prec, to_save)

  return eps, gamma1, gamma2, gamma3




