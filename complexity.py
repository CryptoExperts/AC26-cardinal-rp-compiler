#!/usr/bin/env python3

################################################################################
# This file enables the computation of the complexity of the schemes evaluated
# in our article, namely AES and Raccoon. It computes the randomness, addition,
# and multiplication complexities, as well as the cost of the permutations.
#
# The AES complexity computations are largely inspired by the "complexity.py"
# file available in the following GitHub repository:
#
#   https://github.com/CryptoExperts/AC25-cardinal-rp-compiler
#
# The Raccoon complexity computations are largely inspired by the code available
# in the following GitHub repository:
#
#   https://github.com/CryptoExperts/EC25-random-probing-Raccoon
################################################################################



################################################################################
################################ Packages ######################################
import numpy as np
from math import log, ceil
import math
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
################################################################################
########################### Permutations Complexity ############################

def nbP(nx, ny, k) :
  """
  Args:
    nx (int): Number of shares used for the first secret variable 'x'.
    ny (int): Number of sahres used for the second secret variable 'y'.
    k (int) : An integer > 1
  
  Returns: 
    The number of permutations of size k used in UnifMatMult (appendix H, 
    algorithm 12 of the paper).

  Notes:
    Explore recursively all invocations of UnifMatMult and check when nx or ny is 
    equal to k. This corresponds to the use of permutations of size k.
  """
  
  if (k > max (nx, ny)) :
    return 0
  
  nb = 0
  
  nxhl = nx // 2
  nxhr = nx - nxhl
  nyhl = ny // 2
  nyhr = ny - nyhl
  
  #Base case
  if (k == nxhl or k == nxhr) :
    nb += 2
  
  if (k == nyhl or k == nyhr) :
    nb += 2
  
  #Recursive Call
  nb += nbP(nxhl, nyhl, k) + nbP(nxhr, nyhl, k) + nbP(nxhl, nyhr, k) + nbP(nxhr, nyhr, k)
  
  return nb


def comp_perm (n) :
  """
  Args:
    n (int):  the size of the permutation.

  Returns:  
    The number of random bits necessary to generate the permutation.

  Notes: 
    Compared to the |Knuth-Yates| shuffling presented in Algorithm 10 of the 
    paper, we omit the rejection that can happened when we draw an uniform in a 
    range {1, ..., i} where i is not necessary a power of two.
  """
  
  nb_rand_bits = 0 
  for l in range (2, n  + 1) :
     nb_rand_bits += int(np.ceil(log(l, 2)))
  return nb_rand_bits

def nbRandBits (n) :
  """
  Args:
    n (int): Number of shares.
  
  Returns: 
    Compute the number of random bits necessary when we apply |UnifMatMult| 
    with |n| shares.
  """  
  
  nb = 0
  for k in range (2, n + 1) :
    cperm_k = comp_perm(k)
    nb += cperm_k * nbP(n, n, k)
  return nb

################################################################################  
######################### Compiler gadget complexity ###########################

def comp_ref_rprefresh(n, gamma) :
  """
  Complexity of the RPRefresh.

  Args:
      n (int): Number of shares.
      gamma (int): Number of iterations.

  Returns:
      tuple[int, int, int]: Randomness complexity, Addition complexity,
      Permutation complexity. 
  """
  if (n == 1) :
    return 0, 0, 0
  
  nb_rand = gamma
  nb_add = 2 * gamma + n
  if (gamma == 0) :
    nb_add = 0

  nb_rand_bit =  gamma * (int(np.ceil(log(n, 2))) + int(np.ceil(log(n - 1, 2))))
  return nb_rand, nb_add, nb_rand_bit

def comp_ref_linear (n) :
  """
  Complexity of the Linear refresh.

  Args:
      n (int): Number of shares.

  Returns:
      tuple[int, int, int]: Randomness complexity, Addition complexity,
      Permutation complexity. 
  """
  if (n == 1) :
    return 0, 0, 0
  
  nb_ref = n - 1
  nb_add = 2 * nb_ref

  return nb_ref, nb_add, 0


def comp_ref_half (n) :
  """
  Complexity of the Half refresh.

  Args:
      n (int): Number of shares.

  Returns:
      tuple[int, int, int]: Randomness complexity, Addition complexity,
      Permutation complexity. 
  """
  if (n == 1) :
    return 0, 0, 0
  
  nb_ref = n // 2
  nb_add = 2 * nb_ref

  return nb_ref, nb_add, 0

def comp_ref_doublehalf (n) :
  """
  Complexity of the DoubleHalf refresh.

  Args:
      n (int): Number of shares.

  Returns:
      tuple[int, int, int]: Randomness complexity, Addition complexity,
      Permutation complexity. 
  """
  if (n == 1) :
    return 0, 0, 0
  
  nb_ref = 2 * (n // 2)
  nb_add = 2 * nb_ref
  return nb_ref, nb_add, 0

def comp_ref_nlogn(n) :
  """
  Complexity of the O(nlogn) refresh.

  Args:
      n (int): Number of shares.

  Returns:
      tuple[int, int, int]: Randomness complexity, Addition complexity,
      Permutation complexity. 
  """
  if (n == 1) :
    return 0, 0, 0
  
  if (n == 2) :
    return 1, 2, 0
  
  if (n == 3) :
    return 2, 4, 0

  
  nh = n // 2
  nb_ref1, nb_add1, _ = comp_ref_nlogn(nh)
  nb_ref2, nb_add2, _= comp_ref_nlogn(n - nh)

  nb_ref = nh + nb_ref1 + nb_ref2
  nb_add = n - (n % 2) + nb_add1 + nb_add2
  return nb_ref, nb_add, 0

def comp_add (n, comp_ref) :
  """
  Compute the complexity (random, addition) of Addition gadget of the compiler.
  
  Args:
    n (int): Number of shares.
  
  Returns:
    The random field complexity, the addition complexity and the random bits 
    complexity.

  Notes:
    If |gamma| is equal to zero, then there is no refresh gadget used.
  """

  if (n == 1) :
    return 0, n, 0
  
  nb_rd_ref, nb_add_ref, nb_rd_bit_ref = comp_ref(n)


  nb_rand = 2 * nb_rd_ref
  nb_add = 2 * nb_add_ref + n
  nb_rand_bit =  2 * nb_rd_bit_ref *  (int(np.ceil(log(n, 2))) + int(np.ceil(log(n - 1, 2))))
  return nb_rand, nb_add, nb_rand_bit

def comp_cadd (n, comp_ref) :
  """
  Compute the complexity (random, addition) of the Addition by a constant gadget
  of the compiler.
  
  Args:
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns: 
    The random field complexity, the addition complexity and the random bits 
    complexity.
  
  Notes:
    If |gamma| is equal to zero, then there is no refresh gadget used.
  """
  
  if (n == 1) :
    return 0, 1, 0
  
  nb_rand, nb_add, nb_rand_bit = comp_ref(n)
  nb_add += 1 
  return nb_rand, nb_add, nb_rand_bit
  
def comp_cmult (n, comp_ref):
  """
  Compute the complexity (random, addition, multiplication) of the 
  Multiplication by a constant gadget of the compiler.
  
  Args:
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns: 
    The random, addition and multiplication complexity.
  """

  if (n == 1) :
    return 0, 0, 1, 0
  
  nb_rand, nb_add, nb_rand_bit = comp_ref(n) 
  nb_mult = n
  return nb_rand, nb_add, nb_mult, nb_rand_bit


def comp_matmult (nx, ny, l_comp_ref_MM) : 
  """
  Compute the complexity (random, addition, multiplication) of MatMultSym 
  (Algorihtm 11 of the paper) of the compiler.
  
  Args:
    nx (int): Number of shares of the first value x.
    ny (int): Number of shares of the second secret value y.
    l_gamma(list): List of gamma used in RPZeroEnc according to the number of 
                  shares.
  
  Returns:
    The random, addition, multiplication complexity.
  """
  
  if nx == 1 and ny == 1 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 1
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
  if nx == 2 and ny == 1 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 2
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
  if nx == 1 and ny == 2 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 2
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
  if nx == 2 and ny == 2 :
    nb_rand = 0
    nb_add = 0
    nb_mult = 4
    nb_rand_bit = 0
    return nb_rand, nb_add, nb_mult, nb_rand_bit
    
  nxhl = nx // 2
  nxhr = nx - nxhl
  nyhl = ny // 2
  nyhr = ny - nyhl
  
  nb_rand1, nb_add1, tmp, nb_rand_bit1 = comp_matmult(nxhl, nyhl, l_comp_ref_MM)
  nb_rand2, nb_add2, tmp, nb_rand_bit2 = comp_matmult(nxhr, nyhl, l_comp_ref_MM)
  nb_rand3, nb_add3, tmp, nb_rand_bit3 = comp_matmult(nxhl, nyhr, l_comp_ref_MM)
  nb_rand4, nb_add4, tmp, nb_rand_bit4 = comp_matmult(nxhr, nyhr, l_comp_ref_MM)
  
  nb_rand_rec = nb_rand1 + nb_rand2 + nb_rand3 + nb_rand4
  nb_add_rec = nb_add1 + nb_add2 + nb_add3 + nb_add4
  nb_rand_bit_rec = nb_rand_bit1 + nb_rand_bit2 + nb_rand_bit3 + nb_rand_bit4
  
  comp_ref5 = l_comp_ref_MM[nxhl]
  comp_ref6 = l_comp_ref_MM[nxhr]
  comp_ref7 = l_comp_ref_MM[nyhl]
  comp_ref8 = l_comp_ref_MM[nyhr]

  nb_rand5, nb_add5, nb_rand_bit5 = comp_ref5(nxhl)
  nb_rand6, nb_add6, nb_rand_bit6 = comp_ref6(nxhr)
  nb_rand7, nb_add7, nb_rand_bit7 = comp_ref7(nyhl)
  nb_rand8, nb_add8, nb_rand_bit8 = comp_ref8(nyhr)
  
  nb_rand_ref = 2 * (nb_rand5 + nb_rand6 + nb_rand7 + nb_rand8)
  nb_add_ref = 2 * (nb_add5 + nb_add6 + nb_add7 + nb_add8)
  nb_rand_bit_ref = 2 * (nb_rand_bit5 + nb_rand_bit6 + nb_rand_bit7 + nb_rand_bit8)
  
  
  nb_rand = nb_rand_ref + nb_rand_rec
  nb_add = nb_add_ref + nb_add_rec
  nb_mult = nx * ny
  nb_rand_bit = nb_rand_bit_ref + nb_rand_bit_rec
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit

def comp_tree_add (k, n, comp_ref) :
  """
  Compute the complexity (random, addition) of TreeAdd of the compiler.
  
  Args:
    k (int): Number of n-sharing to add together.
    n (int): Number of shares.
    gamma (int): Number of iteration in RPZeroEnc.
  
  Returns:
    The random, addition complexity.
  """

  if k == 1 :
    return 0, 0, 0
  
  if k == 2 :
    nb_rand, nb_add, nb_rand_bit = comp_add(n, comp_ref)
    return nb_rand, nb_add, nb_rand_bit
  
  #k > 2
  khdo = k // 2
  khup = k - khdo
  
  nb_rand1, nb_add1, nb_rand_bit1 = comp_tree_add(khdo, n, comp_ref)
  nb_rand2, nb_add2, nb_rand_bit2 = comp_tree_add(khup, n, comp_ref)
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, comp_ref)
  
  nb_rand = nb_rand1 + nb_rand2 + nb_rand_add
  nb_add = nb_add1 + nb_add2 + nb_add_add
  nb_rand_bit = nb_rand_bit1 + nb_rand_bit2 + nb_rand_bit_add
  return nb_rand , nb_add, nb_rand_bit

def comp_mult (n, comp_ref, l_comp_ref_MM) :
  """
  Compute the complexity (random, addition, multiplication) of CardSecMult.
  
  Args:
    n (int): Number of shares
    l_gamma (list): List of gamma used in MatMultSym according to the number of 
                    shares.
    gamma (int): gamma used in TreeAdd.
  
  Returns:
    The random, addition, multiplication complexity.
  """

  nb_rand, nb_add, nb_mult, nb_rand_bit = comp_matmult(n, n, l_comp_ref_MM)
  nb_rand2, nb_add2, nb_rand_bit2 = comp_tree_add(n, n, comp_ref)
  
  nb_rand_fin = nb_rand + nb_rand2
  nb_add_fin = nb_add + nb_add2
  nb_mult_fin = nb_mult
  nb_rand_bit_fin = nb_rand_bit + nb_rand_bit2  

  return nb_rand_fin, nb_add_fin, nb_mult_fin, nb_rand_bit_fin
  
def comp_mult_unif (n, comp_ref, l_comp_ref_MM) :
  """
  Compute the complexity (random, addition, multiplication) of CardSecMult.
  
  Args:
    n (int): Number of shares
    l_gamma (list): List of gamma used in UnifMatMult according to the number of 
                    shares.
    gamma (int): gamma used in TreeAdd.
  
  Returns
    The random, addition, multiplication complexity.
  """

  #Warning : It gives us the number of random bits of MatMultSym, it lacks the 
  # random bits used in the permutation. This is why we add |nbRandBits(n)| in 
  # the following.
  nb_rand, nb_add, nb_mult, nb_rand_bit = comp_matmult(n, n, l_comp_ref_MM)
  nb_rand2, nb_add2, nb_rand_bit2 = comp_tree_add(n, n, comp_ref)

  nb_rand_fin = nb_rand + nb_rand2
  nb_add_fin = nb_add + nb_add2
  nb_mult_fin = nb_mult
  nb_rand_bit_fin = nb_rand_bit + nb_rand_bit2 + nbRandBits(n) 
  
  return nb_rand_fin, nb_add_fin, nb_mult_fin, nb_rand_bit_fin

################################################################################
################################################################################



def comp_addrk (n, comp_ref) :
  """
  Compute the complexity of the step AddRoundKey in AES, with input the state of
  16 bytes where each byte is masked by n shares.
  
  Args:
    n (int): The number of shares.
    gamma_ark (int): The gamma used in the refresh of the step AddRoundKey.
  
  Returns: 
    The random, addition complexity of AddRoundKey.
  """

  nb_rand, nb_add, nb_rand_bit = comp_add(n, comp_ref)
  nb_rand_fin = 16 * nb_rand
  nb_add_fin = 16 * nb_add
  nb_rand_bit_fin = 16 * nb_rand_bit
  
  return nb_rand_fin, nb_add_fin, nb_rand_bit_fin

def comp_expo_sb (n, comp_ref, l_comp_ref_MM) :
  """
  Compute the complexity of the exponentiation in the SubBytes step of AES, for 
  a single masked byte with n shares.
  
  Args:
    n (int) : Number of shares.
    l_gamma_sb (list): List of gamma for UnifMatMult in CardSecMult.
    gamma_sb (int): The gamma used in the linear step of the exponentiation 
                   (including TreeAdd)
  Returns: 
    Complexity of the Exponentiation of the SubBytes step for a single byte.
  """

  #There is in the Exponentiation gadget (see Figure 11 of the paper):
  # - 2 refresh gadgets.
  # - 7 squaring gadgets (which can be seen as multiplication by a constant 
  #   gadget in F_{256})
  # - 4 permutations.
  # - 4 multiplication gadgets.

  nb_rand_ref, nb_add_ref,  nb_rand_bit_ref = comp_ref(n)
  nb_rand_sq, nb_add_sq, nb_mult_sq, nb_rand_bit_sq = comp_cmult (n, comp_ref)
  nb_rand_mult, nb_add_mult, nb_mult_mult, nb_rand_bit_mult = (comp_mult_unif(n, comp_ref, l_comp_ref_MM))
  
  nb_rand = 4 * nb_rand_mult + 7 * nb_rand_sq + 2 * nb_rand_ref
  nb_add = 4 * nb_add_mult + 7 * nb_add_sq
  nb_mult = 4 * nb_mult_mult + 7 * nb_mult_sq
  nb_rand_bit = (4 * nb_rand_bit_mult + 7 * nb_rand_bit_sq + 
                 4 * comp_perm(n) + 2 * nb_rand_bit_ref)
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit
  
def comp_aff_sb (n, comp_ref) :
  """
  Compute the complexity of the affine function in the SubBytes step of AES, for
  a single masked byte with n shares.
  
  Args:
    n (int): Number of shares
    gamma_sb (int): The gamma used in the affine function for the refresh gadget.
  
  Returns: 
    Complexity of the affine function of the SubBytes step for a single byte.
  """
  
  #There is in the Affine gadget (see Figure 12 of the paper):
  # - 6 refresh gadgets.
  # - 7 multiplication by a constant gadgets.
  # - 7 squaring gadgets (which can be seen as multiplication by a constant 
  #   gadget in F_{256})
  # - 7 additions gadgets.
  # - 7 permutations.
  # - 1 addition by a constant gadget.

  
  nb_rand_ref, nb_add_ref, nb_rand_bit_ref = comp_ref(n)
  nb_rand_cadd, nb_add_cadd, nb_rand_bit_cadd = comp_cadd(n, comp_ref)  
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, comp_ref)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult, nb_rand_bit_cmult = (
    comp_cmult(n, comp_ref))

  nb_rand = (6 * nb_rand_ref + 14 * nb_rand_cmult + 7 * nb_rand_add + 
             nb_rand_cadd)
  nb_add = 6 * nb_add_ref + 14 * nb_add_cmult + 7 * nb_add_add + nb_add_cadd
  nb_mult = 14 * nb_mult_cmult
  nb_rand_bit = (6 * nb_rand_bit_ref + 14 * nb_rand_bit_cmult + 
                 7 * nb_rand_bit_add + nb_rand_bit_cadd + 7 * comp_perm(n))
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit
  
def comp_sb (n, comp_ref_sb, l_comp_ref_MM) :
  """
  Compute the complexity of SubBytes, with input the 16 masked bytes of the 
  state.
  
  Args:
    n (int): Number of shares.
    l_gamma_sb (list): List of gamma used in the refresh of UnifMatMult in 
                       CardSecMult,  according to the number of shares.
    gamma_sb (int): The gamma used in the linear part of SubBytes 
                    (including TreeAdd).
  
  Returns: 
    Complexity (random, addition, multiplication) of SubBytes for 16 masked 
    bytes.
  """

  nb_rand_exp, nb_add_exp, nb_mult_exp, nb_rand_bit_exp = comp_expo_sb(n, comp_ref_sb, l_comp_ref_MM)
  nb_rand_aff, nb_add_aff, nb_mult_aff, nb_rand_bit_aff = comp_aff_sb(n, comp_ref_sb)
   
  nb_rand = 16 * (nb_rand_exp + nb_rand_aff)
  nb_add = 16 * (nb_add_exp + nb_add_aff)
  nb_mult = 16 * (nb_mult_exp + nb_mult_aff)
  nb_rand_bit = 16 * (nb_rand_bit_exp + nb_rand_bit_aff)
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit
  

def comp_mc (n, comp_ref) :
  """
  Compute the complexity (random, addition, mult) of the MixColumns step, with 
  input 16 masked bytes.
  
  Args:
    n (int): Number of shares.
    gamma_mc (int): The gamma used for the refresh gadget in MixColumn.
  
  Returns: 
    Complexity (random, addition, multiplication) of MixColumn.
  """

  #There is in the MixColumns gadget (see Figure 13 of the paper):
  # - 8 refresh gadgets.
  # - 8 multiplication by a constant gadgets.
  # - 12 additions gadgets.
  # - 12 permutations.

  nb_rand_ref, nb_add_ref, nb_rand_bit_ref = comp_ref(n)
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, comp_ref)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult, nb_rand_bit_cmult = comp_cmult(n, comp_ref)
 
  nb_rand = 12 * nb_rand_add + 8 * nb_rand_cmult + 8 * nb_rand_ref
  nb_add = 12 * nb_add_add + 8 * nb_add_cmult + 8 * nb_rand_add + 8 * nb_add_ref
  nb_mult = 8 * nb_mult_cmult
  nb_rand_bit = (12 * nb_rand_bit_add + 8 * nb_rand_bit_cmult + 
                 8 * nb_rand_bit_ref + 12 * comp_perm(n))
  
  return 4 * nb_rand, 4 * nb_add, 4 * nb_mult, 4 * nb_rand_bit
  
def comp_AES_enc (n, comp_ref_ark, comp_ref_mc, comp_ref_sb, l_comp_ref_MM) :
  """
  Compute the complexity (random, addition, multiplication) of a complete AES 
  encryption.
  
  Args:
    n (int): Number of shares
    ark (int): Gamma for the AddRoundKey step.
    l_gamma_sb (list): Gamma used in UnifMatMult from CardSecMult.
    gamma_sb (int): Gamma for the SubBytes step.
    gamma_mc (int): Gamma for the MixColumns step.
  """
  
  nb_rand_ark, nb_add_ark, nb_rand_bit_ark = comp_addrk(n, comp_ref_ark)
  nb_rand_sb, nb_add_sb, nb_mult_sb, nb_rand_bit_sb = comp_sb(n, comp_ref_sb, l_comp_ref_MM)
  nb_rand_mc, nb_add_mc, nb_mult_mc, nb_rand_bit_mc = comp_mc(n, comp_ref_mc)
  
  nb_rand = 11 * nb_rand_ark + 10 * nb_rand_sb + 9 * nb_rand_mc
  nb_add = 11 * nb_add_ark + 10 * nb_add_sb + 9 * nb_add_mc
  nb_mult = 10 * nb_mult_sb + 9 * nb_mult_mc
  nb_rand_bit = 11 * nb_rand_bit_ark + 10 * nb_rand_bit_sb + 9 * nb_rand_bit_mc
  
  return nb_rand, nb_add, nb_mult, nb_rand_bit

################################################################################
############################ Raccoon Complexity ################################

def comp_copy(n, comp_ref):
  r"""
  Compute the complexity (random, addition) of the copy gadget of the Raccoon 
  compiler (see section E.1 of our article).
  
  Args:
    n (int): Number of shares.
    comp_ref (Callable[[int], tuple[int, int, int]]): Function returning the
    complexity of the refresh gadget for `n` shares, as a tuple `(nb_rand_ref,
    nb_add_ref, nb_rand_bit_ref)`.
  
  Returns:
    The random field complexity, the addition complexity and the random bits 
    complexity.
  """

  nb_rand_ref, nb_add_ref, nb_rand_bit_ref = comp_ref(n)
  return 2 * nb_rand_ref, 2 * nb_add_ref, 2 * nb_rand_bit_ref


def complexity_Gcpmult(n, nR, comp_ref):
  r"""
  Compute the complexity (random, addition) of the gadget Gcpmult of the Raccoon 
  scheme (see section 6 of [BelRivRos25] for more details).
  
  Args:
    n (int): Number of shares.
    nR (int) : Dimension of the lattice in the Raccon scheme. In particular, we
    work on nR coefficients in parallel.
    comp_ref (Callable[[int], tuple[int, int, int]]): Function returning the
    complexity of the refresh gadget for `n` shares, as a tuple `(nb_rand_ref,
    nb_add_ref, nb_rand_bit_ref)`.
  
  Returns:
    The random field complexity, the addition complexity, multiplication
    complexity and the random bits complexity.
  """

  nb_rand_copy, nb_add_copy, nb_rand_bit_copy = comp_copy(n, comp_ref)
  nb_rand_cmult, nb_add_cmult, nb_mult_cmult, nb_rand_bit_cmult = comp_cmult(n, comp_ref)
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, comp_ref)
    
  nb_gcopy = (nR - 1) * nR
  nb_gcmult = nR * nR
  nb_gadd = (nR - 1) * nR

  nb_rand = nb_rand_copy * nb_gcopy + nb_rand_cmult * nb_gcmult + nb_rand_add * nb_gadd
  nb_add = nb_add_copy * nb_gcopy + nb_add_cmult * nb_gcmult + nb_add_add * nb_gadd
  nb_mult = nb_mult_cmult * nb_gcmult
  nb_rand_bit = nb_rand_bit_copy * nb_gcopy + nb_rand_bit_cmult * nb_gcmult + nb_rand_bit_add * nb_gadd

  return nb_rand, nb_add, nb_mult, nb_rand_bit

def complexity_Gdecode(n, comp_ref):
  r"""
  Compute the complexity (random, addition) of the gadget Gdecode of the Raccoon 
  scheme (see section 4 of [BelRivRos25]).
  
  Args:
    n (int): Number of shares.
    comp_ref (Callable[[int], tuple[int, int, int]]): Function returning the
    complexity of the refresh gadget for `n` shares, as a tuple `(nb_rand_ref,
    nb_add_ref, nb_rand_bit_ref)`.
  
  Returns:
    The random field complexity, the addition complexity and the random bits 
    complexity.
  """    

  nb_rand_ref, nb_add_ref, nb_rand_bit_ref = comp_ref(n)
  sums_add_comp = n - 1
  return nb_rand_ref, nb_add_ref + sums_add_comp, nb_rand_bit_ref

def complexity_Raccoon_KeyGen(n, nb_r, raccoon_l, raccoon_k, raccoon_n, 
                              comp_ref1, comp_ref2, comp_ref3):
  r"""
  Compute the complexity (random, addition) of masked Keygen in the Raccoon 
  scheme with our compiler described in section E.1 of our article.

  Args:
      n (int): Number of shares.
      nb_r (int): Number of small uniform random values used per polynomial
      coefficient to generate the Raccoon noise. In the Raccoon parameters,
      this corresponds to `d * rep`.
      raccoon_l (int): Parameter l of the Raccoon scheme.
      raccoon_k (int): Parameter k of the Raccoon scheme.
      raccoon_n (int): Parameter n of the Raccoon scheme.
      comp_ref1 (Callable[[int], tuple[int, int, int]]): Function returning the
      complexity of the refresh gadget used in the first block of Keygen for 
      `n` shares. 
      comp_ref2 (Callable[[int], tuple[int, int, int]]): Function returning the
      complexity of the refresh gadget used in the second block of Keygen for 
      `n` shares. 
      comp_ref3 (Callable[[int], tuple[int, int, int]]): Function returning the
      complexity of the refresh gadget used in the Gcmatmult block of Keygen for 
      `n` shares. 

  Returns:
    The random field complexity, the addition complexity, multiplication
    complexity and the random bits complexity of Keygen.
  """
    
  # KG Block 1 : Gsum
  nb_gadd_b1 = (math.ceil(nb_r//n)-1)*raccoon_l*raccoon_n
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, comp_ref1)
    
  nb_rand_kg_gsum = nb_rand_add * nb_gadd_b1
  nb_add_kg_gsum = nb_add_add * nb_gadd_b1
  nb_rand_bit_kg_gsum = nb_rand_bit_add * nb_gadd_b1
    
  # KG Block 1 : Gcopy
  nb_gcopy_b1 = raccoon_l*raccoon_n
  nb_rand_copy, nb_add_copy, nb_rand_bit_copy = comp_copy(n, comp_ref1)
    
  nb_rand_kg_gcopy = nb_rand_copy * nb_gcopy_b1
  nb_add_kg_gcopy = nb_add_copy * nb_gcopy_b1
  nb_rand_bit_kg_gcopy = nb_rand_bit_copy * nb_gcopy_b1
    
  # KG Block 1
  nb_rand_kg_b1 = nb_rand_kg_gsum + nb_rand_kg_gcopy 
  nb_add_kg_b1 = nb_add_kg_gsum +  nb_add_kg_gcopy
  nb_rand_bit_kg_b1 = nb_rand_bit_kg_gsum +  nb_rand_bit_kg_gcopy

  # KG Block 2 : Gcopies^k
  nb_gcopy_b2 = (raccoon_k-1)*raccoon_l*raccoon_n
  nb_rand_copy, nb_add_copy, nb_rand_bit_copy = comp_copy(n, comp_ref2)
    
  nb_rand_kg_b2_gcopy = nb_rand_copy * nb_gcopy_b2
  nb_add_kg_b2_gcopy = nb_add_copy * nb_gcopy_b2
  nb_rand_bit_kg_b2_gcopy = nb_rand_bit_copy * nb_gcopy_b2

  # KG Block 2 : Gcpmult^k
  nb_gcpmult = raccoon_k*raccoon_l
  nb_rand_gcpmult, nb_add_gcpmult, nb_mult_gcpmult, nb_rand_bit_gcpmult = complexity_Gcpmult(n, raccoon_n, comp_ref2)
    
  nb_rand_kg_b2_gcpmult = nb_rand_gcpmult * nb_gcpmult
  nb_add_kg_b2_gcpmult = nb_add_gcpmult * nb_gcpmult
  nb_mult_kg_b2_gcpmult = nb_mult_gcpmult * nb_gcpmult
  nb_rand_bit_kg_b2_gcpmult = nb_rand_bit_gcpmult * nb_gcpmult


  # KG Block 2 : Gsum
  nb_gadd_b2 = (raccoon_l-1)*raccoon_k*raccoon_n
  nb_rand_add_b2, nb_add_add_b2, nb_rand_bit_add_b2 = comp_add(n, comp_ref2)

  nb_rand_kg_b2_gsum = nb_rand_add_b2 * nb_gadd_b2
  nb_add_kg_b2_gsum = nb_add_add_b2 * nb_gadd_b2
  nb_rand_bit_kg_b2_gsum = nb_rand_bit_add_b2 * nb_gadd_b2

  # KG Block 2
  nb_rand_kg_b2 = nb_rand_kg_b2_gcopy + nb_rand_kg_b2_gcpmult + nb_rand_kg_b2_gsum
  nb_add_kg_b2 = nb_add_kg_b2_gcopy +  nb_add_kg_b2_gcpmult + nb_add_kg_b2_gsum
  nb_mult_kg_b2 = nb_mult_kg_b2_gcpmult
  nb_rand_bit_kg_b2 = nb_rand_bit_kg_b2_gcopy + nb_rand_bit_kg_b2_gcpmult + nb_rand_bit_kg_b2_gsum

  # KG Block 3 : AddNoiseTo
  nb_gadd_b3 = (math.ceil(nb_r//n))*raccoon_k*raccoon_n
  nb_rand_add_b3, nb_add_add_b3, nb_rand_bit_add_b3 = comp_add(n, comp_ref3)

  nb_rand_kg_b3_ANT = nb_rand_add_b3 * nb_gadd_b3
  nb_add_kg_b3_ANT = nb_add_add_b3 * nb_gadd_b3
  nb_rand_bit_kg_b3_ANT = nb_rand_bit_add_b3 * nb_gadd_b3  

  # KG Block 3 : Gdecode
  nb_gdecode = raccoon_k*raccoon_n
  nb_rand_decode, nb_add_decode, nb_rand_bit_decode = complexity_Gdecode(n, comp_ref3)
    
  nb_rand_kg_b3_decode = nb_rand_decode * nb_gdecode
  nb_add_kg_b3_decode = nb_add_decode * nb_gdecode
  nb_rand_bit_kg_b3_decode = nb_rand_bit_decode * nb_gdecode  

  # KG Block 3
  nb_rand_kg_b3 = nb_rand_kg_b3_ANT + nb_rand_kg_b3_decode
  nb_add_kg_b3 = nb_add_kg_b3_ANT +  nb_add_kg_b3_decode
  nb_rand_bit_kg_b3 = nb_rand_bit_kg_b3_ANT +  nb_rand_bit_kg_b3_decode

  # KG 
  nb_rand_kg = nb_rand_kg_b1 + nb_rand_kg_b2 + nb_rand_kg_b3
  nb_add_kg = nb_add_kg_b1 + nb_add_kg_b2 + nb_add_kg_b3
  nb_mult_kg = nb_mult_kg_b2
  nb_rand_bit_kg = nb_rand_bit_kg_b1 + nb_rand_bit_kg_b2 + nb_rand_bit_kg_b3
    
  return nb_rand_kg, nb_add_kg, nb_mult_kg, nb_rand_bit_kg


def complexity_Raccoon_sign(n,nb_r,raccoon_l,raccoon_k,raccoon_n, comp_ref1, comp_ref2, comp_ref3):
  r"""
  Compute the complexity (random, addition) of masked Sign in the Raccoon 
  scheme with our compiler described in section E.1 of our article.

  Args:
      n (int): Number of shares.
      nb_r (int): Number of small uniform random values used per polynomial
      coefficient to generate the Raccoon noise. In the Raccoon parameters,
      this corresponds to `d * rep`.
      raccoon_l (int): Parameter l of the Raccoon scheme.
      raccoon_k (int): Parameter k of the Raccoon scheme.
      raccoon_n (int): Parameter n of the Raccoon scheme.
      comp_ref1 (Callable[[int], tuple[int, int, int]]): Function returning the
      complexity of the refresh gadget used in the first block of Sign for 
      `n` shares. 
      comp_ref2 (Callable[[int], tuple[int, int, int]]): Function returning the
      complexity of the refresh gadget used in the second block of Sign for 
      `n` shares. 
      comp_ref3 (Callable[[int], tuple[int, int, int]]): Function returning the
      complexity of the refresh gadget used in the Gcmatmult block of Sign for 
      `n` shares. 

  Returns:
    The random field complexity, the addition complexity, multiplication
    complexity and the random bits complexity of Sign.
  """    

  # Sign Block 1 : Gsum
  nb_gadd_b1 = (math.ceil(nb_r//n)-1)*raccoon_l*raccoon_n
  nb_rand_add, nb_add_add, nb_rand_bit_add = comp_add(n, comp_ref1)
    
  nb_rand_s_b1_gsum = nb_rand_add * nb_gadd_b1
  nb_add_s_b1_gsum = nb_add_add * nb_gadd_b1
  nb_rand_bit_s_b1_gsum = nb_rand_bit_add * nb_gadd_b1
    
  # Sign Block 1 : Gcopy
  nb_gcopy_b1 = raccoon_l*raccoon_n
  nb_rand_copy, nb_add_copy, nb_rand_bit_copy = comp_copy(n, comp_ref1)
    
  nb_rand_s_b1_gcopy = nb_rand_copy * nb_gcopy_b1
  nb_add_s_b1_gcopy = nb_add_copy * nb_gcopy_b1
  nb_rand_bit_s_b1_gcopy = nb_rand_bit_copy * nb_gcopy_b1

  # Sign Block 1 : Gcpmult
  nb_gcpmult_b1 = raccoon_l
  nb_rand_gcpmult, nb_add_gcpmult, nb_mult_gcpmult, nb_rand_bit_gcpmult = complexity_Gcpmult(n, raccoon_n, comp_ref1)
    
  nb_rand_s_b1_gcpmult = nb_rand_gcpmult * nb_gcpmult_b1
  nb_add_s_b1_gcpmult = nb_add_gcpmult * nb_gcpmult_b1
  nb_mult_s_b1_gcpmult = nb_mult_gcpmult * nb_gcpmult_b1
  nb_rand_bit_s_b1_gcpmult = nb_rand_bit_gcpmult * nb_gcpmult_b1

  # Sign Block 1 : Gadd
  nb_gadd_b1_2 = raccoon_n*raccoon_l

  nb_rand_s_b1_gadd = nb_rand_add * nb_gadd_b1_2
  nb_add_s_b1_gadd = nb_add_add * nb_gadd_b1_2
  nb_rand_bit_s_b1_gadd = nb_rand_bit_add * nb_gadd_b1_2

  # Sign Block 1 : Gdecode
  nb_gdecode = raccoon_l*raccoon_n
  nb_rand_decode, nb_add_decode, nb_rand_bit_decode = complexity_Gdecode(n, comp_ref1)
    
  nb_rand_s_b1_decode = nb_rand_decode * nb_gdecode
  nb_add_s_b1_decode = nb_add_decode * nb_gdecode
  nb_rand_bit_s_b1_decode = nb_rand_bit_decode * nb_gdecode 

  #Sign Block 1
  nb_rand_s_b1 = nb_rand_s_b1_gsum + nb_rand_s_b1_gcopy + nb_rand_s_b1_gcpmult + nb_rand_s_b1_gadd + nb_rand_s_b1_decode
  nb_add_s_b1 = nb_add_s_b1_gsum + nb_add_s_b1_gcopy + nb_add_s_b1_gcpmult + nb_add_s_b1_gadd + nb_add_s_b1_decode
  nb_mult_s_b1 = nb_mult_s_b1_gcpmult        
  nb_rand_bit_s_b1 = nb_rand_bit_s_b1_gsum + nb_rand_bit_s_b1_gcopy + nb_rand_bit_s_b1_gcpmult + nb_rand_bit_s_b1_gadd + nb_rand_bit_s_b1_decode



  # Sign Block 2 : Gcopies^k
  nb_gcopy_b2 = (raccoon_k-1)*raccoon_l*raccoon_n
  nb_rand_copy, nb_add_copy, nb_rand_bit_copy = comp_copy(n, comp_ref2)
    
  nb_rand_s_b2_gcopy = nb_rand_copy * nb_gcopy_b2
  nb_add_s_b2_gcopy = nb_add_copy * nb_gcopy_b2
  nb_rand_bit_s_b2_gcopy = nb_rand_bit_copy * nb_gcopy_b2

  # Sign Block 2 : Gcpmult^k
  nb_gcpmult = raccoon_k*raccoon_l
  nb_rand_gcpmult, nb_add_gcpmult, nb_mult_gcpmult, nb_rand_bit_gcpmult = complexity_Gcpmult(n, raccoon_n, comp_ref2)
    
  nb_rand_s_b2_gcpmult = nb_rand_gcpmult * nb_gcpmult
  nb_add_s_b2_gcpmult = nb_add_gcpmult * nb_gcpmult
  nb_mult_s_b2_gcpmult = nb_mult_gcpmult * nb_gcpmult
  nb_rand_bit_s_b2_gcpmult = nb_rand_bit_gcpmult * nb_gcpmult

  # Sign Block 2 : Gsum
  nb_gadd_b2 = (raccoon_l-1)*raccoon_k*raccoon_n
  nb_rand_add_b2, nb_add_add_b2, nb_rand_bit_add_b2 = comp_add(n, comp_ref2)

  nb_rand_s_b2_gsum = nb_rand_add_b2 * nb_gadd_b2
  nb_add_s_b2_gsum = nb_add_add_b2 * nb_gadd_b2
  nb_rand_bit_s_b2_gsum = nb_rand_bit_add_b2 * nb_gadd_b2

  #Sign Block 2
  nb_rand_s_b2 = nb_rand_s_b2_gcopy + nb_rand_s_b2_gcpmult + nb_rand_s_b2_gsum
  nb_add_s_b2 = nb_add_s_b2_gcopy +  nb_add_s_b2_gcpmult + nb_add_s_b2_gsum
  nb_mult_s_b2 = nb_mult_s_b2_gcpmult
  nb_rand_bit_s_b2 = nb_rand_bit_s_b2_gcopy + nb_rand_bit_s_b2_gcpmult + nb_rand_bit_s_b2_gsum

  # Sign Block 3 : AddNoiseTo
  nb_gadd_b3 = (math.ceil(nb_r//n))*raccoon_k*raccoon_n
  nb_rand_add_b3, nb_add_add_b3, nb_rand_bit_add_b3 = comp_add(n, comp_ref3)

  nb_rand_s_b3_ANT = nb_rand_add_b3 * nb_gadd_b3
  nb_add_s_b3_ANT = nb_add_add_b3 * nb_gadd_b3
  nb_rand_bit_s_b3_ANT = nb_rand_bit_add_b3 * nb_gadd_b3  

  # Sign Block 3 : Gdecode
  nb_gdecode = raccoon_k*raccoon_n
  nb_rand_decode, nb_add_decode, nb_rand_bit_decode = complexity_Gdecode(n, comp_ref3)
    
  nb_rand_s_b3_decode = nb_rand_decode * nb_gdecode
  nb_add_s_b3_decode = nb_add_decode * nb_gdecode
  nb_rand_bit_s_b3_decode = nb_rand_bit_decode * nb_gdecode  

  #Sign Block 3
  nb_rand_s_b3 = nb_rand_s_b3_ANT + nb_rand_s_b3_decode
  nb_add_s_b3 = nb_add_s_b3_ANT +  nb_add_s_b3_decode
  nb_rand_bit_s_b3 = nb_rand_bit_s_b3_ANT +  nb_rand_bit_s_b3_decode

  # Sign 
  nb_rand_s = nb_rand_s_b1 + nb_rand_s_b2 + nb_rand_s_b3
  nb_add_s = nb_add_s_b1 + nb_add_s_b2 + nb_add_s_b3
  nb_mult_s = nb_mult_s_b1 + nb_mult_s_b2
  nb_rand_bit_s = nb_rand_bit_s_b1 + nb_rand_bit_s_b2 + nb_rand_bit_s_b3
  
  return nb_rand_s, nb_add_s, nb_mult_s, nb_rand_bit_s



