#!/usr/bin/env python3

################################################################################
# This file generates the figures presenting the security results reported in
# our article. It is intended to be run as the final step of the evaluation
# process. To reproduce the figures from the article, uncomment the main
# part and run this file.
################################################################################

import math
from refresh_env import (env_cRPC_ze_linear, env_UOcRPC_ze_linear, 
  IRA_env_cRPC_ze_linear, IRA_env_cRPC_di_linear,
  env_cRPC_di_linear, env_UOcRPC_di_linear, env_cRPC_ze_nlogn, 
  env_UOcRPC_ze_nlogn, env_cRPC_di_nlogn, env_UOcRPC_di_nlogn, 
  env_cRPC_di_ideal, env_cRPC_ze_ideal, env_cRPC_ze_half, env_UOcRPC_ze_half,  
  env_cRPC_di_half, env_UOcRPC_di_half, env_cRPC_ze_doublehalf, 
  env_UOcRPC_ze_doublehalf, env_cRPC_di_doublehalf, env_UOcRPC_di_doublehalf,
  env_cRPC_ze_RPRefresh, env_cRPC_di_RPRefresh, ana_env_cRPC_half, 
  ana_env_cRPC_ze_doublehalf, ana_env_cRPC_ze_doublehalf_v2,
  IRA_env_cRPC_di_doublehalf, IRA_env_cRPC_di_nlogn, IRA_env_cRPC_ze_doublehalf, 
  IRA_env_cRPC_ze_half, IRA_env_cRPC_ze_nlogn, 
  IRA_env_UOcRPC_di_doublehalf, IRA_env_UOcRPC_di_linear, 
  IRA_env_UOcRPC_di_nlogn, IRA_env_UOcRPC_ze_doublehalf, 
  IRA_env_UOcRPC_ze_half, IRA_env_UOcRPC_ze_linear, IRA_env_UOcRPC_ze_nlogn, 
  env_gRPC_di_ideal, tRPC_from_gRPC, compo_env_gRPC, 
  env_gRPP_ze_ideal, gRPC_from_gRPP, treeAdd_ze, test_precision_cRPC, 
  test_precision_gRPC) 

from Raccoon import rpc_sign, rpc_keygen,  rpc_sign_optimized_RPRefresh, rpc_keygen_optimized_RPRefresh


from complexity import (comp_ref_doublehalf, comp_ref_half, comp_ref_linear, 
                        comp_ref_nlogn, comp_ref_rprefresh, comp_AES_enc, 
                        complexity_Raccoon_KeyGen, complexity_Raccoon_sign)

from butterfly_network import ref_cRPC_butterfly_stage1, new_tRPC_butterfly


from AES import  compute_RPC_AES, compute_RPC_AES_optimized_RPRefresh
from functools import partial

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
from matplotlib.path import Path


def sector_marker(k, n, start_angle=90):
    theta1 = np.deg2rad(start_angle + 360 * k / n)
    theta2 = np.deg2rad(start_angle + 360 * (k + 1) / n)

    theta = np.linspace(theta1, theta2, 40)

    verts = [(0, 0)]
    codes = [Path.MOVETO]

    for t in theta:
        verts.append((np.cos(t), np.sin(t)))
        codes.append(Path.LINETO)

    verts.append((0, 0))
    codes.append(Path.CLOSEPOLY)

    return Path(verts, codes)


def AES_graph(n, p) :
  """
  Compute and plot the random probing security of AES for several refresh 
  gadgets.

  For each refresh gadget, the function evaluates the AES RPC security advantage
  using the available envelope computation methods (atomic composition, 
  analytical formulas, and IronMask). It also optionally optimizes the number of 
  iterations used by RPRefresh and computes the corresponding implementation 
  costs.

  The resulting security advantages are plotted as a function of the refresh
  gadget and saved as a PDF file.

  Args : 
    n (int): Number of shares.
    p (float): Leakage rate.
  """
  logp = int(math.log(p, 2))
  t = n // 2
  cores = 16
  

  nh = n // 2 + (n % 2)
  gamma = 500

  nlogn_gamma = False
  thr_gamma = True

  if (thr_gamma) :
    thr = 0.1


  precomp_file = f"./prec/AES_refresh/data_n{n}_p{logp}.npy"
  if os.path.exists(precomp_file):
    data = np.load(precomp_file, allow_pickle=True).item()

  else :
    data = {
      "Linear": {},
      "nlogn": {},
      "Ideal": {},
      "Half": {},
      "DoubleHalf": {},
      "RPRefresh": {},
    }
  
    #Linear
    print("Linear...")

    compute_pgref01 = env_cRPC_ze_linear
    eps_AES01 = min(compute_RPC_AES(n, p, logp, compute_pgref01, compute_pgref01, compute_pgref01, [compute_pgref01 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["ZE"] = eps_AES01

    compute_pgref02 = env_UOcRPC_ze_linear
    eps_AES02 = min(compute_RPC_AES(n, p, logp, compute_pgref02, compute_pgref02, compute_pgref02, [compute_pgref02 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["UO_ZE"] = eps_AES02

    compute_pgref03 = env_cRPC_di_linear
    eps_AES03 = min(compute_RPC_AES(n, p, logp, compute_pgref03, compute_pgref03, compute_pgref03, [compute_pgref03 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["DI"] = eps_AES03

    compute_pgref04 = env_UOcRPC_di_linear
    eps_AES04 = min(compute_RPC_AES(n, p, logp, compute_pgref04, compute_pgref04, compute_pgref04, [compute_pgref04 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["UO_DI"] = eps_AES04  

    compute_pgref05 = IRA_env_cRPC_di_linear
    eps_AES05 = min(compute_RPC_AES(n, p, logp, compute_pgref05, compute_pgref05, compute_pgref05, [compute_pgref05 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["IRA_DI"] = eps_AES05

    compute_pgref06 = IRA_env_cRPC_ze_linear
    eps_AES06 = min(compute_RPC_AES(n, p, logp, compute_pgref06, compute_pgref06, compute_pgref06, [compute_pgref06 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["IRA_ZE"] = eps_AES06  

    compute_pgref07 = IRA_env_UOcRPC_di_linear
    eps_AES07 = min(compute_RPC_AES(n, p, logp, compute_pgref07, compute_pgref07, compute_pgref07, [compute_pgref07 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["IRA_UO_DI"] = eps_AES07

    compute_pgref08 = IRA_env_UOcRPC_ze_linear
    eps_AES08 = min(compute_RPC_AES(n, p, logp, compute_pgref08, compute_pgref08, compute_pgref08, [compute_pgref08 for _ in range (nh + 1)], t, cores), 1.0)
    data["Linear"]["IRA_UO_ZE"] = eps_AES08  


    #nlogn
    print("Nlogn...")

    compute_pgref11 = env_cRPC_ze_nlogn
    eps_AES11 = min(compute_RPC_AES(n, p, logp, compute_pgref11, compute_pgref11, compute_pgref11, [compute_pgref11 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["ZE"] = eps_AES11

    compute_pgref12 = env_UOcRPC_ze_nlogn
    eps_AES12 = min(compute_RPC_AES(n, p, logp, compute_pgref12, compute_pgref12, compute_pgref12, [compute_pgref12 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["UO_ZE"] = eps_AES12

    compute_pgref13 = env_cRPC_di_nlogn
    eps_AES13 = min(compute_RPC_AES(n, p, logp, compute_pgref13, compute_pgref13, compute_pgref13, [compute_pgref13 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["DI"] = eps_AES13

    compute_pgref14 = env_UOcRPC_di_nlogn
    eps_AES14 = min(compute_RPC_AES(n, p, logp, compute_pgref14, compute_pgref14, compute_pgref14, [compute_pgref14 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["UO_DI"] = eps_AES14

    compute_pgref15 = IRA_env_cRPC_di_nlogn
    eps_AES15 = min(compute_RPC_AES(n, p, logp, compute_pgref15, compute_pgref15, compute_pgref15, [compute_pgref15 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["IRA_DI"] = eps_AES15

    compute_pgref16 = IRA_env_cRPC_ze_nlogn
    eps_AES16 = min(compute_RPC_AES(n, p, logp, compute_pgref16, compute_pgref16, compute_pgref16, [compute_pgref16 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["IRA_ZE"] = eps_AES16

    compute_pgref17 = IRA_env_UOcRPC_di_nlogn
    eps_AES17 = min(compute_RPC_AES(n, p, logp, compute_pgref17, compute_pgref17, compute_pgref17, [compute_pgref17 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["IRA_UO_DI"] = eps_AES17

    compute_pgref18 = IRA_env_UOcRPC_ze_nlogn
    eps_AES18 = min(compute_RPC_AES(n, p, logp, compute_pgref18, compute_pgref18, compute_pgref18, [compute_pgref18 for _ in range (nh + 1)], t, cores), 1.0)
    data["nlogn"]["IRA_UO_ZE"] = eps_AES18    
    

    #Ideal
    print("Ideal...")


    compute_pgref21 = env_cRPC_ze_ideal
    eps_AES21 = compute_RPC_AES(n, p, logp, compute_pgref21, compute_pgref21, compute_pgref21, [compute_pgref21 for _ in range (nh + 1)], t, cores)
    data["Ideal"]["ZE"] = eps_AES21

    compute_pgref22 = env_cRPC_di_ideal
    eps_AES22 = compute_RPC_AES(n, p, logp, compute_pgref22, compute_pgref22, compute_pgref22, [compute_pgref22 for _ in range (nh + 1)], t, cores)
    data["Ideal"]["DI"] = eps_AES22
    
    #Half
    print("Half...")

    compute_pgref31 = env_cRPC_ze_half
    eps_AES31 = min(compute_RPC_AES(n, p, logp, compute_pgref31, compute_pgref31, compute_pgref31, [compute_pgref31 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["ZE"] = eps_AES31

    compute_pgref32 = env_UOcRPC_ze_half
    eps_AES32 = min(compute_RPC_AES(n, p, logp, compute_pgref32, compute_pgref32, compute_pgref32, [compute_pgref32 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["UO_ZE"] = eps_AES32

    compute_pgref33 = env_cRPC_di_half
    eps_AES33 = min(compute_RPC_AES(n, p, logp, compute_pgref33, compute_pgref33, compute_pgref33, [compute_pgref33 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["DI"] = eps_AES33

    compute_pgref34 = env_UOcRPC_di_half
    eps_AES34 = min(compute_RPC_AES(n, p, logp, compute_pgref34, compute_pgref34, compute_pgref34, [compute_pgref34 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["UO_DI"] = eps_AES34

    compute_pgref35 = ana_env_cRPC_half 
    eps_AES35 = min(compute_RPC_AES(n, p, logp, compute_pgref35, compute_pgref35, compute_pgref35, [compute_pgref35 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["ANA_ZE"] = eps_AES35

    compute_pgref36 = IRA_env_cRPC_ze_half 
    eps_AES36 = min(compute_RPC_AES(n, p, logp, compute_pgref36, compute_pgref36, compute_pgref36, [compute_pgref36 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["IRA_ZE"] = eps_AES36

    compute_pgref37 = IRA_env_UOcRPC_ze_half 
    eps_AES37 = min(compute_RPC_AES(n, p, logp, compute_pgref37, compute_pgref37, compute_pgref37, [compute_pgref37 for _ in range (nh + 1)], t, cores), 1.0)
    data["Half"]["IRA_UO_ZE"] = eps_AES37

    #DoubleHalf
    print("DoubleHalf...")

    compute_pgref41 = env_cRPC_ze_doublehalf
    eps_AES41 = min(compute_RPC_AES(n, p, logp, compute_pgref41, compute_pgref41, compute_pgref41, [compute_pgref41 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["ZE"] = eps_AES41

    compute_pgref42 = env_UOcRPC_ze_doublehalf
    eps_AES42 = min(compute_RPC_AES(n, p, logp, compute_pgref42, compute_pgref42, compute_pgref42, [compute_pgref42 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["UO_ZE"] = eps_AES42

    compute_pgref43 = env_cRPC_di_doublehalf
    eps_AES43 = min(compute_RPC_AES(n, p, logp, compute_pgref43, compute_pgref43, compute_pgref43, [compute_pgref43 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["DI"] = eps_AES43

    compute_pgref44 = env_UOcRPC_di_doublehalf
    eps_AES44 = min(compute_RPC_AES(n, p, logp, compute_pgref44, compute_pgref44, compute_pgref44, [compute_pgref44 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["UO_DI"] = eps_AES44

    compute_pgref45 = ana_env_cRPC_ze_doublehalf 
    eps_AES45 = min(compute_RPC_AES(n, p, logp, compute_pgref45, compute_pgref45, compute_pgref45, [compute_pgref45 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["ANA_ZE"] = eps_AES45

    compute_pgref46 = IRA_env_cRPC_di_doublehalf
    eps_AES46 = min(compute_RPC_AES(n, p, logp, compute_pgref46, compute_pgref46, compute_pgref46, [compute_pgref46 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["IRA_DI"] = eps_AES46

    compute_pgref47 = IRA_env_cRPC_ze_doublehalf
    eps_AES47 = min(compute_RPC_AES(n, p, logp, compute_pgref47, compute_pgref47, compute_pgref47, [compute_pgref47 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["IRA_ZE"] = eps_AES47  

    compute_pgref48 = IRA_env_UOcRPC_di_doublehalf
    eps_AES48 = min(compute_RPC_AES(n, p, logp, compute_pgref48, compute_pgref48, compute_pgref48, [compute_pgref48 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["IRA_UO_DI"] = eps_AES48

    compute_pgref49 = IRA_env_UOcRPC_ze_doublehalf
    eps_AES49 = min(compute_RPC_AES(n, p, logp, compute_pgref49, compute_pgref49, compute_pgref49, [compute_pgref49 for _ in range (nh + 1)], t, cores), 1.0)
    data["DoubleHalf"]["IRA_UO_ZE"] = eps_AES49  


    #RPRefresh
    print("RPRefresh...")

    compute_pgref51 = partial(env_cRPC_ze_RPRefresh, nb_iter=gamma) 
    eps_AES51 = min(compute_RPC_AES(n, p, logp, compute_pgref51, compute_pgref51, compute_pgref51, [compute_pgref51 for _ in range (nh + 1)], t, cores), 1.0)
    data["RPRefresh"]["ZE"] = eps_AES51

    compute_pgref52 = partial(env_cRPC_di_RPRefresh, nb_iter=gamma) 
    eps_AES52 = min(compute_RPC_AES(n, p, logp, compute_pgref52, compute_pgref52, compute_pgref52, [compute_pgref52 for _ in range (nh + 1)], t, cores), 1.0)
    data["RPRefresh"]["DI"] = eps_AES52

    os.makedirs(os.path.dirname(precomp_file), exist_ok=True)
    np.save(precomp_file, data)

  refresh_types = list(data.keys())

  colors = {
    "ZE": "#1f77b4",
    "DI": "#1f77b4",

    "UO_ZE": "#1f77b4",
    "UO_DI": "#1f77b4",

    "IRA_ZE": "#d62728",
    "IRA_DI": "#d62728",
    "IRA_UO_ZE": "#d62728",
    "IRA_UO_DI": "#d62728",

    "ANA_ZE": "#2ca02c",
  }

  marker_family = {
    "ZE": "+",
    "ANA_ZE": "+",
    "IRA_ZE": "+",
    "IRA_UO_ZE" : "+",
    "UO_ZE": "+",

    "DI": "o",
    "IRA_DI": "o",
    "IRA_UO_DI" : "o",
    "UO_DI": "o",
  }

  variants = ["ZE", "UO_ZE", "IRA_ZE", "IRA_UO_ZE", "ANA_ZE", "DI", "UO_DI", "IRA_DI", "IRA_UO_DI"]
  if (n & (n - 1)) :
    #n is not a power of 2 : Cannot use analytical formula.
    variants = ["ZE", "UO_ZE", "IRA_ZE", "IRA_UO_ZE", "DI", "UO_DI", "IRA_DI", "IRA_UO_DI"]



  tmp = partial(comp_ref_rprefresh, gamma = gamma)

  r_comp_RPR, _ , _, _ = comp_AES_enc(n, tmp, tmp, tmp, [tmp for _ in range (nh + 1)])
  r_comp_half, _, _, _ = comp_AES_enc(n, comp_ref_half, comp_ref_half, comp_ref_half, [comp_ref_half for _ in range (nh + 1)])
  r_comp_linear, _, _, _ = comp_AES_enc(n, comp_ref_linear, comp_ref_linear, comp_ref_linear, [comp_ref_linear for _ in range (nh + 1)])
  r_comp_ideal = np.inf
  r_comp_doublehalf, _, _, _ = comp_AES_enc(n, comp_ref_doublehalf, comp_ref_doublehalf, comp_ref_doublehalf, [comp_ref_doublehalf for _ in range (nh + 1)])
  r_comp_nlogn, _, _, _ = comp_AES_enc(n, comp_ref_nlogn, comp_ref_nlogn, comp_ref_nlogn, [comp_ref_nlogn for _ in range (nh + 1)])



  if (nlogn_gamma) :
    gamma, _, _ = comp_ref_nlogn(n)
    compute_pgref51 = partial(env_cRPC_ze_RPRefresh, nb_iter = gamma) 
    eps_AES51 = min(compute_RPC_AES(n, p, logp, compute_pgref51, compute_pgref51, compute_pgref51, [compute_pgref51 for _ in range (nh + 1)], t, cores), 1.0)
    data["RPRefresh"]["ZE"] = eps_AES51
    tmp = partial(comp_ref_rprefresh, gamma = gamma)
    r_comp_RPR, _ , _, _ = comp_AES_enc(n, tmp, tmp, tmp, [tmp for _ in range (nh + 1)])
  
  if (thr_gamma) :
    #I use the ZE version to optimize the gamma, may be non optimized for the
    #direct version.
    
    filename_optimize = f"./prec/opti_gamma/AES_opti_gamma_{n}_p{logp}.npy"
    if(os.path.exists(filename_optimize)):
      eps_AES51, eps_AES52, gamma_ark, gamma_mc, gamma_sb, l_gamma_sb = np.load(filename_optimize, allow_pickle=True)

    else :  
      eps_AES51, gamma_ark, gamma_mc, gamma_sb, l_gamma_sb = (
      compute_RPC_AES_optimized_RPRefresh(n, p, thr, cores, env_cRPC_ze_RPRefresh)
      )
      
      compute_pgref52_ark = partial(env_cRPC_di_RPRefresh, nb_iter=gamma_ark) 
      compute_pgref52_mc = partial(env_cRPC_di_RPRefresh, nb_iter=gamma_mc) 
      compute_pgref52_sb = partial(env_cRPC_di_RPRefresh, nb_iter=gamma_sb) 
      eps_AES52 = min(compute_RPC_AES(n, p, logp, compute_pgref52_ark, 
      compute_pgref52_mc, compute_pgref52_sb, 
      [partial(env_cRPC_di_RPRefresh, nb_iter=l_gamma_sb[i]) for i in range (nh + 1)], t, cores), 1.0)
      
      to_save = np.array(
        [eps_AES51, eps_AES52, gamma_ark, gamma_mc, gamma_sb, l_gamma_sb],
        dtype=object
      )
      os.makedirs(os.path.dirname(filename_optimize), exist_ok=True)
      np.save(filename_optimize, to_save)

    data["RPRefresh"]["ZE"] = eps_AES51
    data["RPRefresh"]["DI"] = eps_AES52

    tmp_ark = partial(comp_ref_rprefresh, gamma = gamma_ark)
    tmp_mc = partial(comp_ref_rprefresh, gamma = gamma_mc)
    tmp_sb = partial(comp_ref_rprefresh, gamma = gamma_sb)
    r_comp_RPR, _, _, _ = comp_AES_enc(n, tmp_ark, tmp_mc, tmp_sb, [ partial(comp_ref_rprefresh, gamma = l_gamma_sb[i]) for i in range (nh + 1)])
  
  print("gamma_ark = ", gamma_ark)
  print("gamma_mc = ", gamma_mc)
  print("gamma_sb = ", gamma_sb)
  print("gamma_mult = ", l_gamma_sb)

  refresh_costs = {
    "Linear": r_comp_linear,
    "nlogn": r_comp_nlogn,
    "Ideal": r_comp_ideal,
    "Half": r_comp_half,
    "DoubleHalf": r_comp_doublehalf,
    "RPRefresh": r_comp_RPR,
  }

  refresh_types = sorted(data.keys(), key=lambda r: refresh_costs[r])

  if (p == 2**-20) :
    i = refresh_types.index("DoubleHalf")
    j = refresh_types.index("RPRefresh")

    refresh_types[i], refresh_types[j] = refresh_types[j], refresh_types[i]

  tol = 1
  fig, ax = plt.subplots(figsize=(7, 5))
  
  
  for idx, refresh in enumerate(refresh_types):
    placed_points = []
    min_dist = tol          # distance verticale sous laquelle on décale
    offset_step = 0.1      # taille du décalage horizontal

    for variant in variants:
        if variant in data[refresh]:
            y = math.log2(data[refresh][variant])

            close_points = [
                poffset
                for py, poffset in placed_points
                if abs(py - y) < min_dist
            ]

            if len(close_points) == 0:
                offset = 0.0
            else:
                used_offsets = set(close_points)
                candidate_offsets = [
                  0.0,
                  -offset_step,
                  offset_step,
                  -2 * offset_step,
                  2 * offset_step,
                  -3 * offset_step,
                  3 * offset_step,
                  -4 * offset_step,
                  4 * offset_step,
                  -5 * offset_step,
                  5 * offset_step,
                ]

                offset = next(
                    off for off in candidate_offsets
                    if off not in used_offsets
                )

            placed_points.append((y, offset))

            marker = marker_family[variant]
            is_permutation = "UO" in variant
            linestyle = (0, (1, 0.5)) if is_permutation else "-"

            if marker == "o":
                ax.scatter(
                    idx + offset,
                    y,
                    s=55,
                    marker="o",
                    facecolors="none",
                    edgecolors=colors[variant],
                    linewidths=1.5,
                    linestyle=linestyle,
                    alpha=0.85,
                    zorder=3,
                )
            else:
                ax.scatter(
                    idx + offset,
                    y,
                    s=55,
                    marker="+",
                    color=colors[variant],
                    linewidths=1.8,
                    linestyle=linestyle,
                    alpha=0.85,
                    zorder=4,
                )


  ax.margins(x=0.1)
  ax.set_xticks(range(len(refresh_types)))
  #ax.set_xticklabels(refresh_types)

  ax.set_xticklabels([
    f"{r}\n{'N/A' if np.isinf(refresh_costs[r]) else f'{refresh_costs[r]:.0f}'}"
    for r in refresh_types
  ], fontsize=13)

  #ax.set_xlabel("Refresh type")
  ax.set_ylabel(r"$\log_2$ security advantage")
  #ax.set_title("RPS security of AES by refresh type")

  ax.grid(True, linestyle="--", alpha=0.4)
  legend_elements = []

  #for variant in variants:
  color_legend = [
      Line2D(
        [0], [0],
        marker='s',
        color='w',
        markerfacecolor="#1f77b4",
        markersize=8,
        label='Atomic Function'
      ),
      Line2D(
        [0], [0],
        marker='s',
        color='w',
        markerfacecolor="#2ca02c",
        markersize=8,
        label='Analytical formula'
      ),
      Line2D(
        [0], [0],
        marker='s',
        color='w',
        markerfacecolor="#d62728",
        markersize=8,
        label='IronMask'
      ),
  ]

  shape_legend = [
    Line2D(
        [0], [0],
        marker='+',
        color='black',
        linestyle='None',
        markersize=8,
        markeredgewidth=1.5,
        label='Zero Encoding'
    ),
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        markerfacecolor='none',
        linestyle='None',
        markersize=8,
        markeredgewidth=1.5,
        label='Direct'
    ),
  ]
  perm_legend = [
    Line2D(
        [0], [0],
        color='black',
        linestyle='-',
        linewidth=2,
        label='Without permutation'
    ),
    Line2D(
        [0], [0],
        color='black',
        linestyle=':',
        linewidth=2,
        label='With permutation'
    ),
  ]

  legend1 = ax.legend(
    handles=color_legend,
    loc='upper right',
    bbox_to_anchor=(0.99, 0.99),
    title='Variant'
  )

  legend2 = ax.legend(
    handles=shape_legend,
    loc='upper right',
    bbox_to_anchor=(0.99, 0.80),
    title='Family'
  )
  
  legend3 = ax.legend(
    handles=perm_legend,
    loc='upper right',
    bbox_to_anchor=(0.99, 0.66),
    title='Permutation'
  )

  ax.add_artist(legend1)
  ax.add_artist(legend2)


  #ax.legend(handles=legend_elements)

  plt.tight_layout()
  plt.savefig(f"AES_refresh_n{n}_p{logp}.pdf")



def compare_raccoon_sign():
  """
  Compare the random probing security of the Raccoon signature for several 
  refresh gadgets and leakage probabilities.

  For each leakage probability, the function computes the RPC advantage and the
  implementation complexity for DoubleHalf, RPRefresh, and ideal refresh gadgets.
  The number of RPRefresh iterations is optimized for the signature circuit.
  Results are cached and then plotted as a function of the refresh complexity.
  """
  n = 16
  t = n // 2
  tri = n - 1
  p_values = [2**-20, 2**-16]
  thr = 0.01

  nb_r = 32
  raccoon_l = 4
  raccoon_k = 5
  raccoon_n = 512

  for i in range (len(p_values)) :

    logp = int(math.log(p_values[i], 2))
    
    filename_prec_raccoon = f"./prec/Raccoon_refresh/Raccoon_Sign_n{n}_p{logp}.npy"
    if(os.path.exists(filename_prec_raccoon)):
      loaded = np.load(filename_prec_raccoon, allow_pickle=True)
      Advantages = loaded[0]
      Complexities = loaded[1]

    else : 

      Advantages = {
        "DoubleHalf" : 1.0,
        "ZE RPRefresh" : 1.0,
        "DI RPRefresh" : 1.0,
        "ZE Ideal" : 1.0,
        "DI Ideal" : 1.0,
      }
      Complexities = {
        "DoubleHalf" : 1.0,
        "ZE RPRefresh" : 1.0,
        "DI RPRefresh" : 1.0,
        "ZE Ideal" : np.inf,
        "DI Ideal" : np.inf,
      }

      pgDoubleHalf_ze = ana_env_cRPC_ze_doublehalf_v2(n, p_values[i])
      res_sign_dh = rpc_sign(n, t, tri, p_values[i], pgDoubleHalf_ze, pgDoubleHalf_ze, pgDoubleHalf_ze)
      Advantages["DoubleHalf"] = res_sign_dh
      Complexities["DoubleHalf"] = complexity_Raccoon_sign(n, nb_r, raccoon_l, raccoon_k, raccoon_n, comp_ref_doublehalf, comp_ref_doublehalf, comp_ref_doublehalf)

      compute_ze_RPRefresh = env_cRPC_ze_RPRefresh
      res_sign_ze_rpr, gamma1, gamma2, gamma3 = rpc_sign_optimized_RPRefresh(n, p_values[i], thr, compute_ze_RPRefresh, True)
      Advantages["ZE RPRefresh"] = res_sign_ze_rpr
      Complexities["ZE RPRefresh"] = complexity_Raccoon_sign(n, nb_r, raccoon_l, raccoon_k, raccoon_n, partial(comp_ref_rprefresh, gamma = gamma1), partial(comp_ref_rprefresh, gamma = gamma2), partial(comp_ref_rprefresh, gamma = gamma3))

      pgZEIdeal = env_cRPC_ze_ideal(n, p_values[i])
      res_sign_ze_id = rpc_sign(n, t, tri, p_values[i], pgZEIdeal, pgZEIdeal, pgZEIdeal)
      Advantages["ZE Ideal"] = res_sign_ze_id

  
      compute_di_RPRefresh = env_cRPC_di_RPRefresh
      res_sign_di_rpr, gamma1, gamma2, gamma3 = rpc_sign_optimized_RPRefresh(n, p_values[i], thr, compute_di_RPRefresh, False)
      Advantages["DI RPRefresh"] = res_sign_di_rpr
      Complexities["DI RPRefresh"] = complexity_Raccoon_sign(n, nb_r, raccoon_l, raccoon_k, raccoon_n, partial(comp_ref_rprefresh, gamma = gamma1), partial(comp_ref_rprefresh, gamma = gamma2), partial(comp_ref_rprefresh, gamma = gamma3))

      pgDIdeal = env_cRPC_di_ideal(n, p_values[i])
      res_sign_di_id = rpc_sign(n, t, tri, p_values[i], pgDIdeal, pgDIdeal, pgDIdeal)
      Advantages["DI Ideal"] = res_sign_di_id


      to_save = np.array(
        [Advantages, Complexities],
        dtype=object
      )
      os.makedirs(os.path.dirname(filename_prec_raccoon), exist_ok=True)
      np.save(filename_prec_raccoon, to_save)
    
    filename_pdf = f"Raccoon_Sign_n{n}_p{logp}.pdf"
    Complexities_ref = {
        "DoubleHalf" : Complexities["DoubleHalf"][0],
        "ZE RPRefresh" : Complexities["ZE RPRefresh"][0],
        "DI RPRefresh" : Complexities["DI RPRefresh"][0],
        "ZE Ideal" : np.inf,
        "DI Ideal" : np.inf,
    }
    graph_raccoon(Advantages, Complexities_ref, p_values[i], filename_pdf)
    

def compare_raccoon_keygen():
  """
  Compare the random probing security of the Raccoon key generation for several
  refresh gadgets and leakage probabilities.

  For each leakage probability, the function computes the RPC advantage and the
  implementation complexity for DoubleHalf, RPRefresh, and ideal refresh 
  gadgets. The number of RPRefresh iterations is optimized for the key
  generation circuit. Results are cached and then plotted as a function of the 
  refresh complexity.
  """
  n = 16
  t = n // 2
  tri = n - 1
  p_values = [2**-20, 2**-16]
  thr = 0.01

  nb_r = 32
  raccoon_l = 4
  raccoon_k = 5
  raccoon_n = 512

  for i in range (len(p_values)) :
    
    logp = int(math.log(p_values[i], 2))
    
    filename_prec_raccoon = f"./prec/Raccoon_refresh/Raccoon_KeyGen_n{n}_p{logp}.npy"
    if(os.path.exists(filename_prec_raccoon)):
      loaded = np.load(filename_prec_raccoon, allow_pickle=True)
      Advantages = loaded[0]
      Complexities = loaded[1]
    else : 
      Advantages = {
        "DoubleHalf" : 1.0,
        "ZE RPRefresh" : 1.0,
        "DI RPRefresh" : 1.0,
        "ZE Ideal" : 1.0,
        "DI Ideal" : 1.0,
      }
      Complexities = {
        "DoubleHalf" : 1.0,
        "ZE RPRefresh" : 1.0,
        "DI RPRefresh" : 1.0,
        "ZE Ideal" : np.inf,
        "DI Ideal" : np.inf,
      }
    
      pgDoubleHalf_ze = ana_env_cRPC_ze_doublehalf_v2(n, p_values[i])
      res_keygen_dh = rpc_keygen(n,t,tri,p_values[i],pgDoubleHalf_ze, pgDoubleHalf_ze, pgDoubleHalf_ze)
      Advantages["DoubleHalf"] = res_keygen_dh
      Complexities["DoubleHalf"] = complexity_Raccoon_KeyGen(n, nb_r, raccoon_l, raccoon_k, raccoon_n, comp_ref_doublehalf, comp_ref_doublehalf, comp_ref_doublehalf)

      compute_ze_RPRefresh = env_cRPC_ze_RPRefresh
      res_keygen_ze_rpr, gamma1, gamma2, gamma3 = rpc_keygen_optimized_RPRefresh(n, p_values[i], thr, compute_ze_RPRefresh, True)
      Advantages["ZE RPRefresh"] = res_keygen_ze_rpr
      Complexities["ZE RPRefresh"] = complexity_Raccoon_KeyGen(n, nb_r, raccoon_l, raccoon_k, raccoon_n, partial(comp_ref_rprefresh, gamma = gamma1), partial(comp_ref_rprefresh, gamma = gamma2), partial(comp_ref_rprefresh, gamma = gamma3))

      pgZEIdeal = env_cRPC_ze_ideal(n, p_values[i])
      res_keygen_ze_id = rpc_keygen(n,t,tri,p_values[i], pgZEIdeal, pgZEIdeal, pgZEIdeal)
      Advantages["ZE Ideal"] = res_keygen_ze_id

      compute_di_RPRefresh = env_cRPC_di_RPRefresh
      res_keygen_di_rpr, gamma1, gamma2, gamma3 = rpc_keygen_optimized_RPRefresh(n, p_values[i], thr, compute_di_RPRefresh, False)
      Advantages["DI RPRefresh"] = res_keygen_di_rpr
      Complexities["DI RPRefresh"] = complexity_Raccoon_KeyGen(n, nb_r, raccoon_l, raccoon_k, raccoon_n, partial(comp_ref_rprefresh, gamma = gamma1), partial(comp_ref_rprefresh, gamma = gamma2), partial(comp_ref_rprefresh, gamma = gamma3))
    
      pgDIdeal = env_cRPC_di_ideal(n, p_values[i])
      res_keygen_di_id = rpc_keygen(n,t,tri,p_values[i],pgDIdeal, pgDIdeal, pgDIdeal)
      Advantages["DI Ideal"] = res_keygen_di_id

      to_save = np.array(
        [Advantages, Complexities],
        dtype=object
      )
      os.makedirs(os.path.dirname(filename_prec_raccoon), exist_ok=True)
      np.save(filename_prec_raccoon, to_save)

    filename_pdf = f"Raccoon_KeyGen_n{n}_p{logp}.pdf"
    Complexities_ref = {
        "DoubleHalf" : Complexities["DoubleHalf"][0],
        "ZE RPRefresh" : Complexities["ZE RPRefresh"][0],
        "DI RPRefresh" : Complexities["DI RPRefresh"][0],
        "ZE Ideal" : np.inf,
        "DI Ideal" : np.inf,
    }
    graph_raccoon(Advantages, Complexities_ref, p_values[i], filename_pdf)



def graph_raccoon(Advantages, Complexities, p, filename):
  """
  Plot the random probing security advantages of the considered Raccoon refresh
  gadgets as a function of their implementation complexity.

  The refresh gadgets are ordered by increasing complexity. Zero Encoding and
  Direct variants are distinguished using different markers, while unavailable
  complexities are displayed as N/A.

  Args:
  Advantages (dict[str, float]): RPC security advantage of each refresh gadget.
  Complexities (dict[str, float]): Implementation complexity of each refresh
  gadget.
  p (float): Leakage probability.
  filename (str): Name of the output PDF file.
  """
  logp = int(math.log(p, 2))

  colors = {
        "DoubleHalf": "#1f77b4",
        "ZE RPRefresh": "#1f77b4",
        "DI RPRefresh": "#1f77b4",
        "ZE Ideal": "#1f77b4",
        "DI Ideal": "#1f77b4",
  }

  markers = {
        "DoubleHalf": "+",
        "ZE RPRefresh": "+",
        "DI RPRefresh": "o",
        "ZE Ideal": "+",
        "DI Ideal": "o",
  }

  # sort by complexity
  labels = sorted(
        Advantages.keys(),
        key=lambda x: Complexities[x]
  )

  fig, ax = plt.subplots(figsize=(7, 5))

  for idx, label in enumerate(labels):

      advantage = Advantages[label]
      complexity = Complexities[label]

      y = math.log2(advantage)
      if (markers[label] == "o"):
        ax.scatter(
          idx,
          y,
          s=55,
          facecolors="none",
          color=colors.get(label, "black"),
          marker=markers.get(label, "x"),
          linewidths=1.5,
          alpha=0.85,
          zorder=3,
        )
      else :
        ax.scatter(
          idx,
          y,
          s=55,
          color=colors.get(label, "black"),
          marker=markers.get(label, "x"),
          linewidths=1.5,
          alpha=0.85,
          zorder=3,
        )


  ax.set_xticks(range(len(labels)))

  ax.set_xticklabels([
        f"{label}\n{Complexities[label]:.0f}"
        if np.isfinite(Complexities[label])
        else f"{label}\nN/A"
        for label in labels
    ], fontsize=13)

  #ax.set_xlabel("Refresh type\n(complexity below)")
  ax.set_ylabel(r"$\log_2$ security advantage")

  ax.grid(True, linestyle="--", alpha=0.4)
  plt.tight_layout()
  plt.savefig(filename)

def compare_AES_enc() :
  """
  Compare the random probing security of the AES encryption for several refresh
  gadgets and leakage probabilities through their t-threshold RPC advantage.

  For each leakage probability, the function computes the t-threshold RPC
  advantage and the implementation complexity for DoubleHalf, RPRefresh, and ideal
  refresh gadgets. The number of RPRefresh iterations is optimized independently
  for the Zero Encoding and Direct variants and for the different AES building
  blocks. Results are cached and then plotted as a function of the refresh
  complexity.
  """

  n = 16
  t = n // 2
  nh = n // 2 + (n % 2)
  p_values = [2**-16, 2**-12]
  cores = 192
  thr = 0.1

  for i in range (len(p_values)) :
    p = p_values[i]
    logp = int(math.log(p, 2))

    filename_prec = f"./prec/AES_refresh/AES_n{n}_p{logp}.npy"
    if(os.path.exists(filename_prec)):
      loaded = np.load(filename_prec, allow_pickle=True)
      Advantages = loaded[0]
      Complexities = loaded[1]
    
    else :
      Advantages = {
        "DoubleHalf" : 1.0,
        "ZE RPRefresh" : 1.0,
        "DI RPRefresh" : 1.0,
        "ZE Ideal" : 1.0,
        "DI Ideal" : 1.0,
      }
    
      Complexities = {
        "DoubleHalf" : 1.0,
        "ZE RPRefresh" : 1.0,
        "DI RPRefresh" : 1.0,
        "ZE Ideal" : np.inf,
        "DI Ideal" : np.inf,
      }

      pgDoubleHalf_ze = ana_env_cRPC_ze_doublehalf_v2
      res_AES_dh = min(compute_RPC_AES(n, p, logp, pgDoubleHalf_ze, pgDoubleHalf_ze, pgDoubleHalf_ze, [pgDoubleHalf_ze for _ in range (nh + 1)], t, cores), 1.0)
      Advantages["DoubleHalf"] = res_AES_dh
      Complexities["DoubleHalf"] = comp_AES_enc(n, comp_ref_doublehalf, comp_ref_doublehalf, comp_ref_doublehalf, [ comp_ref_doublehalf for i in range (nh + 1)])    
    
      filename_optimize_ze = f"./prec/opti_gamma/AES_ze_opti_gamma_{n}_p{logp}.npy"
      if(os.path.exists(filename_optimize_ze)):
        loaded = np.load(filename_optimize_ze, allow_pickle=True)
        res_AES_ze = loaded[0]
        gamma_ark_ze = loaded[1]
        gamma_mc_ze = loaded[2]
        gamma_sb_ze = loaded[3]
        l_gamma_sb_ze = loaded[4] 
      
      else : 
        res_AES_ze, gamma_ark_ze, gamma_mc_ze, gamma_sb_ze, l_gamma_sb_ze = (compute_RPC_AES_optimized_RPRefresh(n, p, thr, cores, env_cRPC_ze_RPRefresh))
        to_save = np.array(
          [res_AES_ze, gamma_ark_ze, gamma_mc_ze, gamma_sb_ze, l_gamma_sb_ze],
          dtype=object
        )
        os.makedirs(os.path.dirname(filename_optimize_ze), exist_ok=True)
        np.save(filename_optimize_ze, to_save)
      Advantages["ZE RPRefresh"] = res_AES_ze
      tmp_ark = partial(comp_ref_rprefresh, gamma = gamma_ark_ze)
      tmp_mc = partial(comp_ref_rprefresh, gamma = gamma_mc_ze)
      tmp_sb = partial(comp_ref_rprefresh, gamma = gamma_sb_ze)
      Complexities["ZE RPRefresh"] = comp_AES_enc(n, tmp_ark, tmp_mc, tmp_sb, [ partial(comp_ref_rprefresh, gamma = l_gamma_sb_ze[i]) for i in range (nh + 1)])

      pgZEIdeal = env_cRPC_ze_ideal
      res_AES_ze_id = min(compute_RPC_AES(n, p, logp, pgZEIdeal, pgZEIdeal, pgZEIdeal, [pgZEIdeal for _ in range (nh + 1)], t, cores), 1.0)
      Advantages["ZE Ideal"]  = res_AES_ze_id   

      filename_optimize_di = f"./prec/opti_gamma/AES_di_opti_gamma_{n}_p{logp}.npy"
      if(os.path.exists(filename_optimize_di)):
        loaded = np.load(filename_optimize_di, allow_pickle=True)
        res_AES_di = loaded[0]
        gamma_ark_di = loaded[1]
        gamma_mc_di = loaded[2]
        gamma_sb_di = loaded[3]
        l_gamma_sb_di = loaded[4]
      else :  
        res_AES_di, gamma_ark_di, gamma_mc_di, gamma_sb_di, l_gamma_sb_di = (compute_RPC_AES_optimized_RPRefresh(n, p, thr, cores, env_cRPC_di_RPRefresh))
        to_save = np.array(
          [res_AES_di, gamma_ark_di, gamma_mc_di, gamma_sb_di, l_gamma_sb_di],
          dtype=object
        )
        os.makedirs(os.path.dirname(filename_optimize_di), exist_ok=True)
        np.save(filename_optimize_di, to_save)
      Advantages["DI RPRefresh"] = res_AES_di
      tmp_ark = partial(comp_ref_rprefresh, gamma = gamma_ark_di)
      tmp_mc = partial(comp_ref_rprefresh, gamma = gamma_mc_di)
      tmp_sb = partial(comp_ref_rprefresh, gamma = gamma_sb_di)
      Complexities["DI RPRefresh"] = comp_AES_enc(n, tmp_ark, tmp_mc, tmp_sb, [ partial(comp_ref_rprefresh, gamma = l_gamma_sb_di[i]) for i in range (nh + 1)])

    
      pgDIdeal = env_cRPC_di_ideal
      res_AES_di_id = min(compute_RPC_AES(n, p, logp, pgDIdeal, pgDIdeal, pgDIdeal, [pgDIdeal for _ in range (nh + 1)], t, cores), 1.0)
      Advantages["DI Ideal"]  = res_AES_di_id

      to_save = np.array(
        [Advantages, Complexities],
        dtype=object
      )
      os.makedirs(os.path.dirname(filename_prec), exist_ok=True)
      np.save(filename_prec, to_save)


    filename_pdf = f"AES_n{n}_p{logp}.pdf"
    New_Advantages = {
      "ZE RPRefresh" : Advantages["ZE RPRefresh"],
      "DI RPRefresh" : Advantages["DI RPRefresh"],
      "ZE Ideal" : Advantages["ZE Ideal"],
      "DI Ideal" : Advantages["DI Ideal"],
    }
    Complexities_ref = {
      #  "DoubleHalf" : Complexities["DoubleHalf"][0],
        "ZE RPRefresh" : Complexities["ZE RPRefresh"][0],
        "DI RPRefresh" : Complexities["DI RPRefresh"][0],
        "ZE Ideal" : np.inf,
        "DI Ideal" : np.inf,
    }
    graph_raccoon(New_Advantages, Complexities_ref, p_values[i], filename_pdf)


def are_close(ax, x, y1, y2, threshold_px=12):
    _, py1 = ax.transData.transform((x, y1))
    _, py2 = ax.transData.transform((x, y2))

    return abs(py1 - py2) < threshold_px

def compare_refresh_st1() :
  """
  Compare the random probing security of the first stage of the Butterfly refresh
  for several numbers of shares and leakage probabilities.

  For each leakage probability and number of shares, the function computes the
  t-threshold RPC advantage obtained with the Ideal Direct, Ideal Zero Encoding,
  and Zero Encoding RPRefresh gadgets. Close points are horizontally shifted for
  readability, and the resulting comparison is saved as a PDF file.
  """
  
  compute_env = ref_cRPC_butterfly_stage1
  compute_eps = new_tRPC_butterfly

  filename = "IdealvsEuro_BTst1_RPC.pdf"
  
  n_values = [i for i in range (2, 9)]
  colors = [
    "#1B5E20",
    "#E65100", 
    "#0D47A1", 
  ]

  markers = {
    "ideal_direct": "o",
    "ideal_ze": "^",
    "rprefresh_ze": "x",
  }
  p_values = [2**-5, 2**-10, 2**-15]

  threshold = 1e-14 
  offset = 0.08 
  
  gamma = 500
  idx = 0
  for p in p_values :

    plt.yscale("log", base=2)
    ax = plt.gca()


    eps_id = []
    eps_id_ze = []
    eps_eu = []

    x_id = []
    x_id_ze = []
    x_eu = []


    for n in n_values :
      ideal_refresh = env_cRPC_di_ideal(n, p)
      ideal_ze_refresh = env_cRPC_ze_ideal(n, p)
      pgref = env_cRPC_ze_RPRefresh(n, p, gamma)

      env_butterfly_ideal = compute_env(n, p, ideal_refresh)
      env_butterfly_ze_ideal = compute_env (n, p, ideal_ze_refresh)
      env_butterfly_pgref = compute_env(n, p, pgref)

      eps_ideal = compute_eps(n, env_butterfly_ideal)
      eps_ideal_ze = compute_eps(n, env_butterfly_ze_ideal)
      eps_euro = compute_eps(n, env_butterfly_pgref)

      eps_id.append(eps_ideal)
      eps_id_ze.append(eps_ideal_ze)
      eps_eu.append(eps_euro)

      x = [n, n, n]

      close_01 = are_close(ax, n, eps_ideal, eps_ideal_ze) 
      close_02 = are_close(ax, n, eps_ideal, eps_euro)
      close_12 = are_close(ax, n, eps_ideal_ze, eps_euro)

      # Les trois points sont proches
      if close_01 and close_02 and close_12:
        x[0] = n - offset
        x[1] = n
        x[2] = n + offset

      # Seulement 0 et 1
      elif close_01:
        x[0] = n - offset / 2
        x[1] = n + offset / 2

      # Seulement 0 et 2
      elif close_02:
        x[0] = n - offset / 2
        x[2] = n + offset / 2

      # Seulement 1 et 2
      elif close_12:
        x[1] = n - offset / 2
        x[2] = n + offset / 2

      x_id.append(x[0])
      x_id_ze.append(x[1])
      x_eu.append(x[2])

    
    plt.plot(x_id, eps_id, marker=markers["ideal_direct"], 
             markerfacecolor='none', markersize=5, linestyle="", color=colors[idx])

    plt.plot(x_id_ze, eps_id_ze, marker = markers["ideal_ze"], 
             markerfacecolor='none', markersize=5, linestyle="", 
             color = colors[idx])

    plt.plot (x_eu, eps_eu, marker = markers["rprefresh_ze"], 
              markerfacecolor='none', markersize=5, linestyle="", 
              color = colors[idx])
    idx += 1

  color_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        color=colors[i],
        markerfacecolor=colors[i],
        markersize=5,
        label=rf"$p = 2^{{{int(math.log2(p))}}}$"
    )
    for i, p in enumerate(p_values)
  ]

  legend_colors = ax.legend(
    handles=color_handles,
    title="Leakage rate",
    loc="lower left",
    bbox_to_anchor=(0.0, 0.0)
  )

  ax.add_artist(legend_colors)

  gadget_handles = [
    Line2D(
        [0], [0],
        marker='o',
        linestyle='',
        color='black',
        markerfacecolor='none',
        markersize=5,
        label="Ideal Direct"
    ),
    Line2D(
        [0], [0],
        marker='^',
        linestyle='',
        color='black',
        markerfacecolor='none',
        markersize=5,
        label="Ideal ZE"
    ),
    Line2D(
        [0], [0],
        marker='x',
        linestyle='',
        color='black',
        markersize=5,
        label="RPRefresh ZE"
    ),
  ]

  ax.legend(
    handles=gadget_handles,
    title="Refresh gadget",
    loc="lower left",
    bbox_to_anchor=(0.0, 0.22)
  )

  plt.xlabel("Number of shares n")
  plt.ylabel("Security Bound")
  plt.grid(True)
  plt.savefig(filename)




def composition_genRPC_ideal_plot(n_values, p, nb_compo):
  colors = ["red", "blue", "green", "orange", "purple", "cyan", "magenta", 
            "yellow", "black"]

  fig, ax = plt.subplots(figsize=(12, 8))
  logp = int(math.log(p, 2))
  for n in n_values :
    random_added_by_index = [2 for _ in range(n + 1)]
    pgref_direct_ideal = env_gRPC_di_ideal(n, p)
    gRPP_ze_ideal = env_gRPP_ze_ideal (n, p)
    pgref_ze_ideal = gRPC_from_gRPP(n, p, gRPP_ze_ideal)

    x_values = [0]
    
    y_direct_values = [tRPC_from_gRPC(n, pgref_direct_ideal)]
    y_ze_values = [tRPC_from_gRPC(n, pgref_ze_ideal)]

    for l in range (1, nb_compo) :
      pgref_direct_ideal = compo_env_gRPC(n, pgref_direct_ideal, pgref_direct_ideal) 
      gRPP_ze_ideal, _ = treeAdd_ze(n, p, gRPP_ze_ideal, random_added_by_index, 
                                 gRPP_ze_ideal, random_added_by_index) 
      pgref_ze_ideal = gRPC_from_gRPP(n, p, gRPP_ze_ideal)
      x_values.append((1 << l) - 1)
      y_direct_values.append(tRPC_from_gRPC(n, pgref_direct_ideal))
      y_ze_values.append(tRPC_from_gRPC(n, pgref_ze_ideal))

    ax.plot(x_values, y_direct_values, linestyle ="-", color = colors[n])
    ax.plot(x_values, y_ze_values, linestyle ="--", color = colors[n])

  color_handles = [
    Line2D([0], [0], color=colors[i], lw=2, label=f"n = {i}")
    for i in range(2, 2 + len(n_values))
  ]

  # légende pour les styles
  style_handles = [
    Line2D([0], [0], color="black", lw=2, linestyle="-", label="Direct"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Zero Encoding"),
  ]

  legend1 = ax.legend(handles=color_handles, title="Number of shares (n)", loc="upper right")
  ax.add_artist(legend1)
  ax.legend(handles=style_handles, title="Method", loc="lower left")
  
  ax.set_xlabel("Number of compositions")
  ax.set_ylabel("Threshold RPC security advantage")
  #ax.legend()
  ax.set_yscale("log", base=2)
  ax.grid(True)

  fig.savefig(f"Convergence_ideal_refresh_p{logp}.pdf")
  plt.close(fig)
      

if __name__ == "__main__" :
  #Responsible of Figure 2 of the full version of the article.
  compare_refresh_st1()

  #Responsible of the last sentence of section 5.3.
  test_precision_gRPC(8, 2**-16)
  test_precision_cRPC(8, 2**-16)


  #Responsible of Figure 6a. of the full version of the article.
  AES_graph(8, 2**-16)

  #Responsible of Figure 6b. of the full version of the article.
  AES_graph(3, 2**-16)

  #Responsible of Figure 7 of the full version of the article.
  composition_genRPC_ideal_plot([2, 3, 4, 5, 6], 2**-5, 16)

  #Responsible of Figure 10 of the full version of the article.
  compare_raccoon_sign()















