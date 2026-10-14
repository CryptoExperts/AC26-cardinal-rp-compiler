
[results.py]:results.py
[refresh_env.py]:refresh_env.py
[mult_gen.py]:mult_gen.py
[fromIRA.py]:fromIRA.py
[AES.py]:AES.py
[Raccoon.py]:Raccoon.py

[complexity.py]:complexity.py
[butterfly_network.py]:butterfly_network.py




# AC26 - Cardinal RP Compiler

This repository contains all the scripts necessary to reproduce the results
presented in the paper :

"A Unified Analysis of Refresh Gadgets in the Random Probing Model" by Sonia Belaïd, Victor Normand and Matthieu Rivain, published at Asiacrypt 2026.

Specifically, these scripts reproduce Figures 2, 6a, 6b, 7, and 10, as well as the result stated in the last sentence of Section 5.3.

In the following, we may refer to the full version of this publication, which includes an appendix and minor revisions to the main text. In particular, additional figures have been included, some in the main body and others in the appendix, which is why we refer to figures from the full version rather than those in the published version. The full version is available here:

https://eprint.iacr.org/2026/2015

This project reuses files from other MIT-licensed CryptoExperts repositories.

## Dependencies

This script requires : 
  - Python 3.x
  - `numpy`, `matplotlib`, `scipy`, `sympy`

To install dependencies, run :

`pip install numpy matplotlib scipy sympy`

## Usage

To obtain the graphs used in the paper, please run the following command : 

`python3 results.py`


By default, all figures are generated at the end of the file [results.py]. 
If you want to obtain only some graphs precisely, please comment the adequate 
function, each one is responsible for a figure of the paper, which is 
commented above the function.

```
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
```

If you intend to change any parameter values, check the function’s documentation 
in [results.py] for a better understanding of what each parameter does.


## Organization of the Repository

  1. [results.py] : Contains the **results** that are exhibited in the paper  
    
        "A Unified Analysis of Refresh Gadgets in the Random Probing Model"
        
        Each function is responsible for one or many graphs of the paper and 
    will be highlighted above the function.

  2. [refresh_env.py] :  Contains the core routines used to evaluate the 
  random probing security of the schemes considered in our paper. It computes
  and manipulates general, cardinal, and UO cardinal RPP/RPC envelopes,
  including their compositions and the corresponding threshold RPC security, for
  every refresh gadget considered in the paper. It also provides implementations
  of the different refresh gadgets, together with analytical formulas and
  validation thanks to IronMask. 


  3. [mult_gen.py] : Computes the **uniform variant of the cardinal RPC
     envelopes** of CardSecMult for any number of shares `n >= 2`. CardSecMult 
     is a multiplication gadget introduced in [BelNorRiv25].
     This file is largely inspired by the `mult_gen.py` file available in the
     following GitHub repository:

     https://github.com/CryptoExperts/AC25-cardinal-rp-compiler

     as well as by the full version of [BelNorRiv25], available at:

     https://eprint.iacr.org/2025/1747.pdf 

  4. [fromIRA.py] : Enables the *parsing* of the output produced by the IronMask
     tool using the *cardRPC* and *genRPC* options, which compute the **cardinal
     RPC envelope** and the **general RPC envelope** of a gadget, respectively. 

   
  5. [AES.py] : Compute the **random probing security of our masked AES
     encryption**, build with *our compiler* described in Section 6 of the
     paper. In addition, it enables to **optimize automatically** the $\gamma$ 
     taken (i.e. the number of iteration in RPRefresh) for each block Subbytes,
     AddRoundKey and MixColumns.
     
     This file is largely inspired by the `AES.py` file available on the GitHub
     repository:

     https://github.com/CryptoExperts/AC25-cardinal-rp-compiler

     It implements formulas from [BelNorRiv25].

  6. [Raccoon.py] : Enables the computation of the **threshold random probing
     composability of our masked implementation of Raccoon**, built using the
     compiler described in Appendix E.1 of our paper. In addition, it enables to
     **optimize automatically** the $\gamma$ taken (i.e. the number of iteration 
     in RPRefresh) for each block in *Raccon-Sign* and *Raccoon-Keygen*. 
     This file is primarily based on the file available at: 
  
     https://github.com/CryptoExperts/EC25-random-probing-Raccoon

     with a few minor modifications to suit our purposes. The paper, denoted
     [BelRivRos25] in the documentation, associated with this GitHub repository 
     is available at: 

     https://eprint.iacr.org/2025/278

  7. [complexity.py] : Enables the computation of the **complexity of the
     schemes** evaluated in our article, namely *AES* and *Raccoon*. It computes 
     the **randomness, addition, and multiplication complexities**, as well as
     the **cost of the permutations**. 
     
     The AES complexity computations are largely inspired by the `complexity.py`
     file available in the following GitHub repository:
     
     https://github.com/CryptoExperts/AC25-cardinal-rp-compiler
     
     The Raccoon complexity computations are largely inspired by the code
     available in the following GitHub repository:
     
     https://github.com/CryptoExperts/EC25-random-probing-Raccoon

  8. [butterfly_network.py] : Enables the computation of the *cardinal RPC
     envelopes* for the *1-stage Butterfly Network*. Then it enables to compute 
     the *threshold RPC security* of the 1-stage butterfly network derived from
     the different envelopes computed.

     This file is largely inspired by the "butterfly_network.py" file available
     on the GitHub repository:

     https://github.com/CryptoExperts/AC25-cardinal-rp-compiler

  9. A repository `prec` containing all the precomputations made for the 
  graphs (which can be removed without any issues).

## License

This project is distributed under the **MIT License**.

## AI Acknowledgment

ChatGPT was used to assist with documentation by improving the spelling and
grammar of text originally written by the authors. It was also occasionally used
to generate functions or parts of functions in the [refresh_env.py] file,
particularly those implementing the conversions between cRPP, gRPP, cRPC, and
gRPC, which can be found in the 'Useful Notions' section of the file. All
AI-generated content was reviewed and verified
by the authors. 

##

For any questions, please refer to the **paper** or contact the **authors**.
 


  
  




  
  
  
  
