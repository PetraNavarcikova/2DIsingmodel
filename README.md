# 2DIsingmodel
Metropolis Monte Carlo simulation of the 2D Ising model as a part of the Computational Physics course at TU Delft. 

# Authors: Petra Navarcikova, Arianna Pasterkamp, Lucio Vecchia


This project investigates the **thermodynamic and phase transition behavior of the 2D Ising model** using **Monte Carlo simulations**.  
Both the **Metropolis** and **Wolff** algorithms were implemented to study magnetization, internal energy, specific heat, and susceptibility under periodic boundary conditions.  
All observables were nondimensionalized, and uncertainties were estimated using **block bootstrapping**.  

To explore real-world effects, **lattice impurities** were introduced by randomly removing lattice spins. Results show that increasing impurity concentration **lowers the critical temperature** and weakens long-range magnetic order.  The **Wolff algorithm** proved more efficient near phase transitions, confirming its advantage in capturing critical phenomena.  

