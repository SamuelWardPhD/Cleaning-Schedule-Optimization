# Cleaning Schedule Optimization

This repository provides the official Python implementation accompanying the following academic paper, which should be cited alongside any use of the code:

    Samuel Ward, Marah-Lisanne Thormann, Julian Wharton and Alain Zemkoho (2026).  
    Data-Driven Hull-Fouling Cleaning Schedule Optimization to Reduce Carbon Footprint of Vessels.





## Structure
The implementation is self-contained within `cleaning_schedule_optimization.py`, which has the following structure:
* `example_problem_instance()`
* `algorithm_1_brute_force_search(n, b0, c, X, B, f)`
* `algorithm_2_dynamic_cleaning_schedule_optimiser(n, b0, c, X, B, f)`
* `main()`

where Algorithms 1 and 2 from the paper correspond to the Python functions of the same name.  The function `main()` provides an example use.


## Installation
You can clone the repository with the following bash commands:

```bash
git clone https://github.com/SamuelWardPhD/Cleaning-Schedule-Optimization
cd Cleaning-Schedule-Optimization
pip install numpy
pip install pandas
```


## Contact
***Samuel Ward***   
📧 <s.ward@soton.ac.uk>


