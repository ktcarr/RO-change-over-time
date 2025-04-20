# Notes

## To-dos
### RO validation
- check that RO can reproduce statistics from MPI

### MPI validation
- compute $T$ and $h$ in ORAS5
- plot $SST$ variance in MPI

### Core RO
- modify code so that $\epsilon$ parameter is fixed when fitting

## Questions
- How to generate RO ensemble? E.g.
    - Take mean parameters?
    - For each RO, draw parameters from distribution? For now we fit a different RO model to each MPI ensemble member (so we have an RO ensemble with equal number of members to MPI). To increase RO ensemble size could (i) randomly draw RO member from ensemble or (ii) estimate covariance of parameters, then randomly draw set of parameters.