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

## math0


\begin{align}
    ||E_j A E_i ||_2^2 &= \text{trace}\left(E_i A^T E_j E_j A E_i\right) = \text{trace}\left(E_i A^T E_j A E_i\right)\\
    &= 
\end{align}
more:
\begin{align}
     \tilde{J} &= \sum_{k,n}\left(Y_{kn} - \sum_j \delta(\left(k,j\right)=\left(p,q\right))A_{kj}X_{jn}\right)^2\\
\end{align}


## Math
\begin{align}
    J &= \sum_{k,n}\left(Y_{kn} - \sum_j A_{kj}X_{jn}\right)^2\\
    \frac{\partial J}{\partial A_{k'j'}} &= \sum_{k,n}2\left(Y_{kn} - \sum_j A_{kj}X_{jn}\right)\left(- \delta(k=k') \cdot X_{j'n}\right)\\
    &= -2\sum_{n}\left(Y_{k'n} - \sum_j A_{k'j}X_{jn}\right)X_{j'n}\\
    &= -2\sum_n Y_{k'n} X_{j'n} + 2\sum_j A_{k'j} \sum_n X_{jn} X_{j'n}\\
    &= -2\left<y_{k'}, x_{j'}\right> + 2\sum_j A_{k'j} \left<x_j, x_{j'}\right>\\
    \implies \frac{\partial J}{\partial A_{k'}} &= -2\left<y_{k'}, \mathbf{x}\right> + 2\sum_j A_{k'j} \left<x_j, \mathbf{x}\right>\\
    &= -2\left<y_{k'}, \mathbf{x}\right> + 2 A_{k'} \left<\mathbf{x},\mathbf{x}\right>
\end{align}
Next, suppose $A_{k', \ell'}\equiv 0$. Then for $j'\neq\ell'$ we have:
\begin{align}
    \frac{\partial J}{\partial A_{k' j'}} &= -2\sum_n Y_{k'n} X_{j'n} + 2\sum_{j\neq \ell} A_{k'j} \sum_n X_{jn} X_{j'n}\\
    &= -2\left<y_{k'}, x_{j'}\right> + 2 \sum_{j\neq \ell'} A_{k'j}\left< x_j, x_{j'}\right>
\end{align}
Then we have:
\begin{align}
    \frac{\partial J}{\partial \tilde{A}_{k'}} &= -2\left<y_{k'}, \tilde{\mathbf{x}}\right> + 2 \tilde{A}_{k'}\left<\tilde{\mathbf{x}}, \tilde{\mathbf{x}}\right>
\end{align}