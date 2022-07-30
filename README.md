---
lang: fr
---

# Solveur d'écoulement de goutte sur pare-brise d'avion, 2D

## Installation

Julia se télécharge et s'installe sur le site https://julialang.org/.
Je conseille fortement l'utilisation de [VSCode](https://code.visualstudio.com/)
combiné avec l'extension [Julia for VSCode](https://www.julia-vscode.org/).

Une fois un REPL Julia lancé dans le dossier projet, activez le projet et
instanciez-le avec

```julia
] activate .
] instantiate
```

ce qui permettra d'installer les dépendances nécessaires.

Les scripts de résolution sont dans le dossier éponymes. Il contient pour le
moment un unique script d'example, d'autres pourront être ajoutés plus tard

## Technologies utilisés

Le code est écrit en Julia. Il tire partie de la librairie
[FLoops.jl] pour paralléliser les boucles de façon efficace. Pour le moment, le
code fonctionne en multithreading, mais FLoops.jl couplés à [FoldsCUDA.jl]
permettra de facilement étendre le code à du GPU.

Le schéma est un schéma semi-discret (utilisant la méthode des lignes,
seul les dérivées spatiales sont discretisé), ce qui permet d'obtenir
un système d'ODE de dimension $n\_1 \times n\_2 \times n\_u$ sous la
forme d'une fonction `f(U, t, parameters...)` va renvoyer un vecteur
d'évolution `dU`.

Ce système d'ODE est ensuite résolu par la librairie [Differential
Equations.jl](https://diffeq.sciml.ai/stable/). Le solveur numérique utilisé est
[SUNDIALS].

Derrière Sundials, une méthode itérative de type GMRES est utilisé avec une
différention numérique de type matrix-free.

Le code étant compatible avec de l'autodiff, il est possible de calculer la
jacobienne automatiquement avec une très bonne précision. Cela donne donc aussi
accès à d'autres solveurs numériques implicites. Il est conseillé d'utiliser
[SparsityTracing.jl] pour obtenir la structure de la jacobienne et s'économiser
le calcul (et le stockage) des termes nuls de la jacobienne.

La sauvegarde des résultats est faite sous le format NetCDF avec un
format standard permettant, entre autre, de le lire en une ligne avec
la librairie xarray en python.

## Schéma numériques

Les schéma utilisés sont ceux décrits dans le papier *Augmented skew- symmetric
system for shallow-water system with surface tension allowing large gradient of
density <sup>[1](#myfootnote1)</sup>*, avec comme seule différence que le schéma
a été écrit sous une forme intégralement implicite sous une forme
semi-discrétisé.

Le schéma spatial est séparé en deux parties

- la partie convective hyperbolique est traité par un schéma volume fini
  MUSCL avec un limiteur minmod
- le reste des termes est discrétisé en différences finis selon le schéma
  décrit par le papier.

## Lecture des résultats

Le plus simple pour la visualisation est d'utiliser [xarray] en Python, mais il
est tout à fait possible de visualiser les résultats en julia en utilisant
[Makie] ou directement [Plots.jl]


## Pour modifier le solveur ?

Si vous avez besoin d'allouer un nouvel array, il faut

- créer le tableau dans la fonction `build_cache` et le placer dans un des deux
  caches
- utiliser @unpack dans les routines `update_hyp_x!`, `update_hyp_y!` ou
  `update_cap!`

Pour modifier la partie hyperbolique et y ajouter d'autres termes, il faut
calculer la matrice A = F'(U) (pour Fx et Fy), en calculer les valeurs propres.
Celles ci devraient se retrouver sous la forme `eigen = c +- a`. Le calcul de F,
a et c (pour les dimensions x et y) se modifient dans `compute_caF_x!` et
`compute_caF_y!`.

Pour modifier les autres termes, deux routines : les coefficients nécessaires
pour les schémas se calculent dans `compute_skew_cap_coeffs!` et les schéma
numériques se calculent dans `skew_cap_kernel!`.

Les opérateurs ont été écrits sous forme de macros permettant de faciliter leur
écriture. De cette façon, `@dx(h)` est transformé en `(h[i + 1, j] - h[i - 1,
j]) / Δx`. De fait, les paramètres i, j, Δx et Δy doivent être présents dans les
arguments de la fonction `skew_cap_kernel!`.

Les opérateurs implémentés sont :

- `@dx(m)`, `@dy(m)`
- `@∇(m)`, `@∇(mx, my)`
- `@div(mx, my)`, `@div(mxx, mxy, myy)`
- `@divh∇(h, mx, my)`

Ce dernier est spécifique au schéma décrit dans le papier.

<a name="myfootnote1">1</a>: Bresch, D., Cellier, N., Couderc, F., Gisclon, M.,
Noble, P., Richard, G.-L., Ruyer-Quil, C., Vila, J.-P., 2020. Augmented
skew-symmetric system for shallow-water system with surface tension allowing
large gradient of density. Journal of Computational Physics 419, 109670..
doi:10.1016/j.jcp.2020.109670

[FLoops.jl]: https://juliafolds.github.io/FLoops.jl/dev/
[FoldsCUDA.jl]: https://github.com/JuliaFolds/FoldsCUDA.jl
[SparsityTracing.jl]: https://github.com/PALEOtoolkit/SparsityTracing.jl
[xarray]: http://xarray.pydata.org
[Makie]: http://makie.juliaplots.org/stable/
[Plots.jl]: http://docs.juliaplots.org/latest/
[SUNDIALS]: https://computing.llnl.gov/projects/sundials