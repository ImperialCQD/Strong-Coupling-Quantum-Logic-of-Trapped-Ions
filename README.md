# Strong-coupling quantum logic of trapped ions

_by Mahdi Sameti, [Jake Lishman](https://www.github.com/jakelishman) and Florian
Mintert._

This repository accompanies the paper "Strong-coupling quantum logic of trapped
ions", available on [arXiv][arxiv] and in [to be inserted].  The files here
provide the conditions enumerated in the supplementary material in
computer-readable JSON form.

[arxiv]: https://arxiv.org/abs/2003.11718

The two files in the directory are

- `order_3_sidebands_2.json`: the operators and their coefficients up to and
  including terms in eta^3.  The driving field per sideband is simply `f_k(t) =
  f_{k,0}(t)`, _i.e._ standard driving with `f_{k,0}` not dependent on eta.
- `order_4_sidebands_3_extra.json`: the operators and their coefficients up to
  and including terms in eta^4.  The driving field per sideband is now `f_k(t) =
  \sum_{h=0}^4 \eta^h f_{k,h}(t)`, where the only eta-dependence allowed is the
  explicit power series in the sum.

The JSON schema in each file is as follows in a modified BNF, where `[x]` means
an array where each element is of type `x`, `{ ... }` is a literal
JSON object, and the string literals in JSON object specifications will always
exist as labels in every occurrence of an object of that type.
```EBNF
root = [transformation] ;
transformation = {
    "eta": integer,
    "terms": [term_sum_element],
} ;
term_sum_element = {
    "op": operator,
    "scalar": [scalar_sum_element],
} ;
operator = {
    "Sy": 1 | 2,
    "create": integer,
    "destroy": integer,
} ;
scalar_sum_element = {
    "imaginary": 0 | 1,
    "fraction": [integer],
    "function": string,
} ;
```

The array elements `"terms"` in `transformation` and `"scalar"` in
`term_sum_element` are to be interpreted as sums of all their elements.  The
array `"fraction"` in `scalar_sum_element` is always two integers, the first is
the numerator, and the second is the denominator.

The object `term_sum_element` is a multiplication of its two parts, as is the
object `scalar_sum_element` of its three parts.

The example `operator` representation `{"Sy": n, "create": p, "destroy": q}`
represents the term `S_y^n a^{\dagger p} a^q`, with the creation and
annihilation operators always in that order.

The example `scalar_sum_element` `{"imaginary": 1, "fraction": [-3, 4],
"function": f}` represents the quantity `i^1 * (-3/4) * f(t)`.

The `"function"` element is a string literal with a root element `function` in
the following BNF grammar.
```EBNF
function       = integral | multiplication | conjugation | base ;
base           = "Base(" identifier ")" ;
conjugation    = "Conj(" function ")" ;
integral       = "Integral(" function ")" ;
multiplication = "Mult(" "(" function [(", " function) ...] ")" ")" ;
identifier     = "(" integer ", " integer ")" ;
```
The `function` `Base((k, h))` represents the form `f_{k,h}(t)`, which appears in
the full kth sideband driving function `f_k = \sum_h \eta^h f_{k,h}(t)`. `Conj`
represents complex conjugate of the enclosed `function`, `Mult` is
multiplication of the arguments, and `Integral` is the same as the `{}` notation
in the paper, i.e.
```latex
    {x} = \int_0^t x(t_1) dt_1
    {x {y}} = \int_0^t dt_1 \int_0^{t_1} dt_2 x(t_1) y(t_2)
    ...
```
