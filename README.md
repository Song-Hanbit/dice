# dice
Dice a PyTorch tensor. :hocho::sweet_potato:

# Dependencies
This script was tested with the version of `python3.8` packages:
```
torch 2.0.0+cu117
numpy 1.23.4
```

# How to use
`"DIR"` is a directory of your dice.py script presents and your current working directory is upper of `"DIR"` (or `"DIR"` is on your home directory:`~`).

`dice` is meant to be a general method for `torch.split` so that the splitting of a tensor at all dimensions ("dicing") may happen at once through this method.

>`input:`
>```python
>from "DIR".dice import dice
>dice?
>```
>
>`printed:`
>```
>Signature:
>dice(
>    tensor: torch.Tensor,
>    length: Union[int, torch.Tensor, list, tuple],
>    dicing_dims: Union[int, torch.Tensor, list, tuple, NoneType] = None,
>    clone: bool = True,
>    return_slice: bool = False,
>) -> tuple
>Docstring: <no docstring>
>File:      ~/booty/dice.py
>Type:      function
>```
>> **Arguments:**
>>
>>`tensor` (*`torch.Tensor`*): Data to be diced.
>>
>>`length` (*`int|torch.Tensor|list|tuple`*): Maximum length(s) a dice may have. If it is an integer or an iterable whose length is one and whose item is an integer, cubic dices whose lengths are `length` for the `dicing_dims` and are lengths of `tensor` for the non-diced dims, a residual cuboid per diced dim and a cuboid whose shape with residual lengths will be returned. (e.g. if a `tensor` has a shape of `(7, 5)` and the input `length` is 3, slicings of `:3`, `3:6` and the residual `6:` will be happened to the `dim=0` and also, `:3` and the residual `3:` will be happened to the `dim=0`.) If it is an iterable of integers and its length is the same as the length of the `dicing_dims`, cuboids whose lengths at the `dicing_dims` are `length` and whose lengths at non-diced dims are the lengths of the input `tensor` and the rests are the same as the case that `length` is an integer.
>>
>>`dicing_dims` (*`int|torch.Tensor|list|tuple|NoneType`*): The dimensions of the `tensor` to be sliced. If it is `None`, all dimensions will be sliced. default:`None`. 
>>
>>`clone` (*`bool`*): whether `tensor` to be cloned (`torch.Tensor.clone()`) so that the return values may not refer to original `tensor` items. default:`True`.
>>
>>`return_slice` (*`bool`*): whether the tuples of `slice`s are to be returned rather than the diced tensors. default:`False`.
>>
>> **Return type:**
>>
>> `tuple[torch.Tensor]` or `tuple[tuple]`
