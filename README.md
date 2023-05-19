# dice
Dice a PyTorch tensor. :hocho::sweet_potato:

# Dependencies
This script was tested with the version of `python3.8` packages:
```
torch 2.0.0+cu117
numpy 1.23.4
```

# What is `dice`?
`dice` is meant to be a general method for `torch.split` so that the splitting of a tensor at all dimensions ("dicing") may happen at once through this method.

>`input:`
>```python
>from "DIR".dice import dice
>dice?
>```
> `"DIR"` is a directory of your dice.py script presents and your current working directory is upper of `"DIR"` (or `"DIR"` is on your home directory:`~`).
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
>Docstring:
>dices a tensor. 
>returns a tuple of diced tensors, or a tuple filled with tuples of slice objects.
>File:      ~/"DIR"/dice.py
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

The example at "<ins>https://pytorch.org/docs/stable/generated/torch.split.html</ins>" was reproduced below, however,

*currently, different size by chunk is not supported.*

>`input:`
>```python
>import torch
>a = torch.arange(10).reshape(5, 2)
>torch.split(a, 2)
>```
>
>`output:`
>```
> (tensor([[0, 1],
>          [2, 3]]),
>  tensor([[4, 5],
>          [6, 7]]),
>  tensor([[8, 9]]))
>```
>
>`input:`
>```python
>dice(a, 2, 0)
>```
>
>`output:`
>```
> (tensor([[0, 1],
>          [2, 3]]),
>  tensor([[4, 5],
>          [6, 7]]),
>  tensor([[8, 9]]))
>```


# How to use

This method can be used when making a set of batches of `torch.Tensor` with a defined shape to handle out-of-memory(OOM) issues or to handle specific operations whose input tensors' shape is important.

**Examples:** Given a `data` tensor whose shape is `(4, 4, 4)`:
>`input:`
>```python
>from "DIR".dice import dice
>import torch
>data = torch.arange(64).view(4, 4, 4)
>data.shape
>```
>
>`output:`
>```
>torch.Size([4, 4, 4])
>```

1. Making a set of cubic tensors.
>`input:`
>```python
>dice(data, 2)
>```
>
>`output:`
>```
> (tensor([[[ 0,  1],
>           [ 4,  5]],
>  
>          [[16, 17],
>           [20, 21]]]),
>  tensor([[[ 2,  3],
>           [ 6,  7]],
>  
>          [[18, 19],
>           [22, 23]]]),
>  tensor([[[ 8,  9],
>           [12, 13]],
>  
>          [[24, 25],
>           [28, 29]]]),
>  tensor([[[10, 11],
>           [14, 15]],
>  
>          [[26, 27],
>           [30, 31]]]),
>  tensor([[[32, 33],
>           [36, 37]],
>  
>          [[48, 49],
>           [52, 53]]]),
>  tensor([[[34, 35],
>           [38, 39]],
>  
>          [[50, 51],
>           [54, 55]]]),
>  tensor([[[40, 41],
>           [44, 45]],
>  
>          [[56, 57],
>           [60, 61]]]),
>  tensor([[[42, 43],
>           [46, 47]],
>  
>          [[58, 59],
>           [62, 63]]]))
>```
2. Slicing only `0, 2` dimensions.
>`input:`
>```python
>for dice_ in dice(data, 2, [0, 2]): print(dice_.shape)
>```
>
>`output:`
>```
>torch.Size([2, 4, 2])
>torch.Size([2, 4, 2])
>torch.Size([2, 4, 2])
>torch.Size([2, 4, 2])
>```
3. Obtaining cuboids.
>`input:`
>```python
>for dice_ in dice(data, [2, 3], [1, 2]): print(dice_.shape)
>```
>
>`output:`
>```
> torch.Size([4, 2, 3])
> torch.Size([4, 2, 1])
> torch.Size([4, 2, 3])
> torch.Size([4, 2, 1])
>```
4. Returning `tuple` of `slice` objects. Given a variable of the returned value is `slices` when `return_slice` is `True`, `tuple([tensor[slice_] for slice_ in slices])` is the output of the default option.
>`input:`
>```python
>dice(data, [2, 3], [1, 2], return_slice=True)
>```
>
>`output:`
>```
> ((slice(0, None, None), slice(0, 2, None), slice(0, 3, None)),
>  (slice(0, None, None), slice(0, 2, None), slice(3, None, None)),
>  (slice(0, None, None), slice(2, None, None), slice(0, 3, None)),
>  (slice(0, None, None), slice(2, None, None), slice(3, None, None)))
>```
5. Original data can be modified when `clone=False`.
>`input:`
>```python
>dice(data, 2, clone=False)[3][0, 0, 0] = 99999
>print(data)
>```
>
>`printed:`
>```
> tensor([[[    0,     1,     2,     3],
>          [    4,     5,     6,     7],
>          [    8,     9, 99999,    11],
>          [   12,    13,    14,    15]],
> 
>         [[   16,    17,    18,    19],
>          [   20,    21,    22,    23],
>          [   24,    25,    26,    27],
>          [   28,    29,    30,    31]],
> 
>         [[   32,    33,    34,    35],
>          [   36,    37,    38,    39],
>          [   40,    41,    42,    43],
>          [   44,    45,    46,    47]],
> 
>         [[   48,    49,    50,    51],
>          [   52,    53,    54,    55],
>          [   56,    57,    58,    59],
>          [   60,    61,    62,    63]]])
>```
