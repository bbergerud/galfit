"""
Module containing base classes for constructing functions
and routines that combine multiple functions.

Classes
-------
Container(**functions)
    A base class that is used for joining multiple functions.
    The functions are stored within a container dictionary.

Function
    Base Function class for working with function-like objects whose
    parameters are contained in a single Parameters object. Designed
    to be inherited rather than utilized directly. The Parameters
    object is assumed to be stored as the attribute `state`.

    Functions can be merged using addition and multiplication as
    follows:

        Func1 + Func2 -> Func1() + Func2()
        Func1 * Func2 -> Func1(Func2())

    See FunctionAdd and FunctionComposition for more information. More
    advanced combinations of Function instances should be done using the
    Container class, of which FunctionAdd and FunctionComposition are
    specialized cases.

FunctionAdd
    Module for merging two functions by adding their outputs together.
    Useful if they contain a set of inputs that are shared between them.

    If the two functions return dictionaries, then they are expected to
    contain the same keys and the values will be added together.

FunctionComposition
    Method for creating a composition of two functions.
    Useful if the outputs of one function serve as the
    inputs to another.
"""
from __future__ import annotations  # Allow same typing

from dataops.utils import flatten
from typing import Dict, Union
from ..parameter import StateManager, Parameter

class Container(StateManager):
    """
    A base class that is used for joining multiple functions.
    The functions are stored within a container dictionary.

    Attributes
    ----------
    container : dict
        A dictionary whose values are function objects.

    Methods
    -------
    __init__(**functions)
        Stores the specified key-Function pairs within the
        container object.

    __getitem__(name)
        Returns the associated function in container with the
        indicated key.

    __iter__
        Iterates through the items found in the container and
        yields the set of Parameter objects.
    """
    def __init__(self, **functions:Dict[str, Union[Container,Function]]):
        """
        Stores the specified key-Function pairs within the
        container object.

        Parameters
        ----------
        **functions
            A dictionary whose keys are the function names
            and whose values are Function instances.
        """
        self.container = functions

    def __getitem__(self, name:str) -> Union[Container, Function]:
        """
        Returns the associated function in container with the
        indicated key.
        """
        return self.container[name]

    def __iter__(self):
        """Iterates through the Parameter objects"""
        for x in flatten(self.container.values(), Parameter):
            yield x

class Function(StateManager):
    """
    Base Function class for working with function-like objects whose
    parameters are contained in a single Parameters object. Designed
    to be inherited rather than utilized directly. The Parameters
    object is assumed to be stored as the attribute `state`.

    Functions can be merged using addition and multiplication as
    follows:

        Func1 + Func2 -> Func1() + Func2()
        Func1 * Func2 -> Func1(Func2())

    See FunctionAdd and FunctionComposition for more information. More
    advanced combinations of Function instances should be done using the
    Container class, of which FunctionAdd and FunctionComposition are
    specialized cases.

    Attributes
    ----------
    state : Parameters
        An instance of a Parameters object. This should be defined
        by child classes.

    Methods
    -------
    __add__(other)
        Constructs a function addition

    __contains__(key)
        Checks to see if the indicated key is found in state dictionary

    __getitem__(key)
        Returns the Parameter object with the given key in the state dictionary

    __setitem__(key, value)
        Sets the value of the indicated parameter

    __iter__
        Iterates through the Parameter objects

    __mul__(other)
        Constructs a function composition

    Examples
    --------
    from galfit.container import Function
    from galfit.parameter import Parameters

    class AddValues(Function):
        def __init__(self, a:dict, b:dict):
            self.state = Parameters()
            self.state.add("a", **a)
            self.state.add("b", **b)

        def __call__(self, *args, **kwargs):
            return self['a'] + self['b']

    func = AddValues(
        a = dict(value=0.75, min=0, max=1),
        b = dict(value=0.25, min=0, max=1)
    )
    func()
    """
    def __add__(self, other:Function) -> FunctionAdd:
        """Constructs a function addition"""
        return FunctionAdd(self, other)

    def __contains__(self, key:str):
        """Checks to see if the indicated key is found in state dictionary"""
        return key in self.state

    def __init__(self):
        raise NotImplementedError("__init__ should be implemented by child class")

    def __getitem__(self, key:str) -> Number:
        """Returns the Parameter object with the given key in the state dictionary"""
        return self.state[key]

    def __setitem__(self, key:str, value:Number):
        """Sets the value of the indicated parameter"""
        self.state[key] = value

    def __iter__(self):
        """Iterates through the Parameter objects"""
        return iter(self.state)

    def __mul__(self, inner:Function) -> FunctionComposition:
        """Constructs a function composition"""
        return FunctionComposition(inner=inner, outer=self)

class FunctionAdd(Container):
    """
    Module for merging two functions by adding their outputs together.
    Useful if they contain a set of inputs that are shared between them.

    If the two functions return dictionaries, then they are expected to
    contain the same keys and the values will be added together.

    Examples
    --------
    from galfit.container import Function
    from galfit.parameter import Parameters

    class MyFunc(Function):
        def __init__(self, **kwargs):
            self.state = Parameters()
            self.state.add("temp", **kwargs)

        def __call__(self, *args, **kwargs):
            return self['temp']

    func = MyFunc(value=0.25) + MyFunc(value=0.75)
    print(func())
    """
    def __init__(self, 
        first:Union[Container,Function],
        second:Union[Container,Function]
    ):
        """
        Parameters
        ----------
        first:Function
            The first function object

        second:Function
            The second function object
        """
        self.container = {
            'first': first,
            'second': second,
        }

    def __call__(self, *args, **kwargs) -> Union[dataops.DTYPES, Dict[str,dataops.DTYPES]]:
        """
        Passes the inputs to the two functions and adds the
        outputs together. If the outputs are dictionaries, then
        the addition is performed across the keys.

        Parameters
        ----------
        *args
            Arguments to pass into the functions

        **kwargs
            Keyword arguments to pass into the functions
        """
        out1 = self['first'](*args, **kwargs)
        out2 = self['second'](*args, **kwargs)

        if isinstance(out1, dict):
            return {k:out1[k]+out2[k] for k in out1.keys()}
        else:
            return out1 + out2

class FunctionComposition(Container):
    """
    Method for creating a composition of two functions.
    Useful if the outputs of one function serve as the
    inputs to another.
    """
    def __init__(self, 
        inner:Union[Container,Function], 
        outer:Union[Container,Function]
    ):
        """
        Parameters
        ----------
        inner : Function
            Inner function. Should return either a dictionary of values
            or a single instance (e.g. array)

        outer : Function
            Outer function. Should take as input the outputs of `inner`.
        """
        self.container = {
            'inner': inner,
            'outer': outer, 
        }

    def __call__(self, *args, **kwargs) -> Union[dataops.DTYPES, Dict[str,dataops.DTYPES]]:
        """
        Returns the results of

            outer(inner(*args, **kwargs))

        if inner returns a non-dictionary and

            outer(**inner(*args, **kwargs))

        if inner returns a dictionary.

        Parameters
        ----------
        *args
            Arguments to pass into the function `inner`

        **kwargs
            Keyword arguments to pass into the function `inner`
        """
        input = self['inner'](*args, **kwargs)
        return self['outer'](**input if isinstance(input,dict) else input)