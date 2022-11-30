"""
Module containing routines for handling parameters. Implemented Parameter and
Parameters classes are similar in design to LMFIT, but aimed at a greater
object-oriented design when building complex functions. The parameters are also
designed to interface with scipy.stats for constructing priors in a Bayesian framework.

Classes
-------
StateManager
    Class containing routines for managing the state of parameter objects.
    Designed to be inherited by other classes containing parameter objects.
        Any inherited methods should define __iter__ such that it iterates
    through all the Parameter objects.

Parameter
    The Parameter class is used as a container to represent a variable
    associated with a fittable functional expression.

Parameters
    The Parameters class is used as a container to hold a set of Parameter
    objects associated with a fittable functional expression.

    This class can be filled by either passing in a dictionary of Parameter
    objects to the constructor or by using the *add* method.
"""

from __future__ import annotations

import math
import numpy
#from astropy.units import Quantity
from dataops.utils import flatten
from inspect import cleandoc
from numbers import Number
from typing import Dict, Iterable, List, Optional, Tuple, Union

class StateManager:
    """
    Class containing routines for managing the state of parameter objects.
    Designed to be inherited by other classes containing parameter objects.

    Any inherited methods should define __iter__ such that it iterates
    through all the Parameter objects.

    Methods
    -------
    get_expr()
        Returns a list of the parameter objects that are expressions.

    get_params()
        Returns a list of the parameter objects that are not expressions.

    get_fixed_params()
        Returns a list of the parameter objects that are fixed.

    get_free_params()
        Returns a list of the parameter objects that are free.

    get_bounds(params)
        Returns a list containing the lower and upper bounds for the
        parameter objects. If no set of parameters is passed, then the
        free parameters are used.

    get_bounds_lower(params)
        Returns a list of the lower bounds for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

    get_bounds_upper(params)
        Returns a list of the upper bounds for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

    get_log_prior(params)
        Returns a list of the log priors for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

    get_names(params)
        Returns a list of the names for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

    get_sample(size, params)
        Returns a list of samples of the designated size for each
        of the parameter objects. If no set of parameters is passed,
        then the free parameters are used.

    get_values(params)
        Returns a list of the values for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

    set_values(values, params)
        Assigns the set of values to the parameters objects.
        If no set of parameters is passed, then the free
        parameters are used.

    fix_params(*names)
        Sets all the parameter objects with the given names to have a `vary`
        attribute of False.  If no names are passed, then all parameters are
        set to have the `vary` attribute as False.

    fix_then_free_params(*names)
        Sets all the parameter objects with the given names to have a `vary`
        attribute of True and all others to have a `vary` attribute of False.

    free_params(*names)
        Sets all the parameter objects with the given names to have a `vary`
        attribute of True. If no names are passed, then all parameters are
        set to have the `vary` attribute as True.

    free_then_fix_params(*names)
        Sets all the parameter objects with the given names to have a `vary`
        attribute of Fallse and all others to have a `vary` attribute of True.

    Properties
    ----------
    bounds
        Returns the lower and upper bounds of the fittable parameters. These
        are returned as a tuple of lists.

    bounds_lower
        Returns a list of the lower bounds of the fittable parameters.

    bounds_upper
        Returns a list of the upper bounds of the fittable parameters.

    log_prior
        Returns a list of the log_prior terms of the fittable parameters.

    names
        Returns a list of the name attributes of the fittable parameters.  

    values
        Returns a list of the value attributes of the fittable parameters.  

    Setter
    ------
    values(*values):
        Assigns the set of values to the free parameters.
    """
    def get_expr(self) -> List[Parameter]:
        """
        Returns a list of the parameter objects that are expressions.
        """
        params = flatten([x for x in self], Parameter)
        return [p for p in params if p.is_expr]

    def get_params(self) -> List[Parameter]:
        """
        Returns a list of the parameter objects that are not expressions.
        Any duplicated parameters are dropped.
        """
        # Dictionary will remove possible duplicate cases (shared parameters)
        params = tuple(dict.fromkeys(flatten([x for x in self], Parameter)))
        return [p for p in params if not p.is_expr]

    def get_fixed_params(self) -> List[Parameter]:
        """
        Returns a list of the parameter objects that are fixed.
        """
        return [p for p in self.get_params() if not p.is_free]

    def get_free_params(self) -> List[Parameter]:
        """
        Returns a list of the parameter objects that are free.
        """
        return [p for p in self.get_params() if p.is_free]

    def get_bounds(self, params:Optional[List[Parameter]]=None) -> List[List[Number], List[Number]]:
        """
        Returns a list containing the lower and upper bounds for the
        parameter objects. If no set of parameters is passed, then the
        free parameters are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        bounds_lower: List[Number]
            A list containing the lower bound associated with each parameter.

        bounds_upper: List[Number]
            A list containing the upper bound associated with each parameter.
        """
        if params is None:
            params = self.get_free_params()
        return (self.get_bounds_lower(params), self.get_bounds_upper(params))

    def get_bounds_lower(self, params:Optional[List[Parameter]]=None) -> List[Number]:
        """
        Returns a list of the lower bounds for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        bounds: List[Number]
            A list containing the lower bound associated with each parameter.
        """
        if params is None:
            params = self.get_free_params()
        return [p.min for p in self.get_free_params()]

    def get_bounds_lower(self, params:Optional[List[Parameter]]=None) -> List[Number]:
        """
        Returns a list of the upper bounds for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        bounds: List[Number]
            A list containing the upper bound associated with each parameter.
        """
        if params is None:
            params = self.get_free_params()
        return [p.max for p in self.get_free_params()]

    def get_log_prior(self, params:Optional[List[Parameter]]=None) -> List[Number]:
        """
        Returns a list of the log priors for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        log_prior: List[Number]
            A list containing the log priors associated with each parameter.
        """
        if params is None:
            params = self.get_free_params()
        return [p.get_log_prior() for p in params]

    def get_names(self, params:Optional[List[Parameter]]=None) -> List[str]:
        """
        Returns a list of the names for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        log_prior: List[Number]
            A list containing the names associated with each parameter.
        """
        if params is None:
            params = self.get_free_params()
        return [p.name for p in params]

    def get_sample(self, size:int, params:Optional[List[Parameter]]=None):
        """
        Returns a list of samples of the designated size for each
        of the parameter objects. If no set of parameters is passed,
        then the free parameters are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        log_prior: List[Number], List[Array]
            A list containing the sampled objects.
        """
        if params is None:
            params = self.get_free_params()
        return [p.get_sample(size) for p in params]

    def get_values(self, params:Optional[List[Parameter]]) -> List[Number]:
        """
        Returns a list of the values for the parameter objects.
        If no set of parameters is passed, then the free parameters
        are used.

        Parameters
        ----------
        params : List[parameters], optional
            The set of parameter objects. If None, then the free
            parameters associated with the model are used.

        Returns
        -------
        values: List[Number]
            A list containing the values associated with each parameter.
        """
        if params is None:
            params = self.get_free_params()
        return [x.get_value() for x in params]

    def set_values(self, values, params:Optional[List[Parameter]]=None) -> None:
        """
        Assigns the set of values to the parameters objects.
        If no set of parameters is passed, then the free
        parameters are used.

        Parameters
        ----------
        *values
            The values to assign
        """
        if params is None:
            params = self.get_free_params()
        for p,v in zip(params, flatten(values)):
            p.set_value(v)

    def fix_params(self, *names:str) -> None:
        """
        Sets all the parameter objects with the given names to have a `vary`
        attribute of False.  If no names are passed, then all parameters are
        set to have the `vary` attribute as False.

        Parameters
        ----------
        *names : str
            A sequence of strings representing the parameters to fix.

        Examples
        --------
        from galfit.parameter import Parameter, Parameters

        a = Parameter(name='a', value=2, vary=True)
        b = Parameter(name='b', value=1, vary=False)
        p = Parameters(a=a, b=b)

        print(p.get_fixed_params())
        p.fix_params('a')
        print(p.get_fixed_params())
        """
        names = list(flatten(names))
        for p in self.get_params():
            if (not names) or (p.name in names):
                p.vary = False

    def fix_then_free_params(self, *names:str) -> None:
        """
        Sets all the parameter objects with the given names to have a `vary`
        attribute of True and all others to have a `vary` attribute of False.

        Parameters
        ----------
        *names : str
            A sequence of strings representing the parameters to free.

        Examples
        --------
        from galfit.parameter import Parameter, Parameters

        a = Parameter(name='a', value=2, vary=True)
        b = Parameter(name='b', value=1, vary=False)
        p = Parameters(a=a, b=b)

        print(p.get_names(p.get_free_params()))
        p.fix_then_free_params('b')
        print(p.get_names(p.get_free_params()))
        """
        names = list(flatten(names))
        for p in self.get_params():
            if p.name in names:
                p.vary = True
            else:
                p.vary = False

    def free_params(self, *names:str) -> None:
        """
        Sets all the parameter objects with the given names to have a `vary`
        attribute of True. If no names are passed, then all parameters are
        set to have the `vary` attribute as True.

        Parameters
        ----------
        *names : str
            A sequence of strings representing the parameters to free.

        Examples
        --------
        from galfit.parameter import Parameter, Parameters

        a = Parameter(name='a', value=2, vary=True)
        b = Parameter(name='b', value=1, vary=False)
        p = Parameters(a=a, b=b)

        print(p.get_names(p.get_free_params()))
        p.free_params('b')
        print(p.get_names(p.get_free_params()))
        """
        names = list(flatten(names))
        for p in self.get_params():
            if (not names) or (p.name in names):
                p.vary = True

    def free_then_fix_params(self, *names:str) -> None:
        """
        Sets all the parameter objects with the given names to have a `vary`
        attribute of Fallse and all others to have a `vary` attribute of True.

        Parameters
        ----------
        *names : str
            A sequence of strings representing the parameters to fix.

        Examples
        --------
        >> from galfit.parameter import Parameter, Parameters
        >> a = Parameter(name='a', value=2, vary=True)
        >> b = Parameter(name='b', value=1, vary=False)
        >> p = Parameters(a=a, b=b)
        >> print(p.get_names(p.get_free_params()))
        ["a"]
        >> p.free_then_fix_params('a')
        >> print(p.get_names(p.get_free_params()))
        ["b"]
        """
        names = list(flatten(names))
        for p in self.get_params():
            if p.name in names:
                p.vary = False
            else:
                p.vary = True

    @property
    def bounds(self) -> List[List[Number], List[Number]]:
        """
        Returns the lower and upper bounds of the fittable parameters. These
        are returned as a tuple of lists.
        """
        return (self.bounds_lower, self.bounds_upper)

    @property
    def bounds_lower(self) -> List[Number]:
        """
        Returns a list of the lower bounds of the fittable parameters.
        """
        return [p.min for p in self.get_free_params()]

    @property
    def bounds_upper(self) -> List[Number]:
        """
        Returns a list of the upper bounds of the fittable parameters.
        """
        return [p.max for p in self.get_free_params()]

    @property
    def log_prior(self) -> List[Number]:
        """
        Returns a list of the log_prior terms of the fittable parameters.
        """
        return [p.get_log_prior() for p in self.get_free_params()]

    @property
    def names(self) -> List[str]:
        """
        Returns a list of the name attributes of the fittable parameters.  
        """
        return [p.name for p in self.get_free_params()]

    @property
    def values(self) -> List[Number]:
        """
        Returns a list of the value attributes of the fittable parameters.  
        """
        return [x.get_value() for x in self.get_free_params()]

    @values.setter
    def values(self, *values):
        """
        Assigns the set of values to the free parameters.
        """
        for p,v in zip(self.get_free_params(), flatten(values)):
            p.set_value(v)

class Parameter(object):
    """
    The Parameter class is used as a container to represent a variable
    associated with a fittable functional expression.

    Methods
    -------
    get_value(parameter_dict)
        Returns the value associated with the Parameter object. If the Parameter
        is an expression of other Parameter objects, then a Parameters dictionary
        containing these objects should be passed.

    get_log_prior()
        Returns the logarithm of the prior. If no prior was set, then it is
        assumed to be uniformly distributed over the contrained region.
        
        Currently assumes that the prior function is one of the following and
        will check for the appropriate methods for calculating the log prior
            - scipy.stats : logpdf
            - tensorflow/torch: log_prob

    set_value(value)
        Sets the value associated with the Parameter object. Mostly used to allow
        the ability of subjecting it to the contraints set by the `min` and `max`
        parameters, although this isn't currently implemented.

    Properties
    ----------
    is_expr
        Returns True if expr!=None and False otherwise.

    is_free
        Returns True if expr=None and vary=True
    """
    def __init__(self,
            name:str,
            value:Optional[Number] = None,
            min:Optional[Number] = -math.inf,
            max:Optional[Number] = math.inf,
            vary:Optional[bool] = True,
            expr:Optional[callable] = None,
            prior:Optional[callable] = None,
            unit = None,
            transform:Optional[callable] = None,
        ) -> None:
        """
        Parameters
        ----------
        name : Text
            The name of the parameter

        prior : callable, optional
            A prior to set on the parameter. Designed to inferface with functions
            from scipy.stats, so it should have implemented `logpdf` and `rvs`
            to calculate log priors and generate random samples.

        value : Number, optional
            The value associated with the parameter. Default is None.

        min : Number, optional
            The lower bound on the value that the parameter can have. Default is -infinity.

        max : Number, optional
            The upper bound on the value that the parameter can have. Default is +infinity.

        vary : bool, optional
            Boolean indicating whether the parameter is free (True) or not (False).

        expr : callable, optional
            A callable expression that represents the parameter value. It can
            take as input a dictionary containing a set of parameters. Useful
            for making a parameter dependent on the value of other parameters.

        unit : Quantity, optional
            An astropy quantity for keeping track of physical units.

        transform : callable
            A function that modifies the values. Useful for working on a log scale.

        Raises
        ------
        ValueError
            If both value and expr are passed, then an exception is raised.

        Examples
        --------
        from astropy import units as u
        from galfit.parameter import Parameter
        from math import inf
        from scipy import stats

        p = Parameter("scale", prior=stats.halfnorm(scale=1))
        print(p.get_sample(5))

        p = Parameter("scale", value=1, min=0.01, max=inf, unit=u.kpc)
        print(p)
        """
        if expr is not None:
            if value is not None:
                raise TypeError("both expr and value can't be passed")

        if (min is not None) and (max is not None):
            if min > max:
                raise ValueError("min must be less than or equal to max")

        self.name = name
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.expr = expr
        self.prior = prior
        self.unit = unit
        self.transform = transform

    def __getitem__(self, name:str):
        return self.__dict__[name]

    def __setitem__(self, name:str, value:Union[callable, Number, Quantity]):
        self.__dict__[name] = value

    def __str__(self):
        return cleandoc(f"""
        name  = {self.name}
        value = {self.value}
        min   = {self.min}
        max   = {self.max}
        vary  = {self.vary}
        expr  = {self.expr}
        prior = {self.prior}
        unit  = {self.unit}
        """)

    def get_value(self, 
        parameter_dict:Optional[Dict[str,Parameter]] = None
    ) -> Number:
        """
        Returns the value associated with the Parameter object. If the Parameter
        is an expression of other Parameter objects, then a dictionary containing
        these objects should be passed.

        Note that a transformation is applied in this case if applicable, and thus
        may differ from the *value* attribute.

        Parameters
        ----------
        parameter_dict : Dict[str,Parameter], optional
            A dictionary containing a set of Parameter objects that the value of
            the parameter may be dependent on.

        Returns
        -------
        value : Number
            The value associated with the parameter
        """
        if self.is_expr:
            if isinstance(parameter_dict, dict):
                parameter_dict = {k:v.value for k,v in parameter_dict.items()}
            value = self.expr(parameter_dict)
        else:
            value = self.value

        if self.transform is not None:
            value = self.transform(value)

        return value if (self.unit is None) else (value*self.unit)

    def set_value(self, value:Number) -> None:
        """
        Sets the value associated with the Parameter object.

        Parameters
        ----------
        value : Number
            The value to assign to the parameter.

        Raises
        ------
        AttributeError
            If the parameter is an expression, then attempting to set a value
            will result in an exception.        
        """
        if self.is_expr:
            raise AttributeError("Can't set an expression to have a value.")
        self.value = value

    def get_log_prior(self) -> Number:
        """
        Returns the logarithm of the prior. If no prior was set, then it is
        assumed to be uniformly distributed over the contrained region.

        Currently assumes that the prior function is one of the following and
        will check for the appropriate methods for calculating the log prior

            - scipy.stats : logpdf
            - tensorflow/torch: log_prob

        Returns
        -------
        log_prior : Number
            The logarithm of the prior.

        Raises
        ------
        AttributeError
            If the parameter is an expression, then calling this
            method will raise an error.

        NotImplementedError
            If the prior has none of the associated methods for calculating
            the log prior, then an error is raised.
        """
        # If parameter is an expression, then there is log prior so raise error
        if self.is_expr:
            raise AttributeError("expr object has no log_prior")

        # If no prior has been set, evaluate over the min/max range
        if self.prior is None:
            return 0 if (self.min <= self.value <= self.max) else -math.inf
        # Use the prior's built in log probability function.
        # Raise exception if indicated methods are not found
        else:
            attrs = ["logpdf", "log_prob"]
            for attr in attrs:
                if hasattr(self.prior, attr):
                    return getattr(self.prior, attr)(self.value)

            # If method not found, then raise error
            raise NotImplementedError(f"Prior does not contain one of the following methods to calculate the log prior: {attrs}")

    def get_sample(self, size:Union[int,Tuple[int]]):
        """
        Generates random samples of the designated size based on the prior.
        If no prior was set, then it is assumed to be uniformly distributed
        over the contrained region.

        Currently assumes that the prior function is one of the following and
        will check for the appropriate methods for calculating the log prior

            -scipy.stats : rvs
            -tensorflow/torch: sample

        Parameters
        ----------
        size : int, tuple
            The size of the sample to generate.

        Returns
        -------
        sample
            A sample of random values of the designated size.

        Raises
        ------
        ValueError
            Is no prior is set and min/max are not finite,
            then an error is raised.
        """

        if self.prior is None:
            if math.isinf(self.min) or math.isinf(self.max):
                raise ValueError("To sample without a prior, min and max must both be finite")
            return numpy.random.uniform(self.min, self.max, size)
        else:
            attrs = ["rvs", "sample"]
            for attr in attrs:
                if hasattr(self.prior, attr):
                    # Cast to tuple for use with PyTorch
                    return getattr(self.prior, attr)((size,) if isinstance(size,int) else size)
            raise NotImplementedError(f"Prior does not contain one of the following methods to generate random samples: {attrs}")

    @property
    def is_expr(self) -> bool:
        """Boolean indicating whether the parameter is an expression"""
        return self.expr is not None

    @property
    def is_free(self) -> bool:
        """Boolean indicating whether the parameter is fittable"""
        return True if (not self.is_expr and self.vary) else False

class Parameters(StateManager):
    """
    The Parameters class is used as a container to hold a set of Parameter
    objects associated with a fittable functional expression.

    This class can be filled by either passing in a dictionary of Parameter
    objects to the constructor or by using the *add* method.

    Attributes
    ----------
    params : dict
        A dictionary containing the set of Parameter objects, where the
        keys are the parameter names and the values are the Parameter objects.

    Methods
    -------
    __contains__(name)
        Returns a boolean indicating whether the provided parameter
        is contained within the set of parameters.

    __getitem__(name)
        Returns the value of the Parameter object with the given name.

    __iter__
        Iterator that sequentially yields the Parameter objects that are not
        expressions (parameter.expr = None)

    __setitem__(name, value):
        Sets the value of the Parameter object with the given name.

    add(name, value, min, max, vary, expr, key_name)
        Adds a Parameter object with the given attributes to the dictionary.

    items()
        Implementation of dict.items() for iterating through Parameter objects.

    Examples
    --------
    from galfit.parameter import Parameter, Parameters

    p = Parameters()
    p.add(name="scale", value=1, min=0.01, max=10)
    p.add(name="amplitude", value=1, min=0, max=10)
    p.add(name="add", expr=lambda d: d['scale'] + d['amplitude'])

    # Print initial values
    print(p['scale'])
    print(p['amplitude'])
    print(p['add'])

    # Change the parameters.
    p['scale'] = 2
    p['amplitude'] = 0.5
    print(p['scale'])
    print(p['amplitude'])
    print(p['add'])

    # Fix the scale parameter. Print free parameters.
    p.fix_params('scale')
    print(p.names, p.values)

    # Free Parameters
    p.free_then_fix_params()
    for x in p:
        print(x.name, x.value)
    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs
            A dictionary containing the (key,value) pairs, where the values
            are Parameter objects. See the `add` method for adding parameters
            to the dictionary.
        """
        self.params = kwargs

    def __contains__(self, name:str) -> bool:
        """
        Returns a boolean indicating whether the provided parameter
        is contained within the set of parameters.
        """
        return name in self.params

    def __getitem__(self, name:str) -> Number:
        """
        Returns the value associated with the given parameter.

        Parameters
        ----------
        name : str
            The key name of the parameter in the dictionary.
        
        Returns
        -------
        value : Number
            The value associated with the parameter.
        """
        param = self.params[name]
        return param.get_value(self.params if param.is_expr else None)

    def __iter__(self):
        """
        Iterates through all the parameter objects that are not expressions.
        """
        for v in self.params.values():
            if not v.is_expr:
                yield v

    def __setitem__(self, name:str, value:Number) -> None:
        """
        Sets the parameter with the given name to have the indicated
        value for the attribute. Note that the name corresponds to the
        keyname in the dictionary rather than the parameter name.

        Parameters
        ----------
        name : str
            The key name of the parameter in the dictionary.

        value : Number
            The value to assign to the parameter.
        """
        self.params[name].set_value(value)

    def add(self, 
            name:str,
            key_name:Optional[str]=None,
            **kwargs
        ):
        """
        Adds a Parameter with the given attributes.

        Parameters
        ----------
        name : str
            The name associated with the parameter object. For storing the value
            with a different key name in the dictionary, see the `key_name` parameter.

        key_name : str, optional
            The name to use for the key in the dictionary. Useful for grouping parameter
            names together while allowing for separate key names in the dictionary. If 
            None, then the `name` attribute is used.

        **kwargs
            Additional parameter values to set. See Parameter for options.
        """
        if key_name is None:
            key_name = name
        self.params[key_name] = Parameter(name, **kwargs)

    def items(self):
        return self.params.items()
