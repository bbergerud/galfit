import galfit
import importlib
import math
import scipy.stats as stats
import unittest
from galfit.parameter import Parameter, Parameters

class TestParameter(unittest.TestCase):

    def test_value(self):
        """
        Tests to see if an exception is raised when passing
        in both a value and expr.
        """
        # Assign value
        p = Parameter("temp", value=1)
        self.assertEqual(p.value, 1)

        # Assign value and expression, which should raise an error
        self.assertRaises(TypeError, Parameter, name="temp", value = 1, expr = lambda d: 1)

    def test_get_value(self):
        """
        Tests to see that get_value works properly
        """
        # Assign value
        p = Parameter("temp", value=5)
        self.assertEqual(p.get_value(), 5)

        # Expression
        p = Parameter("temp", expr = lambda d: 5)
        self.assertEqual(p.get_value(), 5)

    def test_set_value(self):
        """
        Tests to see that set_value works properly 
        """
        # Assign value
        p = Parameter("temp", value = 5)
        p.set_value(10)
        self.assertEqual(p.get_value(), 10)

        # Expression (should raise error)
        p = Parameter("temp", expr = lambda d: 1)
        self.assertRaises(AttributeError, p.set_value, 10)

    def test_bounds(self):
        """
        Tests that the bounds are stored properly and that sampling
        producing values within the specified range.
        """
        # Test if min > max
        self.assertRaises(ValueError, Parameter, name="temp", min=1, max=0)

        # Test min and max values are properly stored
        vmin = 0; vmax = 10
        p = Parameter("temp", min=vmin, max=vmax)
        self.assertEqual(p.min, vmin)
        self.assertEqual(p.max, vmax)

        # test sampled values are within range
        sample = p.get_sample(100)
        self.assertTrue(min(sample) >= vmin)
        self.assertTrue(max(sample) <= vmax)

        # test raise if min or max are not finite
        self.assertRaises(ValueError, Parameter(name="temp", min=-math.inf, max=0).get_sample, size=1)
        self.assertRaises(ValueError, Parameter(name="temp", min=0, max=math.inf).get_sample, size=1)
        self.assertRaises(ValueError, Parameter(name="temp", min=-math.inf, max=math.inf).get_sample, size=1)

        # test log prior (within range)
        p = Parameter("temp", value=0.5, min=0, max=1)
        self.assertEqual(p.get_log_prior(), 0)

        # test log prior (outside range)
        p = Parameter("temp", value=-1, min=0, max=1)
        self.assertEqual(p.get_log_prior(), -math.inf)

    def test_prior(self):
        # Scipy
        if importlib.util.find_spec("scipy") is not None:
            import scipy.stats as stats
            p = Parameter("scipy", value=0, prior=stats.norm(loc=0, scale=1))
            self.assertEqual(len(p.get_sample(5)), 5)
            self.assertEqual(p.get_log_prior(), stats.norm(loc=0, scale=1).logpdf(0))
        else:
            print("Did not test scipy.stats prior")

        # Tensorflow
        if importlib.util.find_spec("tensorflow_probability") is not None:
            import tensorflow_probability as tfp
            prior = tfp.distributions.normal(loc=0, scale=1)
            p = Parameter("tensorflow", value=0, prior=prior)
            self.assertEqual(p.get_log_prior(), prior.log_prob(0))
            self.assertEqual(len(p.get_sample(5)), 5)
        else:
            print("Did not test tensorflow prior")

        # PyTorch
        if importlib.util.find_spec("torch") is not None:
            import torch
            prior = torch.distributions.normal.Normal(loc=0, scale=1)
            p = Parameter("tensorflow", value=torch.Tensor([0.]), prior=prior)
            self.assertEqual(p.get_log_prior(), prior.log_prob(torch.Tensor([0.])))
            self.assertEqual(len(p.get_sample(5)), 5)
        else:
            print("Did not test torch prior")

    def test_quantity(self):
        pass

    def test_transform(self):
        p = Parameter("temp", value=1, transform=math.log)
        self.assertEqual(p.get_value(), 0)

        p = Parameter("temp", value=0, transform=math.exp)
        self.assertEqual(p.get_value(), 1)

class TestParameters(unittest.TestCase):
    def test_values(self):
        p = Parameters()
        p.add(name="a", value=1)
        p.add(name="b", value=2)
        self.assertEqual(p.values, [1,2])

        p.values=[2,1]
        self.assertEqual(p.values, [2,1])

        p.set_values([1,2])
        self.assertEqual(p.values, [1,2])

    def test_contains(self):
        p = Parameters()
        p.add(name="a", value=1)

        self.assertTrue("a" in p)
        self.assertFalse("b" in p)

    def test_getitem(self):
        p = Parameters()
        p.add(name="a", value=1)        
        p.add(name="b", expr=lambda d: 2)
        self.assertEqual(p["a"], 1)
        self.assertEqual(p["b"], 2)

    def test_setitem(self):
        p = Parameters()
        p.add(name="a", value=1)
        p["a"] = 2
        self.assertEqual(p["a"], 2)

class TestStateManager(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main(exit=False)