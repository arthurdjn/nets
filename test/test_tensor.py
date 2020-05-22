"""
This modules test all methods within a Tensor.
"""

import unittest
import numpy as np
import nets


class TestTensor(unittest.TestCase):
    def test_init(self):
        # Instantiate a tensor
        tensor = nets.Tensor(1)
        tensor = nets.Tensor(tensor)
        tensor = nets.Tensor(np.eye(3))

        # Check the data
        data = np.ones((100, 100))
        tensor = nets.Tensor(data)
        assert np.array_equal(tensor.data, data)
        tensor = nets.Tensor(data, requires_grad=True)
        assert tensor.requires_grad == True
        assert np.array_equal(tensor.grad.data, np.zeros_like(data))

    def test_ops(self):
        tensor1 = nets.Tensor([[2, 4], [6, 8]])
        tensor2 = nets.Tensor([[1, 3], [5, 7]])
        tensor1 + tensor2
        tensor1 += 1
        tensor1 * tensor2
        tensor1.T


if __name__ == '__main__':
    unittest.main()
