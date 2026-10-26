from qmlbench.scaling import gradient_variance, kernel_concentration


def test_global_cost_gradient_variance_shrinks_with_qubits():
    small = gradient_variance(2, cost="global", samples=150, seed=0)
    large = gradient_variance(6, cost="global", samples=150, seed=0)
    assert large < small / 4


def test_local_cost_decays_slower_than_global():
    assert gradient_variance(6, cost="local", samples=150) > gradient_variance(6, cost="global", samples=150)


def test_kernel_values_concentrate_with_qubits():
    mean2, var2 = kernel_concentration(2, num_points=20)
    mean8, var8 = kernel_concentration(8, num_points=20)
    assert var8 < var2 / 10
    assert mean8 < mean2
