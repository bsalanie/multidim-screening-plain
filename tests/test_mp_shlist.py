# example of sharing a sharablelist between processes
from multiprocessing import Pool
from multiprocessing.shared_memory import ShareableList

import numpy as np
from bs_python_utils.bs_opt import minimize_free


def obj_grad(x, args, gr=False):
    a = args[0]
    obj = np.sum((x - a) ** 2)
    if gr:
        grad = 2.0 * (x - a)
        return obj, grad
    else:
        return obj


def obj_fun(x, args):
    return obj_grad(x, args, gr=False)


def grad_fun(x, args):
    return obj_grad(x, args, gr=True)[1]


def minimize_one(sl, i):
    m = 10
    # we take m at a time
    r = np.array([sl[m * i + j] for j in range(m)])
    print(f"Doing {i=}, {r=}")
    res = minimize_free(
        obj_fun,
        grad_fun,
        np.zeros(m),
        args=[
            r,
        ],
    )
    print(f"   got {res.x=}")
    return res.x


def test_mini_mp():
    n_cases, m = 20, 10
    rng = np.random.default_rng(67432)
    random_list = rng.normal(size=n_cases * m).tolist()
    # create a shared list
    sl = ShareableList(random_list)
    # report the shared list
    print(sl)

    with Pool() as pool:
        list_res = pool.starmap(minimize_one, [(sl, i) for i in range(n_cases)])

    for j in range(n_cases):
        assert np.allclose(list_res[j], random_list[j * m : (j + 1) * m])

    # close the shared memory
    sl.shm.close()
    # release the shared memory
    sl.shm.unlink()
