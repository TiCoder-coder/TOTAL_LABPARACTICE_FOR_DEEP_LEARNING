import random
import unittest

import numpy as np
import torch

from course_work.utils.reproducibility import (
    build_torch_generator,
    configure_reproducibility,
    randomness_smoke_test,
    set_seed,
)


class ReproducibilityTest(unittest.TestCase):
    def tearDown(self) -> None:
        configure_reproducibility("D0")

    def test_set_seed_repeats_all_global_sequences(self) -> None:
        set_seed(42)
        first = (random.random(), np.random.random(), torch.rand(1))
        set_seed(42)
        second = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_randomness_smoke_test_passes(self) -> None:
        self.assertEqual(randomness_smoke_test()["status"], "PASS")

    def test_modes_are_explicit(self) -> None:
        self.assertFalse(configure_reproducibility("D0")["deterministic_algorithms"])
        self.assertTrue(configure_reproducibility("D1")["deterministic_algorithms"])
        with self.assertRaises(ValueError):
            configure_reproducibility("unknown")

    def test_seeded_generator_repeats(self) -> None:
        first = torch.rand(4, generator=build_torch_generator(42))
        second = torch.rand(4, generator=build_torch_generator(42))
        self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
