import math
import numpy as np


class SPHConfig:
    def __init__(self):
        self.rho0 = 1000.0
        self.dp = 0.020

        # Smoothing length
        self.h = 1.3 * self.dp

        # Wendland C2 support = 2h
        self.support = 2.0 * self.h

        # Physical particle mass
        self.mass = self.rho0 * self.dp**3


class WendlandC2:
    def __init__(self, h):
        self.h = h
        self.support = 2.0 * h

        # 3D normalization
        self.alpha = 21.0 / (16.0 * math.pi * h**3)

    def W(self, r):
        q = r / self.h

        if q >= 2.0:
            return 0.0

        a = 1.0 - 0.5 * q

        return self.alpha * a**4 * (2.0 * q + 1.0)


class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.cells = {}

    def cell_index(self, p):
        return (
            math.floor(p[0] / self.cell_size),
            math.floor(p[1] / self.cell_size),
            math.floor(p[2] / self.cell_size),
        )

    def build(self, pos):
        self.cells = {}

        for i, p in enumerate(pos):
            c = self.cell_index(p)

            if c not in self.cells:
                self.cells[c] = []

            self.cells[c].append(i)

    def find_neighbors(self, pos, support):
        n = len(pos)
        neighbors = [[] for _ in range(n)]

        support2 = support * support

        for i, p in enumerate(pos):
            cx, cy, cz = self.cell_index(p)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):

                        cell = (cx + dx, cy + dy, cz + dz)

                        indices = self.cells.get(cell)

                        if indices is None:
                            continue

                        for j in indices:
                            if j == i:
                                continue

                            d = pos[j] - p
                            r2 = np.dot(d, d)

                            if r2 < support2:
                                neighbors[i].append(j)

        return neighbors


def compute_density(pos, neighbors, kernel, mass):
    n = len(pos)

    rho = np.zeros(n, dtype=np.float64)

    self_contribution = mass * kernel.W(0.0)

    for i in range(n):
        density = self_contribution
        pi = pos[i]

        for j in neighbors[i]:
            r = np.linalg.norm(pos[j] - pi)
            density += mass * kernel.W(r)

        rho[i] = density

    return rho