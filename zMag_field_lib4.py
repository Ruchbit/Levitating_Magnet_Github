import numpy as np
import magpylib as magpy
from magpylib import Collection

def make_coil(current, num_coils, coil_spacing, position, diameter=0.01):
    x, y, z = position
    coils = []
    for i in range(num_coils):
        coil = magpy.current.Circle(
            current=current,
            diameter=diameter,
            position=(x, y, z - i * coil_spacing)
        )
        coils.append(coil)
    return coils

def construct_system(x_current, y_current):
    """
    Constructs a full magnetic coil system with 4 symmetric coil pillars.
    Each pillar is made of stacked coils with varying diameters.
    
    Currents are applied as:
      - X pillars: +x_current and -x_current
      - Y pillars: +y_current and -y_current
    """
    coil_pillars = []
    spacing = 0.027 / 2

    # Positions for the 4 pillars (forming a square)
    x1_pos = (-spacing, -spacing, 0)
    x2_pos = (spacing, spacing, 0)
    y1_pos = (spacing, -spacing, 0)
    y2_pos = (-spacing, spacing, 0)

    # Coil diameters, from large to small
    diameter_values = np.linspace(0.007, 0.019, 34)[::-1]

    for diameter in diameter_values:
        coil_pillars.extend(make_coil(-x_current, 30, 0.00033, x1_pos, diameter))
        coil_pillars.extend(make_coil(x_current, 30, 0.00033, x2_pos, diameter))
        coil_pillars.extend(make_coil(y_current, 30, 0.00033, y1_pos, diameter))
        coil_pillars.extend(make_coil(-y_current, 30, 0.00033, y2_pos, diameter))
    system = Collection(coil_pillars)
    return system

def compute_gradient_vectorized(system, position, delta=0.001):
    x, y, z = position
    positions = np.array([
        [x + delta, y, z],
        [x - delta, y, z],
        [x, y + delta, z],
        [x, y - delta, z],
        [x, y, z + delta],
        [x, y, z - delta]
    ])
    B_fields = system.getB(positions)

    dBx_dx = (B_fields[0, 0] - B_fields[1, 0]) / (2 * delta)
    dBy_dy = (B_fields[2, 1] - B_fields[3, 1]) / (2 * delta)
    dBz_dz = (B_fields[4, 2] - B_fields[5, 2]) / (2 * delta)

    return dBx_dx, dBy_dy, dBz_dz

def compute_force_vector(dBx_dx, dBy_dy, dBz_dz):
    magnetic_moment = 2.06
    force = dBx_dx * magnetic_moment, dBy_dy * magnetic_moment, dBz_dz * magnetic_moment
    return force

def force_for_position(x_current, y_current, position):
    system = construct_system(x_current, y_current)
    dBx_dx, dBy_dy, dBz_dz = compute_gradient_vectorized(system, position)
    force = compute_force_vector(dBx_dx, dBy_dy, dBz_dz)
    return force
