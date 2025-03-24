import math
import magpylib as magpy
import numpy as np
from simple_pid import PID
import matplotlib.pyplot as plt
from vpython import sphere, vector, rate, canvas


def make_coil(current, num_coils, coil_spacing, position,  diameter = 0.01):
    x, y, z = position 
    coils = []
    for i in range(num_coils):
        coil = magpy.current.Circle(
            current=current, 
            diameter= diameter, 
            position=(x, y, z - i * coil_spacing)
        )
        coils.append(coil)
    
    return coils

def construct_system(x_current, y_current):
    # Generate 4 coil pillars in a square formation with 2cm spacing
    coil_pillars = []
    square_spacing = 0.027/2  # 2 cm spacing

    #positions
    x1_pos = (-square_spacing, -square_spacing, 0)
    x2_pos = (square_spacing, square_spacing, 0)
    y1_pos = (square_spacing, -square_spacing, 0)
    y2_pos =  (-square_spacing, square_spacing, 0)

    # 29 coil high

    diameter_start = 0.007
    diameter_end = 0.019
    diameter_values = np.linspace(diameter_start, diameter_end, 34)[::-1]

    for diameter in diameter_values:
        # coil_pillars.append(make_coil(current=5, num_coils=370, coil_spacing=0.00033, position=pos, diameter=0.019))
        coil_pillars.append(make_coil(current= x_current, num_coils=30 ,coil_spacing=0.00033, position=x1_pos, diameter=diameter))
        coil_pillars.append(make_coil(current= -x_current, num_coils=30 ,coil_spacing=0.00033, position=x2_pos, diameter=diameter))
        coil_pillars.append(make_coil(current= y_current, num_coils=30 ,coil_spacing=0.00033, position=y1_pos, diameter=diameter))
        coil_pillars.append(make_coil(current= -y_current, num_coils=30 ,coil_spacing=0.00033, position=y2_pos, diameter=diameter))

    return coil_pillars
    
    
def plot_coil_pillars(coil_pillars):
    """Plots multiple coil pillars with alternating colors."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define colors to alternate
    colors = ['b', 'r' , 'g'] #, 'g', 'm', 'c', 'y'  # You can add more colors if needed

    for i, coils in enumerate(coil_pillars):
        for j, coil in enumerate(coils):
            x, y, z = coil.position
            circle_theta = np.linspace(0, 2 * np.pi, 100)
            x_circle = x + (coil.diameter / 2) * np.cos(circle_theta)
            y_circle = y + (coil.diameter / 2) * np.sin(circle_theta)
            z_circle = np.full_like(circle_theta, z)

            # Alternate colors based on index
            color = colors[(i + j) % len(colors)]
            ax.plot(x_circle, y_circle, z_circle, color=color)

    ax.scatter(0, 0, 0.03, color='red', label='Object')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Coil Pillars')
    plt.show()




def compute_magnetic_field(coil_pillars, position):
    """
    Computes the magnetic field at a given point in space due to all coil pillars.

    Parameters:
        coil_pillars (list of lists): Each sublist contains coils forming a pillar.
        position (tuple): The (x, y, z) coordinates where the field is calculated.

    Returns:
        np.array: Magnetic field vector [Bx, By, Bz] in Tesla.
    """
    total_B = np.array([0.0, 0.0, 0.0])

    for coils in coil_pillars:
        for coil in coils:
            total_B += coil.getB(position)  # Summing the field contributions
    return total_B

def compute_magnetic_moment(M=1.2e6, R=0.0125, h=0.0035):
    """
    Computes the magnetic moment of a cylindrical neodymium magnet.
    
    Parameters:
        M (float): Magnetization of the material (A/m), default ~1.2e6 A/m for strong NdFeB.
        R (float): Radius of the magnet (m).
        h (float): Thickness of the magnet (m).

    Returns:
        float: Magnetic moment (A·m²).
    """
    V = np.pi * R**2 * h  # Volume of the cylindrical magnet
    m = M * V  # Magnetic moment
    return m


def compute_magnetic_gradient(coil_pillars, position, delta=0.001):
    """
    Computes the gradient of the magnetic field in the z-direction numerically.
    
    Parameters:
        coil_pillars (list): List of coils in the system.
        position (tuple): (x, y, z) coordinates where the gradient is computed.
        delta (float): Small displacement for numerical differentiation.

    Returns:
        float: Approximate gradient dBz/dz in Tesla per meter.
    """
    z = position[2]
    
    Bz_above = compute_magnetic_field(coil_pillars, (position[0], position[1], z + delta))[2]
    Bz_below = compute_magnetic_field(coil_pillars, (position[0], position[1], z - delta))[2]

    dBz_dz = (Bz_above - Bz_below) / (2 * delta)
    return dBz_dz

def compute_magnetic_gradient_xyz(coil_pillars, position, delta=0.001):
    """
    Computes the gradient of the magnetic field in all three spatial directions numerically.
    
    Parameters:
        coil_pillars (list): List of coils in the system.
        position (tuple): (x, y, z) coordinates where the gradient is computed.
        delta (float): Small displacement for numerical differentiation.

    Returns:
        tuple: Approximate gradients (dBx/dx, dBy/dy, dBz/dz) in Tesla per meter.
    """
    x, y, z = position
    
    # Compute B field components at displaced positions
    Bx_plus = compute_magnetic_field(coil_pillars, (x + delta, y, z))[0]
    Bx_minus = compute_magnetic_field(coil_pillars, (x - delta, y, z))[0]
    By_plus = compute_magnetic_field(coil_pillars, (x, y + delta, z))[1]
    By_minus = compute_magnetic_field(coil_pillars, (x, y - delta, z))[1]
    Bz_plus = compute_magnetic_field(coil_pillars, (x, y, z + delta))[2]
    Bz_minus = compute_magnetic_field(coil_pillars, (x, y, z - delta))[2]

    # Compute gradients in each direction
    dBx_dx = (Bx_plus - Bx_minus) / (2 * delta)
    dBy_dy = (By_plus - By_minus) / (2 * delta)
    dBz_dz = (Bz_plus - Bz_minus) / (2 * delta)
    
    return dBx_dx, dBy_dy, dBz_dz





def magnetic_field(x1_current, x2_current, y1_current, y2_current, magnetic_moment):

    coil_pillars = construct_system(x1_current, x2_current, y1_current, y2_current)

    z_magnet = 0.03 # Height of the magnet roughly at equilibrium
    magnetic_point = (0.0, 0.0, z_magnet)
    dBx_dx, dBy_dy, dBz_dz = compute_magnetic_gradient_xyz(coil_pillars, magnetic_point)

    magnetic_force_x = np.round((dBx_dx* magnetic_moment), 3)
    magnetic_force_y = np.round((dBy_dy* magnetic_moment), 3)
    magnetic_force_z = np.round((dBz_dz* magnetic_moment), 3)

    magnetic_force = (magnetic_force_x, magnetic_force_y, magnetic_force_z)

    return magnetic_force


def update_position(starting_position, force_vector, mass, dt):
    """
    Calculates the change in position of an object given the starting position, force vector, mass, and time step.

    Parameters:
        starting_position (tuple): The initial (x, y, z) coordinates of the object.
        force_vector (tuple): The force vector (Fx, Fy, Fz) acting on the object.
        mass (float): The mass of the object.
        dt (float): The small time step.

    Returns:
        tuple: The new (x, y, z) coordinates of the object.
    """
    # Unpack the starting position and force vector
    x, y, z = starting_position
    Fx, Fy, Fz = force_vector

    # Calculate acceleration
    ax = Fx / mass
    ay = Fy / mass
    az = Fz / mass

    # Update position using basic kinematics
    new_x = x + 0.5 * ax * dt**2
    new_y = y + 0.5 * ay * dt**2
    new_z = z + 0.5 * az * dt**2

    return new_x, new_y, new_z
def update_position_velocity(position, velocity, force, mass, dt):
    """
    Updates position and velocity using force and time step (basic Newtonian mechanics).
    
    Parameters:
        position (tuple): Current (x, y, z) position.
        velocity (tuple): Current (vx, vy, vz) velocity.
        force (tuple): Force vector (Fx, Fy, Fz).
        mass (float): Mass of the object.
        dt (float): Time step.
    
    Returns:
        tuple: Updated (position, velocity)
    """
    x, y, z = position
    vx, vy, vz = velocity
    Fx, Fy, Fz = force

    # Acceleration
    ax = Fx / mass
    ay = Fy / mass
    az = Fz / mass

    # Update velocity
    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    vz_new = vz + az * dt

    # Update position
    x_new = x + vx * dt + 0.5 * ax * dt**2
    y_new = y + vy * dt + 0.5 * ay * dt**2
    z_new = z + vz * dt + 0.5 * az * dt**2

    return (x_new, y_new, z_new), (vx_new, vy_new, vz_new)



def animate_particle_vpython(positions, dt, skip=1):
    """
    Animates the position of a particle given a list of positions using VPython.

    Parameters:
        positions (list of tuples): A list of (x, y, z) coordinates of the particle.
        dt (float): The time step for the animation.
        skip (int): The number of positions to skip between frames to speed up the animation.
    """
    # Create a VPython canvas
    scene = canvas(title='Particle Animation', width=800, height=600, center=vector(0, 0, 0), background=vector(0.8, 0.8, 0.8))

    # Create a sphere to represent the particle
    particle = sphere(pos=vector(positions[0][0], positions[0][1], positions[0][2]), radius=0.01, color=vector(1, 0, 0))

    # Animate the particle
    for i in range(0, len(positions), skip):
        rate(1/dt)  # Control the animation speed
        pos = positions[i]
        particle.pos = vector(pos[0], pos[1], pos[2])
