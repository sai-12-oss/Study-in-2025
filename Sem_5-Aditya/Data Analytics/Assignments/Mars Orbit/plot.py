import math
import numpy as np
from matplotlib import pyplot as plt


def intersection(c, r, e1, e2, z):
    c = math.radians(c)
    e2 = math.radians(e2)
    z = math.radians(z)

    a = e1*math.cos(e2) - math.cos(c)
    b = e1*math.sin(e2) - math.sin(c)

    roots = np.roots([1, 2*(a*math.cos(z) + b*math.sin(z)), a**2 + b**2 - r**2])
    root = max(roots)
    point = (e1*math.cos(e2) + root*math.cos(z), e1*math.sin(e2) + root*math.sin(z))
    
    return point


def polar_to_cartesian(r, theta):
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return x, y


def plot(c, r, e1, e2, z, s, times, oppositions):
    equant_longitudes = [(z + s * i) % 360 for i in times]
    _, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim([-r - 2, r + 2])
    ax.set_ylim([-r - 2, r + 2])

    center_x, center_y = polar_to_cartesian(1, c)
    ax.scatter(center_x, center_y, color='b', label=f"Center (1,{round(c,2)})", s=75)
    
    circle = plt.Circle((center_x, center_y), r, color='b', fill=False, label="Orbit")
    ax.add_artist(circle)

    ax.scatter(0, 0, color='orange', label="Sun (0,0)", s=75)

    equant_x, equant_y = polar_to_cartesian(e1, e2)
    ax.scatter(equant_x, equant_y, color='green', label=f"Equant ({round(e1,2)},{round(e2,2)})", s=75)

    for angle in equant_longitudes:
        x_end, y_end = polar_to_cartesian(100, angle)
        ax.plot([equant_x, x_end], [equant_y, y_end], color='green', linestyle='--')

    for angle in oppositions:
        x_end, y_end = polar_to_cartesian(100, angle)
        ax.plot([0, x_end], [0, y_end], color='orange', linestyle='--')

    intersection_sun = [intersection(c, r, 0, 0, angle) for angle in oppositions]
    intersection_equant = [intersection(c, r, e1, e2, angle) for angle in equant_longitudes]
    
    intersection_x_sun, intersection_y_sun = zip(*intersection_sun)
    ax.scatter(intersection_x_sun, intersection_y_sun, color='black', label="Intersection (Sun)", s=20)
    
    intersection_x_equant, intersection_y_equant = zip(*intersection_equant)
    ax.scatter(intersection_x_equant, intersection_y_equant, color='red', label="Intersection (Equant)", s=20)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Best Fit Orbit of Mars")
    ax.legend(prop={'size': 8})
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()
