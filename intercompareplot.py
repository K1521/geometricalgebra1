# import matplotlib.pyplot as plt

# # X values (levels)
# x9 = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
# x10 = [0,1, 2, 3, 4, 5, 6, 7, 8, 9,10]

# # Direct (e.g., point cloud)
# t1 = [1,8, 64, 512, 3136, 17024, 88320, 477568, 2004416, 8125504]
# i1 = [1,8, 48, 384, 1328, 3288, 7776, 19024, 39120, 78096,159248]
# p1 = [1,8, 56, 392, 2768, 16520, 97136, 553624, 3138360, 17760032]

# # Transformed (e.g., raymarching)
# t2 = [1,8, 64, 512, 1984, 7296, 25856, 95104, 367808, 1440064,5689536]
# i2 = [1,8, 64, 424, 448, 720, 1072, 2208, 3872, 7712,13312]
# p2 = [1,8, 64, 472, 2464, 10528, 42784, 171696, 687568, 2750608,11002432]

# # Plot setup
# plt.figure(figsize=(12, 8))

# # Direct (solid lines)
# plt.plot(x9, t1, 'o-', label='Torus (Direct)', color='tab:blue')
# plt.plot(x10, i1, 's-', label='Intersection (Direct)', color='tab:orange')
# plt.plot(x9, p1, '^-', label='Plane (Direct)', color='tab:green')

# # Transformed (dashed lines)
# plt.plot(x10, t2, 'o--', label='Torus (Transformed)', color='tab:blue')
# plt.plot(x10, i2, 's--', label='Intersection (Transformed)', color='tab:orange')
# plt.plot(x10, p2, '^--', label='Plane (Transformed)', color='tab:green')


# plt.plot(x10, [8**x for x in x10], '--', color='tab:red')
# #plt.plot(x, [2**x for x in x], '--', color='tab:red')
# # Log₂ scale for y-axis
# plt.yscale('log', base=8)

# # Labels and title
# plt.xlabel("Subdivisions")
# plt.ylabel("Count (log_8 scale)")
# plt.title("Torus, Intersection, and Plane - Comparison of Two Methods")

# # Grid and legend
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.legend()
# plt.tight_layout()

# plt.show()




import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, FuncFormatter

# Levels
x9 = list(range(10))       # Levels 0..9
x10 = list(range(11))      # Levels 0..10

# Direct
t1 = [1,8, 64, 512, 3136, 17024, 88320, 477568, 2004416, 8125504]
i1 = [1,8, 48, 384, 1328, 3288, 7776, 19024, 39120, 78096,159248]
p1 = [1,8, 56, 392, 2768, 16520, 97136, 553624, 3138360, 17760032]

# Transformed
t2 = [1,8, 64, 512, 1984, 7296, 25856, 95104, 367808, 1440064,5689536]
i2 = [1,8, 64, 424, 448, 720, 1072, 2208, 3872, 7712,13312]
p2 = [1,8, 64, 472, 2464, 10528, 42784, 171696, 687568, 2750608,11002432]

# Plot
plt.figure(figsize=(12, 8))

# Plot Direct (solid)
plt.plot(x9, t1, 'o-', label='Torus (Direct)', color='tab:blue')
plt.plot(x10, i1, 's-', label='Intersection (Direct)', color='tab:orange')
plt.plot(x9, p1, '^-', label='Plane (Direct)', color='tab:green')

# Plot Transformed (dashed)
plt.plot(x10, t2, 'o--', label='Torus (Transformed)', color='tab:blue')
plt.plot(x10, i2, 's--', label='Intersection (Transformed)', color='tab:orange')
plt.plot(x10, p2, '^--', label='Plane (Transformed)', color='tab:green')

# Reference curve 8^x
plt.plot(x10, [8**x for x in x10], '--', color='tab:red', label='8^x')

# Set log scale with base 8 and custom ticks
plt.yscale('log', base=8)
ticks = [8**i for i in x10]  # 8¹ to 8⁸
plt.yticks(ticks)
#plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"$8^{{{int(np.log(y)/np.log(8))}}}$"))

# Labels and styling
plt.xlabel("Subdivisions")
plt.ylabel("Number of Voxels")
plt.title("Torus, Intersection, and Plane - Comparison of Two Methods")
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- Compute ratios Transformed / Direct ---
# Must align by same length
def safe_div(a, b):
    return [aa / bb if bb != 0 else float('inf') for aa, bb in zip(a, b)]

print("Torus ratio (Transformed / Direct):")
print(safe_div(t2[:10], t1))  # only 10 elements in t1

print("\nIntersection ratio (Transformed / Direct):")
print(safe_div(i2, i1))  # all 11 elements

print("\nPlane ratio (Transformed / Direct):")
print(safe_div(p2[:10], p1))  # only 10 elements in p1


def percentage_of_8_power(levels, values):
    return [100 * v / (8 ** x) for x, v in zip(levels, values)]

print("\n--- Percentage of 8^x ---")

print("\nTorus (Direct) % of 8^x:")
print(percentage_of_8_power(x9, t1))

print("\nTorus (Transformed) % of 8^x:")
print(percentage_of_8_power(x10, t2))

print("\nIntersection (Direct) % of 8^x:")
print(percentage_of_8_power(x10, i1))

print("\nIntersection (Transformed) % of 8^x:")
print(percentage_of_8_power(x10, i2))

print("\nPlane (Direct) % of 8^x:")
print(percentage_of_8_power(x9, p1))

print("\nPlane (Transformed) % of 8^x:")
print(percentage_of_8_power(x10, p2))