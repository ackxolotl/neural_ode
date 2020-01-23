from sphere import generate_concentric_shape


inner_ring, outer_ring = generate_concentric_shape((0, 4), (10, 15), 100, 400)
x_inner, y_inner = inner_ring
x_outer, y_outer = outer_ring

print(x_inner.shape)
print(y_inner.shape)

print(x_outer.shape)
print(y_outer.shape)