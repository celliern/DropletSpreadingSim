# %%
using GLMakie
using YAXArrays, NetCDF
# %%

# read dataset from file
c1 = Cube("data/outputs/2D_multi_drop/dθₛ=0.nc");
c2 = Cube("data/outputs/2D_multi_drop/dθₛ=7.nc");

# %%
h1 = c1[Variable="h"].data[:, :, :]
h2 = c2[Variable="h"].data[:, :, :]

# %%

t = c1.t
x = c1.x
y = c1.y
# use makie to generate animation on h

# %%
h1_ = Observable(h1[1, :, :])
h2_ = Observable(h2[1, :, :])

fig = Figure(resolution=(1000, 500), dpi=300)
display(fig)

# %%
ax1 = Axis(fig[1, 1], aspect=DataAspect())
ax2 = Axis(fig[2, 1], aspect=DataAspect())

heatmap!(ax1, x, y, h1_)
heatmap!(ax2, x, y, h2_)

record(fig, "droplets.mp4", 1:length(t), framerate=60, sleep=false) do t
    h1_[] = h1[t, :, :]
    h2_[] = h2[t, :, :]
end
