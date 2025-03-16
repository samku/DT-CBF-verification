import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering and set default font size to 14
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.unicode_minus": False,
    "font.size": 14,  # Set default font size
})

# Data
x = list(range(1, 38, 2))  # x-values from 1 to 37, step 2
x_red = list(range(1, 38, 2))  # x-values for red, blue, green lines (1 to 29)

# Red y-values
red_y = [
    2580300, 1143612, 667582, 495736, 404612,
    348002, 309838, 282743, 262252, 246676,
    234261, 224264, 215880, 208965, 203159, 
    198069, 193579, 189721, 186353
]

# Blue y-values
blue_y = [
    2317355, 1011414, 576858, 416771, 329776,
    274901, 237380, 210168, 189592, 173411,
    160566, 150040, 141303, 133953, 127765, 
    122449, 117740, 113662, 109955
]

# Green y-values
green_y = [
    2228219, 968394, 550191, 395033, 310088,
    256277, 219165, 192110, 171453, 155355,
    142334, 131632, 122721, 115201, 108727,
    103131, 98215, 94011, 90217
]

# Scatter points
red_scatter = (36, 162748)
blue_scatter = (16, 94160)
green_scatter = (7, 152307)

# Convert y-axis values to 10^5 units
red_y_1e5 = [y / 1e5 for y in red_y]
blue_y_1e5 = [y / 1e5 for y in blue_y]
green_y_1e5 = [y / 1e5 for y in green_y]

red_scatter_1e5 = (red_scatter[0], red_scatter[1] / 1e5)
blue_scatter_1e5 = (blue_scatter[0], blue_scatter[1] / 1e5)
green_scatter_1e5 = (green_scatter[0], green_scatter[1] / 1e5)

# Plotting
plt.figure(figsize=(10, 6))

# Plot lines with LaTeX labels
plt.plot(x_red, red_y_1e5, color='red', label=r'$\bar{\alpha} = 0.4$',linestyle='--')
plt.plot(x_red, blue_y_1e5, color='blue', label=r'$\bar{\alpha} = 0.7$',linestyle='--')
plt.plot(x_red, green_y_1e5, color='green', label=r'$\bar{\alpha} = 0.95$',linestyle='--')

# Plot scatter points
plt.scatter(*red_scatter_1e5, color='red', marker='o')
plt.scatter(*blue_scatter_1e5, color='blue', marker='o')
plt.scatter(*green_scatter_1e5, color='green', marker='o')

# Labels with LaTeX formatting and font size 14
plt.xlabel(r'$q$', fontsize=14)
plt.ylabel(r'$N_{\mathrm{tot}}$ (×$10^5$)', fontsize=14)

# Customize ticks to use font size 14
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Legend with font size 14
plt.legend(fontsize=14)

# Grid and layout
plt.grid(True)
plt.tight_layout()
plt.xlim(1,37)

# Show plot
plt.show()
