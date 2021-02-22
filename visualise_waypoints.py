##################################################################
# VISUALISE WAYPOINTS
# Script which can be used to animate the waypoints of the boom 
# and hook block. 
##################################################################

# Libraries
import numpy as np
import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Visualisation of crane transforms
def read_csv_tf(file_name):
    R_list = []
    t_list = []

    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            R = np.array([
                [float(row[0]), float(row[1]), float(row[2])],
                [float(row[3]), float(row[4]), float(row[5])],
                [float(row[6]), float(row[7]), float(row[8])]
            ])

            t = np.array([float(row[9]), float(row[10]), float(row[11])])

            R_list.append(R)
            t_list.append(t)
    
    return R_list, t_list

R_I, t_I = read_csv_tf('data/waypoints/link_I_3.csv')
R_K, t_K = read_csv_tf('data/waypoints/link_K_3.csv')

margin = 0.25

u_limit = np.max(np.array([np.max(np.array(t_I), axis=0), np.max(np.array(t_K), axis=0)]), axis=0)
l_limit = np.min(np.array([np.min(np.array(t_I), axis=0), np.min(np.array(t_K), axis=0)]), axis=0)

print(u_limit)
print(l_limit)

width = np.max(np.array(u_limit) - np.array(l_limit))
center = (np.array(u_limit) - np.array(l_limit)) / 2.0

u_limit = center + width
l_limit = center - width


# Create OpenCV window to continuously capture from webcam
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

l = 4.12940952255126
t_pivot = []
t_filtered = []
for i in range(len(R_I)):
    # Project the translation upwards by l in the y axis
    offset = R_K[i] @ np.array([-l, 0., 0.])
    filtered = R_K[i] @ np.array([-l, 0., 0.]) + np.array([0., -l, 0.])
    t_pivot.append(offset + t_K[i])
    t_filtered.append(filtered + t_K[i])


offset_K = np.array([
    [0., 1., 0.],
    [-1., 0., 0.],
    [0., 0., 1.]
])

for i in range(len(R_I)):

    print(np.linalg.norm(t_I[i] - t_K[i]))

    plt.cla()

    ax.set_xlim(l_limit[0], u_limit[0])
    ax.set_ylim(l_limit[1], u_limit[1])
    ax.set_zlim(l_limit[2], u_limit[2])

    x, y, z = t_I[i]
    R = R_I[i]

    ax.quiver(x, y, z, R[0][0], R[1][0], R[2][0], length=1, color='r')
    ax.quiver(x, y, z, R[0][1], R[1][1], R[2][1], length=1, color='g')
    ax.quiver(x, y, z, R[0][2], R[1][2], R[2][2], length=1, color='b')

    ax.plot([x, t_filtered[i][0]],[y, t_filtered[i][1]],[z, t_filtered[i][2]])

    x, y, z = t_K[i]
    R = R_K[i]

    ax.quiver(x, y, z, R[0][0], R[1][0], R[2][0], length=1, color='r')
    ax.quiver(x, y, z, R[0][1], R[1][1], R[2][1], length=1, color='g')
    ax.quiver(x, y, z, R[0][2], R[1][2], R[2][2], length=1, color='b')

    ax.plot([x, t_pivot[i][0]],[y, t_pivot[i][1]],[z, t_pivot[i][2]])
    
    
    plt.pause(0.01)

# Write results to file
with open('output.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['x_I', 'y_I', 'z_I', 'x_K', 'y_K', 'z_K', 'x_pivot', 'y_pivot', 'z_pivot','x_filtered', 'y_filtered', 'z_filtered'])
    for i in range(len(R_I)):
        writer.writerow(list(t_I[i]) + list(t_K[i]) + list(t_pivot[i]) + list(t_filtered[i]))
        # writer.writerow(list(t_history[i]) + list(R_history[i].flatten()) + list(t_actual[i]) + list(R_actual[i].flatten()))