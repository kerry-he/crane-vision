import numpy as np
import csv
from scipy.spatial.transform import Rotation


R_list = []
t_list = []
t_actual_list = []
R_actual_list = []

with open('results.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    first_row = True

    for row in csv_reader:
        if first_row:
            first_row = False
            continue

        R = np.array([
            [float(row[3]), float(row[4]), float(row[5])],
            [float(row[6]), float(row[7]), float(row[8])],
            [float(row[9]), float(row[10]), float(row[11])]
        ])

        t = np.array([float(row[0]), float(row[1]), float(row[2])])
        t_actual = np.array([float(row[12]), float(row[13]), float(row[14])])

        R_actual = np.array([
            [float(row[15]), float(row[16]), float(row[17])],
            [float(row[18]), float(row[19]), float(row[20])],
            [float(row[21]), float(row[22]), float(row[23])]
        ])

        R_list.append(R)
        t_list.append(t)
        t_actual_list.append(t_actual)
        R_actual_list.append(R_actual)

t_rotated = []
euler_list = []
filtered_list = []

for i, (R, t) in enumerate(zip(R_list, t_list)):
    # print(i)
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz')
    euler_list.append(euler)

    # r = Rotation.from_euler('xyz', [0., 0., -euler[-1]])
    # R = r.as_matrix()

    t_rotated.append(R @ t)


t_list -= t_list[0]
t_rotated -= t_rotated[0]
t_actual_list -= t_actual_list[0]
t_actual_list *= 1000



actual_euler_list = []
for R in R_actual_list:
    r = Rotation.from_matrix(R)
    # offset = Rotation.from_euler('xyz', [np.pi, 0., np.pi/2.])
    offset = Rotation.from_quat([0.707, 0.707, 0., 0.])
    r = r * offset.inv() 
    euler = r.inv().as_euler('xyz')
    actual_euler_list.append(euler)

l = 200.
for i in range(len(t_list)):
    filtered_list.append(t_rotated[i] - (np.eye(3) - R_list[i]) @ np.array([0., l, 0.]))


# Write results to file
with open('processed.csv', mode='w') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(['Measured (x)', 'Measured (y)', 'Measured (z)', 
                    'Actual (x)', 'Actual (y)', 'Actual (z)', 
                    'Rotated (x)','Rotated (y)', 'Rotated (z)', 
                    'Measured R (x)','Measured R (y)', 'Measured R (z)',
                    'Actual R (x)','Actual R (y)', 'Actual R (z)',
                    'Filtered (x)', 'Filtered (y)', 'Filtered (z)'])
    for i in range(len(t_list)):
        writer.writerow(list(t_list[i]) + list(t_actual_list[i]) + list(t_rotated[i]) + list(euler_list[i]) + list(actual_euler_list[i]) + list(filtered_list[i]))