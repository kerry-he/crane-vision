##################################################################
# PROCESS POSE
# Script used to process results from a YuMi experiment using
# main.py. Includes rotating measured translation frame camera
# frame to tag frame, and doing filtering calculation on measured
# translation.
##################################################################

# Libraries
import numpy as np
import csv
import transforms3d

def main():
    R_list = []
    t_list = []
    t_actual_list = []
    R_actual_list = []

    # Read results from CSV
    with open('data/output_pose_raw.csv') as csv_file:
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
        # Covnert matrix to Euler angles
        euler = transforms3d.euler.mat2euler(R)
        euler_list.append(euler)

        # Calculate measured translation in tag frame
        t_rotated.append(np.matmul(R, t))


    t_list -= t_list[0]
    t_rotated -= t_rotated[0]
    t_actual_list -= t_actual_list[0]
    t_actual_list *= 1000

    # Process actual rotation to be in a comparable frame to measured
    actual_euler_list = []
    for R in R_actual_list:
        R_offset = np.array([[0., 1., 0.],
                            [1., 0., 0.],
                            [0., 0., -1.]])

        R_transformed = np.linalg.inv(np.matmul(R, np.linalg.inv(R_offset)))
        euler = transforms3d.euler.mat2euler(R_transformed)        
        actual_euler_list.append(euler)

    # Apply filtering equation
    l = 200.
    for i in range(len(t_list)):
        filtered_list.append(t_rotated[i] - np.matmul((np.eye(3) - R_list[i]), np.array([0., l, 0.])))


    # Write results to file
    with open('data/output_pose_processed.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['Measured (x)', 'Measured (y)', 'Measured (z)',
                        'Actual (x)', 'Actual (y)', 'Actual (z)',
                        'Rotated (x)', 'Rotated (y)', 'Rotated (z)',
                        'Measured R (x)', 'Measured R (y)', 'Measured R (z)',
                        'Actual R (x)', 'Actual R (y)', 'Actual R (z)',
                        'Filtered (x)', 'Filtered (y)', 'Filtered (z)'])
        for i in range(len(t_list)):
            writer.writerow(list(t_list[i]) + list(t_actual_list[i]) + list(t_rotated[i]) + list(
                euler_list[i]) + list(actual_euler_list[i]) + list(filtered_list[i]))

if __name__ == "__main__":
    main()
