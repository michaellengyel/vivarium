import csv
import math
import sqlite3
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def convert_to_colmap_pose(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
    # Blender to colmap rotation matrix
    rot_mat_b2c = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()

    # Convert Euler angles to a rotation matrix using the 'xyz' order
    rot_mat_blender = R.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=False).as_matrix()
    rot_mat = rot_mat_b2c @ rot_mat_blender.T

    # Construct the quaternion representation of the rotation
    rot_quat = R.from_matrix(rot_mat).as_quat()

    # Compute the translation vector
    translation_vector = np.dot(-rot_mat, np.array([pos_x, pos_y, pos_z]))

    # Return the pose in the desired format
    return rot_quat[3], rot_quat[0], rot_quat[1], rot_quat[2], translation_vector[0], translation_vector[1], translation_vector[2]


def generate_images_txt(csv_file_path, image_order, output_file_path):

    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    lines = []
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header if present
        for row in csvreader:
            frame_number, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = map(float, row)
            QW, QX, QY, QZ, TX, TY, TZ = convert_to_colmap_pose(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z)
            data = [str(int(frame_number)), str(QW), str(QX), str(QY), str(QZ), str(TX), str(TY), str(TZ), "1", "frame_" + str(int(frame_number)).zfill(4) + ".png"]
            lines.append(data)

    filename_dict = {l[-1]: l for l in lines}
    for (id, name) in image_order:
        filename_dict[name][0] = str(id)

    lines = list(filename_dict.values())
    for i in range(len(lines)):
        lines[i] = ' '.join(lines[i])

    with open(output_file_path, 'w') as file:
        for string in lines:
            file.write(f"{string}\n\n")


def get_colmap_image_order(db_file_path):

    image_order = []
    connection = sqlite3.connect(db_file_path)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM images')
    rows = cursor.fetchall()
    for row in rows:
        image_order.append((row[0], row[1]))
    return image_order


def main():

    # The script needs:
    # +── path to scene
    # │   +── images
    # │   +── camera_poses.csv
    # │   +── cameras.txt

    # colmap feature_extractor --database_path database.db --image_path images --ImageReader.camera_model PINHOLE
    # colmap exhaustive_matcher --database_path database.db --ExhaustiveMatching.block_size 300
    # Run sparse_from_known_poses.py
    # colmap point_triangulator --database_path database.db --image_path images --input_path sparse/0 --output_path sparse/0

    csv_file_path = "./scenes/scene_3/camera_poses.csv"
    db_file_path = "./scenes/scene_3/database.db"
    output_file_path = "./scenes/scene_3/sparse/0/images.txt"

    image_order = get_colmap_image_order(db_file_path)
    generate_images_txt(csv_file_path, image_order, output_file_path)


if __name__ == '__main__':
    main()
