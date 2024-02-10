import csv
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation


# def convert_to_quaternion(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
#     # Convert Euler angles (rot_x, rot_y, rot_z) to a quaternion
#     quaternion = Quaternion(axis=[1, 0, 0], angle=np.radians(rot_x)) \
#                  * Quaternion(axis=[0, 1, 0], angle=np.radians(rot_y)) \
#                  * Quaternion(axis=[0, 0, 1], angle=np.radians(rot_z))
#
#     # Return quaternion components along with position
#     return quaternion.w, quaternion.x, quaternion.y, quaternion.z, pos_x, pos_y, pos_z


def convert_to_quaternion(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z):
    # Create a rotation matrix from Euler angles
    rotation_matrix = Rotation.from_euler('xyz', [rot_x, rot_y, rot_z], degrees=False).as_matrix()

    # Create a translation vector
    translation_vector = np.array([pos_x, pos_y, pos_z, 1])  # Include 1 for homogenous coordinates

    # My hack
    translation_vector = rotation_matrix.T @ translation_vector[:3]
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

    # Combine rotation and translation to form the RT matrix
    # rt_matrix = np.zeros((4, 4))
    # rt_matrix[:3, :3] = rotation_matrix
    # rt_matrix[:, 3] = translation_vector

    # return rt_matrix
    return quaternion[0], quaternion[1], quaternion[2], quaternion[3], translation_vector[0], translation_vector[1], translation_vector[2]


def generate_images_txt(csv_file_path):

    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME

    lines = []
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header if present
        for row in csvreader:
            frame_number, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = map(float, row)
            QW, QX, QY, QZ, TX, TY, TZ = convert_to_quaternion(pos_x, pos_y, pos_z, rot_x, rot_y, rot_z)
            data = [str(int(frame_number)), str(QW), str(QX), str(QY), str(QZ), str(TX), str(TY), str(TZ), "1", "frame_" + str(int(frame_number)).zfill(4) + ".png"]
            lines.append(data)

    # Sort for colmap format
    lines = sorted(lines, key=lambda x: x[0])
    for i in range(len(lines)):
        lines[i] = ' '.join(lines[i])

    file_path = "./output/scene_2/images.txt"
    with open(file_path, 'w') as file:
        for string in lines:
            file.write(f"{string}\n\n")


def main():

    # The script needs:
    # +── path to scene
    # │   +── images
    # │   +── camera_poses.csv
    # │   +── cameras.txt

    # colmap feature_extractor --database_path database.db --image_path images --ImageReader.camera_model PINHOLE
    # colmap exhaustive_matcher --database_path database.db
    # colmap point_triangulator --database_path database.db --image_path images --input_path sparse/model --output_path sparse/model

    csv_file_path = "./output/scene_2/camera_poses.csv"
    generate_images_txt(csv_file_path)


if __name__ == '__main__':
    main()
