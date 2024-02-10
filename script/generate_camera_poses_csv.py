import bpy
import csv
import math
import random
from mathutils import Vector

# Set the path for the output CSV file
output_csv_file = "camera_poses.csv"

# Open the CSV file in write mode
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header
    csvwriter.writerow(["frame_number", "pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"])

    # Generate camera poses for 100 frames
    for frame_number in range(1, 301):
        # Set the frame
        bpy.context.scene.frame_set(frame_number)

        # Random spherical coordinates within a 10m radius
        theta = random.uniform(0, 2 * math.pi)
        phi = random.uniform(0, math.pi)
        radius = random.uniform(5, 15)

        # Convert spherical coordinates to Cartesian coordinates
        pos_x = radius * math.sin(phi) * math.cos(theta) + random.uniform(-10, 10)
        pos_y = radius * math.sin(phi) * math.sin(theta) + random.uniform(-10, 10)
        pos_z = radius * math.cos(phi)

        # Set camera location
        bpy.context.scene.camera.location = Vector((pos_x, pos_y, pos_z))

        # Set camera rotation to look away from the origin
        direction = bpy.context.scene.camera.location - Vector((0, 0, 0))
        rot_quat = direction.to_track_quat('Z', 'Y')
        bpy.context.scene.camera.rotation_euler = rot_quat.to_euler()

        # Get Euler angles in radians
        rot_x, rot_y, rot_z = bpy.context.scene.camera.rotation_euler

        # Write camera pose to CSV
        csvwriter.writerow([frame_number, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z])

print(f"Camera poses generated and saved to {output_csv_file}.")
