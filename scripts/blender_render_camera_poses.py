import bpy
import os
import csv
from mathutils import Euler, Vector

# Set the path to the CSV file
csv_file_path = "/home/peter/PycharmProjects/vivarium/scenes/scene_1/camera_poses.csv"
# Set the output directory for rendered images
output_directory = "/home/peter/PycharmProjects/vivarium/scenes/scene_0/images"

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Open the CSV file and read camera poses
with open(csv_file_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # Skip header if present
    for row in csvreader:
        frame_number, pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = map(float, row)

        # Set the frame
        bpy.context.scene.frame_set(int(frame_number))

        # Set camera location
        bpy.context.scene.camera.location = Vector((pos_x, pos_y, pos_z))

        # Set camera rotation
        bpy.context.scene.camera.rotation_euler = Euler((rot_x, rot_y, rot_z), 'XYZ')

        # Render the image
        bpy.ops.render.render(write_still=True)

        # Save the rendered image to the specified output directory
        output_file_path = os.path.join(output_directory, f"frame_{int(frame_number):04d}.png")
        bpy.data.images['Render Result'].save_render(filepath=output_file_path)


# Print a message when the rendering is complete
print(f"Images rendered for each frame using camera poses from {csv_file_path}.")
print(f"Rendered images saved to {output_directory}.")
