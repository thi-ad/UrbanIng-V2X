import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
import torch


def transform_vehicle_to_ego_2d(v_loc, v_angles, v_extent, ego_pos):
    ego_y, ego_x, ego_yaw = ego_pos[0], ego_pos[1], ego_pos[4]
    ego_yaw = np.radians(ego_yaw)

    v_y, v_x = v_loc[:2]  # center positions
    v_yaw = v_angles[1]
    v_yaw = np.radians(-v_yaw)
    v_extent_x, v_extent_y = v_extent[:2]

    vehicle_vertices = np.array([
        [v_extent_y, v_extent_x],
        [-v_extent_y, v_extent_x],
        [-v_extent_y, -v_extent_x],
        [v_extent_y, -v_extent_x]
    ])

    vehicle_R = np.array([
        [np.cos(v_yaw), -np.sin(v_yaw)],
        [np.sin(v_yaw), np.cos(v_yaw)]
    ])

    # rotate the vehicle vertices
    rotated_vehicle_vertices = np.dot(vehicle_vertices, vehicle_R.T)

    # translate the vehicle vertices
    translated_vehicle_vertices = rotated_vehicle_vertices + np.array([v_x, v_y])

    # Convert global vertices to ego frame
    ego_R = np.array([
        [np.cos(ego_yaw), -np.sin(ego_yaw)],
        [np.sin(ego_yaw), np.cos(ego_yaw)]
    ])

    # rotate the vehicle vertices
    rotated_vehicle_vertices_ego = np.dot(translated_vehicle_vertices - np.array([ego_x, ego_y]), ego_R.T)

    return rotated_vehicle_vertices_ego


def create_vehicle_box(v_loc, v_angles, v_extent, ego_pos, max_distance):
    # rotate the vehicle vertices
    rotated_vehicle_vertices_ego = transform_vehicle_to_ego_2d(v_loc, v_angles, v_extent, ego_pos)

    if np.all(np.abs(rotated_vehicle_vertices_ego) > max_distance):
        return None

    # Create a polygon
    polygon = patches.Polygon(rotated_vehicle_vertices_ego, closed=True, fill=True, edgecolor='white', facecolor='white')

    return polygon


def extract_individual_vehicles(bev_map):
    # Find contours of vehicles in the BEV map
    contours, _ = cv2.findContours(bev_map.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a list to store individual vehicle BEV maps
    vehicle_bevs = []

    # Iterate through contours
    for contour in contours:
        # Get bounding box coordinates of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create a new BEV map for the current vehicle
        vehicle_bev = np.zeros_like(bev_map)
        
        # Convert the numpy array to a cv::UMat object
        vehicle_bev_um = cv2.UMat(vehicle_bev)
        
        # Draw the bounding box of the vehicle on the new BEV map
        cv2.rectangle(vehicle_bev_um, (x, y), (x + w, y + h), 1, thickness=-1)
        
        # Copy the data back to the numpy array
        vehicle_bev = vehicle_bev_um.get()
        
        # Append the new BEV map to the list
        vehicle_bevs.append(torch.as_tensor(vehicle_bev))
    
    return vehicle_bevs



def create_bev(vehicles, t_ego_pos, bev_image_size, bev_width, bev_height):
    # create a blank image
    plt.close()
    fig, ax = plt.subplots(figsize=(bev_image_size/100, bev_image_size/100), dpi=100)
    # make fig and ax default to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Add plt limits
    ax.set_xlim(-bev_width/2, bev_width/2)
    ax.set_ylim(-bev_height/2, bev_height/2)

    ax.set_aspect('equal')
    ax.axis('off')

    visible_vehicles = dict()
    for vehicle_id in vehicles:
        v_loc = vehicles[vehicle_id]['location']
        v_angles = vehicles[vehicle_id]['angle']
        v_extent = vehicles[vehicle_id]['extent']

        polygon = create_vehicle_box(v_loc=v_loc, v_angles=v_angles, v_extent=v_extent, ego_pos=t_ego_pos, max_distance=bev_width/2)

        if polygon is not None:
            ax.add_patch(polygon)
            visible_vehicles[vehicle_id] = vehicles[vehicle_id]
    
    # fig bbox inches
    fig.tight_layout(pad=0)

    # Render the plot to an image buffer
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    # plt.close()
    # Reset buffer position to the start to read it
    buf.seek(0)
    # Read the image buffer as an array
    img = plt.imread(buf, format='RGBA')
    buf.close()
    # Convert to grayscale
    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    img = np.array(img * 255, dtype=np.uint8)
    # Convert image to binary
    binary_img = np.where(img > 0, 1, 0)

    # save image to disk
    # plt.imsave('bev.png', img, cmap='gray')

    return binary_img, visible_vehicles
