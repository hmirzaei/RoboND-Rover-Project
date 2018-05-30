import numpy as np
import cv2
from skimage.morphology import opening

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Identify pixels of the rock
def rock_color_thresh(img, rgb_thresh=(110,70,50)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] < rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


dst_size = 5
bottom_offset = 6
# Define a function to perform a perspective transform
# I've used the example grid image above to choose source points for the
# grid cell in front of the rover (each grid cell is 1 square meter in the sim)
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    mask = cv2.warpPerspective(np.ones_like(img)*255, M, (img.shape[1], img.shape[0]))# keep same size as input image
    xv, yv = np.meshgrid(range(img.shape[1]), range(img.shape[0]))
    radius_lim = (xv - img.shape[1]/2)**2 + (yv - img.shape[0] + bottom_offset)**2 < 100**2
    if len(img.shape) > 2:
        for i in range(3): mask[:,:,i] = mask[:,:,i] * radius_lim.astype(np.float)
    else:
        mask = mask * radius_lim.astype(np.float)
    
    return warped, mask



# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform
    # 2) Apply perspective transform
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image

    # 5) Convert map image pixel values to rover-centric coords
    # 6) Convert rover-centric pixel values to world coordinates
    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # 8) Convert rover-centric pixel positions to polar coordinates
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    img = Rover.img
    xpos = Rover.pos[0]
    ypos = Rover.pos[1]
    yaw = Rover.yaw

    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])

    threshed = color_thresh(img, rgb_thresh=(160, 160, 160))
    opened = opening(threshed)
    warped, mask = perspect_transform(img, source, destination)
    thr_warped, mask1 = perspect_transform(opened, source, destination)
    threshed = mask1 * thr_warped

    rock_threshed = rock_color_thresh(warped, rgb_thresh=(110, 110, 50))

    xpix, ypix = rover_coords(threshed)
    distances, angles = to_polar_coords(xpix, ypix) # Convert to polar coords
    Rover.nav_dists = distances
    Rover.nav_angles = angles

    rock_xpix, rock_ypix = rover_coords(rock_threshed)
    distances, angles = to_polar_coords(rock_xpix, rock_ypix) # Convert to polar coords
    Rover.rock_dists = distances
    Rover.rock_angles = angles

    abs_tilt = 360 - Rover.pitch if Rover.pitch > 180 else Rover.pitch
    if abs_tilt < 0.5:
        obs = mask[:,:,0] * (np.ones_like(threshed, np.int) * 255 - threshed);
        obs_xpix, obs_ypix = rover_coords(obs)
        obs_distances, obs_angles = to_polar_coords(obs_xpix, obs_ypix) # Convert to polar coords
        Rover.obs_dists = obs_distances
        Rover.obs_angles = obs_angles

        scale = 10
        x_world, y_world = pix_to_world(xpix, ypix, xpos, 
                                    ypos, yaw, 
                                    Rover.worldmap.shape[0], scale)
        x_obs, y_obs = pix_to_world(obs_xpix, obs_ypix, xpos, 
                                    ypos, yaw, 
                                    Rover.worldmap.shape[0], scale)
        x_rock, y_rock = pix_to_world(rock_xpix, rock_ypix, xpos, 
                                    ypos, yaw, 
                                    Rover.worldmap.shape[0], scale)
        Rover.worldmap[y_world, x_world, 2] = 255
        Rover.worldmap[y_obs, x_obs, 0] = 255
        nav = Rover.worldmap[:, :, 2]  > 0
        Rover.worldmap[nav, 0] = 0
        Rover.worldmap[y_rock, x_rock, 1] = 255
                

        Rover.vision_image[:,:,2] = threshed;
        Rover.vision_image[:,:,1] = rock_threshed * 255;
    else:
        Rover.vision_image[:,:,:] = 0;
    

    return Rover