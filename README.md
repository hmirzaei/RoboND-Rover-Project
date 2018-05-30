[//]: # (Image References)
[image_0]: ./misc/rover_image.jpg
[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)
# Project: Search and Sample Return

[auto_gif]: ./output/auto_nav.gif
![auto_gif][auto_gif]

### Notebook Analysis
The notebook contains following element that are used in the perception step of the rover navigation:

#### Perspective Transform

The following image image is used to calculate the camera projection transformation:

[grid_image]: ./calibration_images/example_grid1.jpg
![grid_image][grid_image]

The coordinates of the four grid square corners are specified in the source and destination (top view) images and are applied to the OpenCV warpPersepective function.

Also, a mask is defined in order to represent the valid pixels in the warped image. A radius limitation is applied to the mask because the coordinates of the pixels that are close to the horizon will be inaccurate when transformed to the world coordinate system.

```python
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
```

Following images shows the resulting warped and mask images for the [above example grid image][grid_image].

[warped]: ./misc/warped.jpg
[mask]: ./misc/mask.jpg
![warped][warped]    ![mask][mask]

#### Color Thresholding

Three different types are used in the perception step:

- Terrain thresholding: The pixel brighter that specific value are considered to represent navigable terrain. So are all the RGB components should be greater than some threshold.

- Obstacle thresholding: The inverse image of the Terrain pixels within the perspective mask is considered to be the Obstacles.

- Rock thresholding: rocks are gold colored. So the yellowish pixels should be detected. Therefore, to detect a rock pixel, red and green components should be greater than some thresholds and the blue component should be less than a threshold.

```python
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
```

Following images show the result of above thresholding steps on a sample image:


[rock_image]: ../calibration_images/example_rock1.jpg
[rock_threshold]: ./misc/rock_threshold.jpg
[nav_threshold]: ./misc/nav_threshold.jpg
[obs_threshold]: ./misc/obs_threshold.jpg

| ![rock_image][rock_image]			| ![rock_threshold][rock_threshold] |
|:---------------------------------:|:---------------------------------:|
| Original Image 					| Rock thresholding 				|
| ![nav_threshold][nav_threshold]   | ![obs_threshold][obs_threshold]   |   
| Terrain Thresholding 				| Obstacle Thresholding				|

#### Removing false terrain pixels

Some pixels on the mountains might pass color thresholding and be recognized as terrain pixels. It causes the map to be inaccurate if the rover gets very close to the mountains. To filter out those pixels, an opening morphological filter is applied.

```python
from skimage.morphology import opening
opened = opening(nav_threshed)
```
Following images show an example:

[opening]: ./misc/opening.jpg
![opening][opening] 

#### Coordinate Transformations

The coordinate transformation step consists of following transformations:

- Image coordinates to rover coordinates: to do this the x and y coordinates should be swapped and a translation from the bottom center of image to the origin of rover coordinates should be performed.

- Rover coordinates to World Coordinate: which consists of a rotation with the rover's yaw angle and a translation to the rover world coordinates.

- Cartesian to Polar coordinate conversion: to find the average angle of navigable terrain. Also, polar data is used to calculate the angles and distances of rocks and obstacles.

#### Preview of the output video

[output_gif]: ./output/test_mapping.gif
![output_gif][output_gif]

### Autonomous Navigation and Mapping

#### Perception Step

The perception step is mainly the methods described in Notebook Analysis section. In every sample time, if the pitch angle of the rover is very close to zero the world map is updated. The pitch angle check is done because the perspective projection is not valid in the case that the pitch angle is not zero.

#### Decision Step

In decision step, the rover operates in one of the following modes:

- 'explore': In this mode the rover follows the most navigable terrain by setting the steering angle to the average navigable terrain angle (same as what is provided initially). However, if the rover does not make progress in 5 seconds, e.g. it is stuck in a loop or following some previously discovered area, the mode is changed to 'init_path'

- 'init_path': In this mode, a path from current rover's location to some border point (which is defined later on) is assigned to the rover in order to approach unexplored areas. The mode changed to 'follow_path' afterward.

- 'follow_path': In this mode, the rover follows the assigned path step by step. In each step, the steering angle is set to the closest navigable terrain angle to the direction of current path point. If there is not navigable terrain angle, the mode is changed to 'obst_avoid'. If the rover's location is close enough to the path point, the rover proceeds toward the next path point.

- 'obst_avoid': This mode has similar function as the initially provided 'stop' mode. If the rover is close to an obstacle or mountains, the mode is changed to this mode to avoid obstacle. In this mode, the speed is set to zero and rover turns counter-clockwise until there is no obstacle detected ahead. the mode is then changed to 'explore' because 'explore' mode helps the rover to find a navigable path to get far from the obstacle.

In the following sections, each computational element used in the decision step is explained:

#### Detecting Obstacles

To detect that the rover is close to an obstacle, the minimum obstacle distances are calculated in some angle range ahead (40 degrees). This 40 degrees range is divided into to 40 bins (each 1 degree). If the minimum obstacle distance is less than some threshold in 15 bins out of 40 bins, obstacle is detected and the mode is changed to 'obst_avoid'.

```python
obst_angle_range = 40   # angle range to look for obstacles
num_angle_bins = 40     # number of angle bins for obstacle detection 
obst_avoid_dist = 20    # the distance of obstacles to avoid
obst_avoid_thr  = 15    # if the number of bins with close obstacles are greater that this threshold 
                        # the obstacle should be avoided

# init all bins to an arbitrary large value
obst_dist = np.ones((num_angle_bins,)) * 10000 
# find minimum distance of the obstacles in each angle bin
for i, ang in enumerate(Rover.obs_angles):
    ang *= 180 / np.pi
    if -obst_angle_range/2 <= ang < obst_angle_range/2:
        obst_dist[int((ang + obst_angle_range / 2) / obst_angle_range * num_angle_bins)] =\
            min (obst_dist[int((ang + obst_angle_range / 2) / obst_angle_range * num_angle_bins)], Rover.obs_dists[i])

# check if the number of angle bins with close obstacles are larger that the threshold
Rover.near_obstacle = (np.sum(obst_dist < obst_avoid_dist) > obst_avoid_thr)
# if we are not already in 'obst_avoid' mode change the mode to 'obst_avoid' if obstacle is detected
if Rover.mode != 'obst_avoid':
    if Rover.near_obstacle and Rover.vel > 0:
        Rover.mode = 'obst_avoid'		
```

#### Finding border points

Border pixels are defined as the pixels in the world map that belong to the navigable terrain and have some unexplored pixel in their vicinity. These pixels are marked with cyan color in the world map view. Please note that the areas where the obstacle boundaries are not 'thick' enough labeled as border pixels as well. This helps the rover to get close to the mountains and other obstacles to find the rocks with higher probability.

[border_img]: ./misc/borders.png
![border_img][border_img]

After finding all the border pixels, all the connected border pixels (border blobs) are found using depth first search algorithm. Next, the centers of border blobs are calculated and defined as 'border points'. 

[border_points_img]: ./misc/border_points.png
![border_points_img][border_points_img]


```python
 grid = np.zeros(Rover.worldmap.shape[:2], np.int)
# go through the world map pixels and find navigable pixels with unexplored pixels in their vicinity
for i in range(10, Rover.worldmap.shape[0] - 10):
    for j in range(10, Rover.worldmap.shape[1] - 10):
        if Rover.worldmap[i,j,2] == 255 and Rover.worldmap[i,j,0] == 0:
            for m in range(i-3, i+3):
                for n in range(j-3, j+3):
                    if all(Rover.worldmap[m, n, :] == [0, 0, 0]):
                        grid[i,j] = 1

#find all border point confidantes for display purposes only
Rover.borders = list(zip(*[list(e) for e in grid.nonzero()]))
#find all the separated border blobs by depth first search algorithm
borders = find_borders(grid)
#calculate the center point of each border blob
border_points = [np.mean([*zip(*b)],1).astype(np.int) for b in borders]
Rover.border_points = np.array(border_points).T
```

#### Motion Planning

To find a path from current rover's location to a border point (or initial rover point when the task is finished), A* algorithm is employed to find a path through the navigable map. To make sure that the path has a safe distance from the boundaries, the skeleton of the navigable map is found and the A* search is performed on the skeleton map.

[nav_map]: ./misc/nav_map.png
[skel_map]: ./misc/nav_skel.png
![nav_map][nav_map]   &nbsp;&nbsp; &rightarrow; &nbsp;&nbsp;  ![skel_map][skel_map]


```python
from skimage.morphology import skeletonize

nav_map = (Rover.worldmap[:,:, 2] > 0).astype(np.uint8)
skel = skeletonize(nav_map)
Rover.graph = Graph(skel) 
Rover.map_points = skel.nonzero()
```

After finding the skeleton map, the closest points on the skeleton to the current location and the border point are found and the A* search is ran on them as start and the goal points. The path points are down-sampled by a factor of 5 to smooth out the path for the path planning step.

[path_image]: ./misc/path.png
![path_image][path_image]  

```python
j, i = int(Rover.pos[0]), int(Rover.pos[1])
start = find_closest_point((i, j), Rover.map_points)
closest_border_point = find_closest_point((i, j), Rover.border_points)
goal = find_closest_point(closest_border_point, Rover.map_points)
Rover.path = a_star_search(Rover.graph, start, goal)[::5]
Rover.path.append(closest_border_point)
```


#### Results

Using the proposed techniques, the rover is able to pick up all 6 samples, explore 98% of the map with 69% fidelity and return to initial point in 18 minutes with no manual intervention. A video of the autonomous mode is provided output folder. However, different runs might lead to different results since the autonomous navigation algorithm is not perfect, e.g. it might miss some samples in tricky positions, get stuck in some rocks and need manual assistance or fail to pick up some located samples.

Following are the simulation settings:
- Resolution: 1280x720 
- Quality: Fantastic
- Frame rate: 21

#### Further steps to improve the autonomous navigation:

- The rover's search of the environment takes a long time to complete. Addressing the following issues helps to improve the speed and reliability of the autonomous navigation.
- Modifying the sample pick up routine to make it faster and more reliable. Currently, it is assumed that there is only one sample in the camera view which might not be the case and causes some troubles.
- Design a more advanced motion planning to speed up the search especially around the center of the map where the terrain path is wide and currently it takes a lot of rover's time to cover that area.
- Design a more effective path tracking algorithm to address frequent stops and slow navigating along a planned path.
- Improving obstacle avoidance to make it more reliable
- Adding more decision states to help the rover recover from dead locks. 








