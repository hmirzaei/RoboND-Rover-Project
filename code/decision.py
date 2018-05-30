import numpy as np

from supporting_functions import a_star_search, update_map_vars
from perception import to_polar_coords
import sys
import time

# find closest point to p in points
def find_closest_point(p, points):
    points_p_diff = points - np.repeat(np.array([[p[0]],[p[1]]]),len(points[0]),1)
    min_dist_ind = np.argmin(sum(points_p_diff**2))
    return (points[0][min_dist_ind], points[1][min_dist_ind])

# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function

def decision_step(Rover):
    if Rover.mode == 'book_start':
        i, j = int(Rover.pos[1]), int(Rover.pos[0])
        Rover.start_pos_on_map = (i, j)
        Rover.mode = 'explore'

    if Rover.mode == 'done':
        Rover.brake = Rover.brake_set
        return Rover # We are done !!

    # Implement conditionals to decide what to do given perception data
    # Here you're all set up with some basic functionality but you'll need to
    # improve on this decision tree to do a good job of navigating autonomously!

    # Example:
    # Check if we have vision data to make decisions with
    if Rover.nav_angles is not None:
        print(Rover.mode)

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

        if len(Rover.rock_angles) > 0:
            Rover.mode = 'pick_rock'


        if Rover.mode == 'pick_rock':
            if len(Rover.rock_angles) == 0:
                Rover.explore_time = time.time()
                Rover.path = []
                Rover.explored_pixels = len(Rover.worldmap.nonzero()[0])
                Rover.max_vel = 2.0
                Rover.mode = 'explore'
            else:   
                # Set steering to average rock angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.rock_angles * 180/np.pi), -15, 15)
                if Rover.near_sample or abs(Rover.steer) == 15:
                    Rover.max_vel = 0
                else:
                    Rover.max_vel = 1

                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0

                if Rover.vel > Rover.max_vel + 0.1:
                    # Set throttle value to throttle setting
                    Rover.brake = Rover.brake_set
                else: # Else coast
                    Rover.brake = 0   


        
        if Rover.mode == 'obst_avoid':
            Rover.steer = 15          
            if Rover.vel > 0:
                # Set throttle value to throttle setting
                Rover.throttle = 0
                Rover.brake = Rover.brake_set
            else: # Else coast
                Rover.brake = 0

            if not Rover.near_obstacle:
                Rover.explore_time = time.time()
                Rover.path = []
                Rover.explored_pixels = len(Rover.worldmap.nonzero()[0])
                Rover.max_vel = 2.0
                Rover.mode = 'explore'

        # Check for Rover.mode status
        if Rover.mode == 'explore':
            Rover.border_points = []
            Rover.borders = []
            if len(Rover.nav_angles) >= Rover.stop_forward:  
                # If mode is forward, navigable terrain looks good 
                # and velocity is below max, then throttle 
                if Rover.vel < Rover.max_vel:
                    # Set throttle value to throttle setting
                    Rover.throttle = Rover.throttle_set
                else: # Else coast
                    Rover.throttle = 0
                Rover.brake = 0
                # Set steering to average angle clipped to the range +/- 15
                Rover.steer = np.clip(np.mean(Rover.nav_angles * 180/np.pi), -15, 15)
            # If there's a lack of navigable terrain pixels then go to 'obst_avoid' mode
            elif len(Rover.nav_angles) < Rover.stop_forward:
                    # Set mode to "obst_avoid" and hit the brakes!
                    Rover.throttle = 0
                    # Set brake to stored brake value
                    Rover.brake = Rover.brake_set
                    Rover.steer = 0
                    Rover.mode = 'obst_avoid'
            # check if the rover making progress after 5 seconds by counting
            # newly explored world map pixels
            if time.time() - Rover.explore_time > 5:
                explored_pixels = len(Rover.worldmap.nonzero()[0])
                if explored_pixels - Rover.explored_pixels < 10:
                    Rover.mode = 'init_path'
                else:
                    Rover.explored_pixels = explored_pixels
                    Rover.explore_time = time.time()


        if Rover.mode == 'init_path':
            update_map_vars(Rover)
            j, i = int(Rover.pos[0]), int(Rover.pos[1])
            start = find_closest_point((i, j), Rover.map_points)
            if (len(Rover.border_points) > 0):    
                closest_border_point = find_closest_point((i, j), Rover.border_points)
                goal = find_closest_point(closest_border_point, Rover.map_points)
                Rover.path = a_star_search(Rover.graph, start, goal)[::5]
                Rover.path.append(closest_border_point)
            else: #probably we are done searching and should return
                Rover.done_searching = True
                goal = find_closest_point(Rover.start_pos_on_map, Rover.map_points)
                Rover.path = a_star_search(Rover.graph, start, goal)[::5]
                Rover.path.append(Rover.start_pos_on_map)

            if len(Rover.path) > 0:
                Rover.path_counter = 0
                Rover.mode = 'follow_path'
                Rover.path_step_time  = time.time()
            else:
                # probably the map is falsely marked as navigable terrain. 
                # So change it obstacle and try another border point in the next iteration
                i, j = closest_border_point
                Rover.worldmap[-1:i+2, j-1:j+2] = [255, 0, 0]

        if Rover.mode == 'follow_path':
            p = Rover.path[Rover.path_counter]
            Rover.goal = p
            goal_dist, goal_angle = to_polar_coords(p[1] - Rover.pos[0], p[0] - Rover.pos[1]) 
            goal_angle *= 180 / np.pi 
            if goal_angle < 0: goal_angle += 360

            angle_diff = goal_angle - Rover.yaw
            if angle_diff >  180: angle_diff -= 360
            if angle_diff < -180: angle_diff += 360

            if (abs(angle_diff) < 70):
                thr = 0
                ang_candids = []
                ang_counts = np.zeros((80,))
                for i in (Rover.nav_angles + np.pi/2)//(np.pi/80):
                    ang_counts[int(i)] += 1
                ang_counts = (ang_counts > thr).astype(np.float)
                ang_counts = np.convolve(ang_counts, np.ones((5,)))[2:-2]

                for i, count in enumerate(ang_counts):
                    if ang_counts[i] == 5: 
                        ang = -np.pi/2 + i * np.pi/80 + np.pi/160
                        ang_candids.append(ang * 180 / np.pi)

                ang_candids = np.array(ang_candids)
                if len(ang_candids) == 0:
                    Rover.explore_time = time.time()
                    Rover.path = []
                    Rover.explored_pixels = len(Rover.worldmap.nonzero()[0])
                    Rover.max_vel = 2.0
                    Rover.mode = 'explore'

                ang_candids_cp = np.array(ang_candids)  
                ang_candids_cp -= angle_diff
                ang_candids_cp[ang_candids_cp > 180] -= 360
                ang_candids_cp[ang_candids_cp < -180] += 360
                Rover.steer = ang_candids[np.argmin(np.abs(ang_candids_cp))]

                if abs(Rover.steer) > 45:
                    Rover.max_vel = 0
                else:
                    Rover.max_vel = 2.0
            else:
                Rover.max_vel = 0
                Rover.steer = angle_diff

            Rover.steer=np.clip(Rover.steer, -15, 15)

            if Rover.vel < Rover.max_vel:
                # Set throttle value to throttle setting
                Rover.throttle = Rover.throttle_set
            else: # Else coast
                Rover.throttle = 0

            if Rover.vel > Rover.max_vel + 0.4:
                # Set throttle value to throttle setting
                Rover.brake = Rover.brake_set
            else: # Else coast
                Rover.brake = 0

            if goal_dist < 3:
                Rover.path_counter += 1
                Rover.path_step_time  = time.time()
                if Rover.path_counter == len(Rover.path):
                    if Rover.done_searching:
                        Rover.mode = 'done'
                    else:
                        Rover.explore_time = time.time()
                        Rover.path = []
                        Rover.explored_pixels = len(Rover.worldmap.nonzero()[0])
                        Rover.max_vel = 2.0
                        Rover.mode = 'explore'

            if time.time() - Rover.path_step_time > 10:
                Rover.mode = 'init_path'


    else:
        Rover.throttle = Rover.throttle_set
        Rover.steer = 0
        Rover.brake = 0
        
    # If in a state where want to pickup a rock send pickup command
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
    
    return Rover

