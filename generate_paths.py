# @title Setup and Imports { display-mode: "form" }
# @markdown (double click to see the code)

import math
import os
import random

import git
import imageio
import magnum as mn
import numpy as np

# %matplotlib inline
from matplotlib import pyplot as plt

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

repo = git.Repo(".", search_parent_directories=True)



# @title Define Observation Display Utility Function { display-mode: "form" }

# @markdown A convenient function that displays sensor observations with matplotlib.

# @markdown (double click to see the code)


# Change to do something like this maybe: https://stackoverflow.com/a/41432704
def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([]),id = 0):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    # plt.show(block=False)
    plt.savefig(f'vis_{id}.png')

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return
# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None,filename = None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    # plt.show(block=False)
    if filename is not None:
        plt.savefig(filename)




###################################################How to Take your ~~Dragon~~ [Agent] for a (Random) Walk

def render_video(data_path, glb_file, scene, fps = 4, max_seed = 1, dataset_type = 'scannet'):
    
    # @title Configure Sim Settings
    test_scene = os.path.join(
        data_path, glb_file
    )

    mp3d_scene_dataset ="./mp3d.scene_dataset_config.json"

    rgb_sensor = True  # @param {type:"boolean"}
    depth_sensor = True  # @param {type:"boolean"}
    semantic_sensor = True  # @param {type:"boolean"}

    sim_settings = {
        # "width": 256,  # Spatial resolution of the observations
        # "height": 256,
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "scene": test_scene,  # Scene path
        "scene_dataset": mp3d_scene_dataset,  # the scene dataset configuration files
        "default_agent": 0,
        "sensor_height": 1.5,  # Height of sensors in meters
        "color_sensor": rgb_sensor,  # RGB sensor
        "depth_sensor": depth_sensor,  # Depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }


 
    cfg = make_cfg(sim_settings)
    # Needed to handle out of order cell run in Jupyter
    try:  # Got to make initialization idiot proof
        sim.close()
    except NameError:
        pass
    sim = habitat_sim.Simulator(cfg)

    # Print semantic annotation information (id, category, bounding box details)
    # about levels, regions and objects in a hierarchical fashion
    # scene = sim.semantic_scene
    # print_scene_recur(scene)

    # the randomness is needed when choosing the actions
    # random.seed(sim_settings["seed"])
    # sim.seed(sim_settings["seed"])
    for seed in range(1,max_seed):
        dir_path = repo.working_tree_dir
        output_dir = f'{dataset_type}/'+scene+f'/seed_{seed}'
        output_directory = os.path.join(
            dir_path, 'generated_paths',output_dir
        )

        if os.path.exists(output_directory):
            print(f'output directory {output_directory} already exists, skip this seed {seed}')
            continue

        output_path = os.path.join(dir_path, output_directory)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        random.seed(seed)
        sim.seed(seed)

        # Set agent state
        agent = sim.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
        agent.set_state(agent_state)

        # Get agent state
        agent_state = agent.get_state()
        print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

        # total_frames = 0
        # action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())

        # max_frames = 5

        


        # @markdown ###Configure Example Parameters:
        # @markdown Configure the map resolution:
        meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}
        # @markdown ---
        # @markdown Customize the map slice height (global y coordinate):
        custom_height = False  # @param {type:"boolean"}
        height = 1  # @param {type:"slider", min:-10, max:10, step:0.1}
        # @markdown If not using custom height, default to scene lower limit.
        # @markdown (Cell output provides scene height range from bounding box for reference.)

        print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
        if not custom_height:
            # get bounding box minimum elevation for automatic height
            height = sim.pathfinder.get_bounds()[0][1]

        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            # @markdown You can get the topdown map directly from the Habitat-sim API with *PathFinder.get_topdown_view*.
            # This map is a 2D boolean array
            sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

            if display:
                # @markdown Alternatively, you can process the map using the Habitat-Lab [maps module](https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/maps.py)
                hablab_topdown_map = maps.get_topdown_map(
                    sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                )
                recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
                hablab_topdown_map = recolor_map[hablab_topdown_map]
                print("Displaying the raw map from get_topdown_view:")
                display_map(sim_topdown_map)
                print("Displaying the map from the Habitat-Lab maps module:")
                display_map(hablab_topdown_map)

                # easily save a map to file:
                map_filename = os.path.join(output_path, "top_down_map.png")
                imageio.imsave(map_filename, hablab_topdown_map)


        # @markdown ## Querying the NavMesh

        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            # @markdown NavMesh area and bounding box can be queried via *navigable_area* and *get_bounds* respectively.
            print("NavMesh area = " + str(sim.pathfinder.navigable_area))
            print("Bounds = " + str(sim.pathfinder.get_bounds()))

            # @markdown A random point on the NavMesh can be queried with *get_random_navigable_point*.
            pathfinder_seed = 1  # @param {type:"integer"}
            sim.pathfinder.seed(pathfinder_seed)
            nav_point = sim.pathfinder.get_random_navigable_point()
            print("Random navigable point : " + str(nav_point))
            print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

            # @markdown The radius of the minimum containing circle (with vertex centroid origin) for the isolated navigable island of a point can be queried with *island_radius*.
            # @markdown This is analogous to the size of the point's connected component and can be used to check that a queried navigable point is on an interesting surface (e.g. the floor), rather than a small surface (e.g. a table-top).
            print("Nav island radius : " + str(sim.pathfinder.island_radius(nav_point)))

            # @markdown The closest boundary point can also be queried (within some radius).
            max_search_radius = 2.0  # @param {type:"number"}
            print(
                "Distance to obstacle: "
                + str(sim.pathfinder.distance_to_closest_obstacle(nav_point, max_search_radius))
            )
            hit_record = sim.pathfinder.closest_obstacle_surface_point(
                nav_point, max_search_radius
            )
            print("Closest obstacle HitRecord:")
            print(" point: " + str(hit_record.hit_pos))
            print(" normal: " + str(hit_record.hit_normal))
            print(" distance: " + str(hit_record.hit_dist))

            vis_points = [nav_point]

            # HitRecord will have infinite distance if no valid point was found:
            if math.isinf(hit_record.hit_dist):
                print("No obstacle found within search radius.")
            else:
                # @markdown Points near the boundary or above the NavMesh can be snapped onto it.
                perturbed_point = hit_record.hit_pos - hit_record.hit_normal * 0.2
                print("Perturbed point : " + str(perturbed_point))
                print(
                    "Is point navigable? " + str(sim.pathfinder.is_navigable(perturbed_point))
                )
                snapped_point = sim.pathfinder.snap_point(perturbed_point)
                print("Snapped point : " + str(snapped_point))
                print("Is point navigable? " + str(sim.pathfinder.is_navigable(snapped_point)))
                vis_points.append(snapped_point)

            # @markdown ---
            # @markdown ### Visualization
            # @markdown Running this cell generates a topdown visualization of the NavMesh with sampled points overlaid.
            meters_per_pixel = 0.1  # @param {type:"slider", min:0.01, max:1.0, step:0.01}

            if display:
                xy_vis_points = convert_points_to_topdown(
                    sim.pathfinder, vis_points, meters_per_pixel
                )
                # use the y coordinate of the sampled nav_point for the map height slice
                top_down_map = maps.get_topdown_map(
                    sim.pathfinder, height=nav_point[1], meters_per_pixel=meters_per_pixel
                )
                recolor_map = np.array(
                    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                )
                top_down_map = recolor_map[top_down_map]
                print("\nDisplay the map with key_point overlay:")
                display_map(top_down_map, key_points=xy_vis_points)

        navmesh_settings = habitat_sim.NavMeshSettings()

        # @markdown Choose Habitat-sim defaults (e.g. for point-nav tasks), or custom settings.
        use_custom_settings = False  # @param {type:"boolean"}
        sim.navmesh_visualization = True  # @param {type:"boolean"}
        navmesh_settings.set_defaults()
        if use_custom_settings:
            # fmt: off
            #@markdown ---
            #@markdown ## Configure custom settings (if use_custom_settings):
            #@markdown Configure the following NavMeshSettings for customized NavMesh recomputation.
            #@markdown **Voxelization parameters**:
            navmesh_settings.cell_size = 0.05 #@param {type:"slider", min:0.01, max:0.2, step:0.01}
            #default = 0.05
            navmesh_settings.cell_height = 0.2 #@param {type:"slider", min:0.01, max:0.4, step:0.01}
            #default = 0.2

            #@markdown **Agent parameters**:
            navmesh_settings.agent_height = 1.5 #@param {type:"slider", min:0.01, max:3.0, step:0.01}
            #default = 1.5
            navmesh_settings.agent_radius = 0.1 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
            #default = 0.1
            navmesh_settings.agent_max_climb = 0.2 #@param {type:"slider", min:0.01, max:0.5, step:0.01}
            #default = 0.2
            navmesh_settings.agent_max_slope = 45 #@param {type:"slider", min:0, max:85, step:1.0}
            # default = 45.0
            # fmt: on
            # @markdown **Navigable area filtering options**:
            navmesh_settings.filter_low_hanging_obstacles = True  # @param {type:"boolean"}
            # default = True
            navmesh_settings.filter_ledge_spans = True  # @param {type:"boolean"}
            # default = True
            navmesh_settings.filter_walkable_low_height_spans = True  # @param {type:"boolean"}
            # default = True

            # fmt: off
            #@markdown **Detail mesh generation parameters**:
            #@markdown For more details on the effects
            navmesh_settings.region_min_size = 20 #@param {type:"slider", min:0, max:50, step:1}
            #default = 20
            navmesh_settings.region_merge_size = 20 #@param {type:"slider", min:0, max:50, step:1}
            #default = 20
            navmesh_settings.edge_max_len = 12.0 #@param {type:"slider", min:0, max:50, step:1}
            #default = 12.0
            navmesh_settings.edge_max_error = 1.3 #@param {type:"slider", min:0, max:5, step:0.1}
            #default = 1.3
            navmesh_settings.verts_per_poly = 6.0 #@param {type:"slider", min:3, max:6, step:1}
            #default = 6.0
            navmesh_settings.detail_sample_dist = 6.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
            #default = 6.0
            navmesh_settings.detail_sample_max_error = 1.0 #@param {type:"slider", min:0, max:10.0, step:0.1}
            # default = 1.0
            # fmt: on

            # @markdown **Include STATIC Objects**:
            # @markdown Optionally include all instanced RigidObjects with STATIC MotionType as NavMesh constraints.
            navmesh_settings.include_static_objects = True  # @param {type:"boolean"}
            # default = False

        navmesh_success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)

        if not navmesh_success:
            print("Failed to build the navmesh! Try different parameters?")

        if sim.pathfinder.is_loaded:
            navmesh_save_path = os.path.join(data_path, "test_saving.navmesh") #@param {type:"string"}
            sim.pathfinder.save_nav_mesh(navmesh_save_path)
            print('Saved NavMesh to "' + navmesh_save_path + '"')
            sim.pathfinder.load_nav_mesh(navmesh_save_path)


        # @markdown ## Pathfinding Queries on NavMesh

        # @markdown The shortest path between valid points on the NavMesh can be queried as shown in this example.

        
        # @markdown With a valid PathFinder instance:
        if not sim.pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            seed = seed  # @param {type:"integer"}
            sim.pathfinder.seed(seed)

            # fmt off
            # @markdown 1. Sample valid points on the NavMesh for agent spawn location and pathfinding goal.
            # fmt on
            sample1 = sim.pathfinder.get_random_navigable_point()
            sample2 = sim.pathfinder.get_random_navigable_point()

            # @markdown 2. Use ShortestPath module to compute path between samples.
            path = habitat_sim.ShortestPath()
            path.requested_start = sample1
            path.requested_end = sample2
            found_path = sim.pathfinder.find_path(path)
            geodesic_distance = path.geodesic_distance
            path_points = path.points
            # @markdown - Success, geodesic path length, and 3D points can be queried.
            print("found_path : " + str(found_path))
            print("geodesic_distance : " + str(geodesic_distance))
            print("path_points : " + str(path_points))

            # @markdown 3. Display trajectory (if found) on a topdown map of ground floor
            if found_path:
                meters_per_pixel = 0.025
                scene_bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
                height = scene_bb.y().min
                if display:
                    top_down_map = maps.get_topdown_map(
                        sim.pathfinder, height, meters_per_pixel=meters_per_pixel
                    )
                    recolor_map = np.array(
                        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
                    )
                    top_down_map = recolor_map[top_down_map]
                    grid_dimensions = (top_down_map.shape[0], top_down_map.shape[1])
                    # convert world trajectory points to maps module grid points
                    trajectory = [
                        maps.to_grid(
                            path_point[2],
                            path_point[0],
                            grid_dimensions,
                            pathfinder=sim.pathfinder,
                        )
                        for path_point in path_points
                    ]
                    grid_tangent = mn.Vector2(
                        trajectory[1][1] - trajectory[0][1], trajectory[1][0] - trajectory[0][0]
                    )
                    path_initial_tangent = grid_tangent / grid_tangent.length()
                    initial_angle = math.atan2(path_initial_tangent[0], path_initial_tangent[1])
                    # draw the agent and trajectory on the map
                    maps.draw_path(top_down_map, trajectory)
                    maps.draw_agent(
                        top_down_map, trajectory[0], initial_angle, agent_radius_px=8
                    )
                    print("\nDisplay the map with agent and path overlay:")
                    try:
                        display_map(top_down_map,filename = output_directory+'/agent_path_overlay.png')
                    except:
                        pass

                # @markdown 4. (optional) Place agent and render images at trajectory points (if found).
                display_path_agent_renders = True  # @param{type:"boolean"}
                if display_path_agent_renders:
                    observations = []
                    Poses = []
                    print("Rendering observations at path points:")
                    # tangent = path_points[1] - path_points[0]
                    def linear_interpolation(point1, point2, num_intermediate_points):
                        return np.linspace(point1, point2, num=num_intermediate_points + 2)
                    def insert_intermediate_points(path_points, num_intermediate_points):
                        all_points = []
                        for i in range(len(path_points) - 1):
                            point1 = path_points[i]
                            point2 = path_points[i + 1]
                            interpolated_points = linear_interpolation(point1, point2, num_intermediate_points)
                            # all_points.extend(interpolated_points[:-1])
                            # all_points.append(path_points[-1])
                            # all_points = all_points + [point1]
                            all_points.extend(interpolated_points[:-1])
                            # all_points = all_points + [point2]
                        return all_points

                    path_points_inter = insert_intermediate_points(path_points,3)

                    agent_state = habitat_sim.AgentState()
                    for ix, point in enumerate(path_points_inter):
                        if ix < len(path_points_inter) - 1:
                            tangent = path_points_inter[ix + 1] - point
                            agent_state.position = point
                            tangent_orientation_matrix = mn.Matrix4.look_at(
                                point, point + tangent, np.array([0, 1.0, 0])
                            )
                            # print(f'id is {ix}\n point is {point}\n tangent_orientation_matrix is {tangent_orientation_matrix},\ntangent is {tangent}')
                            tangent_orientation_q = mn.Quaternion.from_matrix(
                                tangent_orientation_matrix.rotation()
                            )
                            agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                            agent.set_state(agent_state)

                            observation = sim.get_sensor_observations()
                            rgb = observation["color_sensor"]
                            semantic = observation["semantic_sensor"]
                            depth = observation["depth_sensor"]

                            # if display:
                            #     display_sample(rgb, semantic, depth,id = ix)
                            # import pdb;pdb.set_trace()
                            sensor_state = sim._Simulator__last_state[0].sensor_states
                            rgb_sensor = sensor_state['color_sensor']
                            sensor_data = [agent_state.position.flatten(),agent_state.rotation.components.astype(np.float32).flatten(),rgb_sensor.position.flatten(),rgb_sensor.rotation.components.astype(np.float32).flatten()]
                            sensor_data = np.concatenate(sensor_data)

                            # agent_state.position, agent_state.rotation, rgb_sensor.position, rgb_sensor.rotation

                            Poses.append(sensor_data)

                            observations.append(observation)

                Poses = np.stack(Poses)

                np.savetxt(output_directory+f'/poses.txt',Poses)

                if do_make_video:
                    # fps = 4
                # use the video utility to render the observations
                    vut.make_video(
                        observations=observations,
                        primary_obs="color_sensor",
                        primary_obs_type="color",
                        video_file=output_directory + f"/shortest_path_render",
                        fps=fps,
                        open_vid=show_video,
                    )
                    vut.make_video(
                        observations=observations,
                        primary_obs="depth_sensor",
                        primary_obs_type="depth",
                        video_file=output_directory + f"/depth_shortest_path_render",
                        fps=fps,
                        open_vid=show_video,
                    )
    sim.close()
    return 


### visualization setting
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--no-make-video", dest="make_video", action="store_false")
    parser.set_defaults(show_video=True, make_video=True)
    args, _ = parser.parse_known_args()

    show_video = False
    display = args.display
    # do_make_video = args.make_video
    do_make_video = False
else:
    show_video = False
    do_make_video = False
    display = False

# import the maps module alone for topdown mapping
if display:
    from habitat.utils.visualizations import maps



import os
import argparse
import multiprocessing as mp

def process_scene(args_tuple):
    scene, dataset_path, assimp_path, dataset_type, max_seed = args_tuple

    glb_path = os.path.join('./glb_files', dataset_type, scene)
    os.makedirs(glb_path, exist_ok=True)

    glb_filename = os.path.join(glb_path, f'{scene}.glb')

    if dataset_type == 'scannet':
        data_path = os.path.join(dataset_path, 'scans', scene)
        ply_filename = os.path.join(data_path, f'{scene}_vh_clean.ply')
    elif dataset_type == 'scannetpp':
        data_path = os.path.join(dataset_path, 'data', scene, 'scans')
        ply_filename = os.path.join(data_path, f'mesh_aligned_0.05.ply')
    else:  # arkitscene
        data_path = os.path.join(dataset_path, './raw/Training/', scene)
        ply_filename = os.path.join(data_path, f'{scene}_3dod_mesh.ply')

    if os.path.exists(glb_filename):
        print(f'have already processed this scene {scene}')
    else:
        # transfer ply → glb
        os.system(f'{assimp_path} export "{ply_filename}" "{glb_filename}"')

    if not os.path.exists(glb_filename):
        print(f'can not process this scene {scene}')
        return

    print(f'The glb file is {glb_filename}. ')
    try:
        render_video(data_path='./', glb_file=glb_filename, scene= scene, fps=4, max_seed=max_seed, dataset_type=dataset_type)
    except Exception as e:
        print(f'can not process this scene {scene}, error: {e}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--assimp_path', type=str, required=True)
    parser.add_argument('--dataset_type', type=str, default='scannet', help='scannet, scannetpp, or arkitscene')
    parser.add_argument('--num_workers', type=int, default=4, help='number of parallel processes')
    args = parser.parse_args()

    dataset_path = args.dataset_path
    assimp_path = args.assimp_path
    dataset_type = args.dataset_type

    if dataset_type == 'scannet':
        scenes = sorted(os.listdir(os.path.join(dataset_path, 'scans')))
    elif dataset_type == 'scannetpp':
        scenes = sorted(os.listdir(os.path.join(dataset_path, 'data')))
    elif dataset_type == 'arkitscene':
        scenes = sorted(os.listdir(os.path.join(dataset_path, './raw/Training/')))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # test for the first 2 scenes
    scenes = scenes[:2]

    max_seed = 30
    tasks = [(scene, dataset_path, assimp_path, dataset_type, max_seed) for scene in scenes]

    with mp.Pool(processes=args.num_workers) as pool:
        pool.map(process_scene, tasks)

if __name__ == "__main__":
    main()