from .scene import ScenePCD, SceneMap, SceneTrav


class SceneMap():
    pcd = ScenePCD()
    pcd.file_name = 'map.pcd'

    map = SceneMap()
    map.resolution = 0.15
    map.ground_h = 0.0
    map.slice_dh = 0.5

    trav = SceneTrav()
    trav.kernel_size = 5
    trav.interval_min = 0.50
    trav.interval_free = 0.6
    trav.slope_max = 1
    trav.step_max = 0.7
    trav.standable_ratio = 0.4
    trav.cost_barrier = 50.0
    trav.safe_margin = 0.1
    trav.inflation = 0.05
