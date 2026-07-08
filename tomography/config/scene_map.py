from .scene import ScenePCD, SceneMap, SceneTrav


class SceneMap():
    pcd = ScenePCD()
    pcd.file_name = 'dshp.pcd'

    map = SceneMap()
    map.resolution = 0.10
    map.ground_h = -9.0
    map.slice_dh = 0.5

    trav = SceneTrav()
    trav.kernel_size = 5
    trav.interval_min = 0.50
    trav.interval_free = 0.6
    trav.slope_max = 1
    trav.step_max = 0.5
    trav.standable_ratio = 0.4
    trav.cost_barrier = 50.0
    trav.safe_margin = 0.15
    trav.inflation = 0.15
