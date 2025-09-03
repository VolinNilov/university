DEFAULT_PARAMS = {
    'radius': 1.0,
    'gravity': 9.81,
    'friction_coeff': 0.2,
    'mass': 1.0,
    'angle_rad': 0.1,
    'vx0': 0.0,
    'vy0': 0.0,
    't_start': 0.0,
    't_end': 5.0,
    'dt_coarse': 0.05,
    'dt_fine': 0.005
}

STYLESHEET = """
QGroupBox {
    font-weight: bold;
    border: 1px solid gray;
    border-radius: 5px;
    margin-top: 1ex;
    padding-top: 10px;
}
QGroupBox::title {
    subline-position: -10px;
    padding: 0px 5px 0px 5px;
}
"""
