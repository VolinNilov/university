DEFAULT_PARAMS = {
    "gravity": 9.81,
    "friction_coeff": 0.2,
    "mass": 1.0,
    "x0": 0.0,
    "y0": 4.0,
    "vx0": 0.0,
    "vy0": 0.0,
    "t_start": 0.0,
    "t_end": 5.0,
    "dt_fine": 0.005,
    "n_runs": 5,
    "noise_sigma": 0.02,
    "random_seed": 42,
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
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0px 5px 0px 5px;
}
"""
