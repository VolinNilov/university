# (0,4) по варианту; dt_coarse чуть крупнее 0.05 — иначе кривые Хьюна/RK4 слипаются
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
    "dt_coarse": 0.11,
    "dt_fine": 0.005,
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
