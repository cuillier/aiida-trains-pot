"""Utility to generate LAMMPS MD configuration files."""


def generate_lammps_md_config(
    temperatures = [0],
    pressures = [0],
    steps = [0],
    styles = ['nve'],
    dt=0.001,
    ttime=100,
    ptime=1000,
):
    """Generate a YAML-like configuration for a set of parameters.

    Parameters:
        temperatures (list): A list of temperatures.
        pressures (list): A list of pressures.
        steps (list): A list of max number of steps for the integration.
        dt (float): Timestep of simulation. This parameter is used for thermostat parameters.
            A Nose-Hoover thermostat will not work well for arbitrary values of Tdamp.
            If Tdamp is too small, the temperature can fluctuate wildly; if it is too large,
            the temperature will take a very long time to equilibrate.
            A good choice for many models is a Tdamp of around 100 timesteps.
            A Nose-Hoover barostat will not work well for arbitrary values of Pdamp.
            If Pdamp is too small, the pressure and volume can fluctuate wildly;
            if it is too large, the pressure will take a very long time to equilibrate.
            A good choice for many models is a Pdamp of around 1000 timesteps.
        styles (list): A list of integration styles (e.g., "npt", "nvt", ...).
        ttime (float): Thermostat time constant, in multiples of dt
        ptime (float): Barostat time constant, in multiples of dt

    Returns:
        str: A YAML-formatted string of the configuration.
    """
    config = []
    for temp in temperatures:
        for press in pressures:
            for step in steps:
                for style in styles:
                    constraint = {
                        "temp": [temp, temp, ttime * dt],
                        "x": [press, press, ptime * dt],
                        "y": [press, press, ptime * dt],
                        "z": [press, press, ptime * dt],
                    }
                    md_block = {
                        "max_number_steps": step,
                        "velocity": [{"create": {"temp": temp}}],
                        "integration": {"style": style, "constraints": constraint},
                    }
                    config.append(md_block)

    return config
