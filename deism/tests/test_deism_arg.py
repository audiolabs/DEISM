import numpy as np
from deism.core_deism import *
from deism.data_loader import *


def test_deism_arg_simulation():
    """Test DEISM-ARG simulation with impedance update."""
    deism = DEISM("RIR", "convex")
    # Testing impedance update
    # Example of room volumn and roomAreas
    roomVolumn = 36
    roomAreas = np.array([9, 10, 9, 10, 12, np.sqrt(10) * 4])
    deism.update_room(roomVolumn=roomVolumn, roomAreas=roomAreas)
    deism.update_wall_materials()
    deism.update_wall_materials(
        np.array([[100, 100, 100, 100, 100, 100], [100, 100, 100, 100, 100, 100]]).T,
        np.array([10, 20]),
        "impedance",
    )
    deism.update_freqs()
    deism.update_source_receiver()
    deism.update_directivities()
    deism.run_DEISM()
    pressure = deism.params["RTF"]

    # Basic assertions to verify the simulation ran successfully
    assert pressure is not None, "Pressure should not be None"
    assert len(pressure) > 0, "Pressure should have data"
    assert deism.params["posSource"] is not None, "Source position should be set"
    assert deism.params["posReceiver"] is not None, "Receiver position should be set"
    assert deism.params["freqs"] is not None, "Frequencies should be set"

    print("Simulation done!")


# -------------------------------------------------------
if __name__ == "__main__":
    test_deism_arg_simulation()
