from . import run_to_failure, steady_state

EXPERIMENTS = {"steady_state": steady_state.run, "run_to_failure": run_to_failure.run}
