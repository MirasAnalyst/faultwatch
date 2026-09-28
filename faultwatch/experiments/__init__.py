from . import event_detection, labeled_faults, run_to_failure, steady_state

EXPERIMENTS = {"steady_state": steady_state.run, "run_to_failure": run_to_failure.run,
               "labeled_faults": labeled_faults.run, "event_detection": event_detection.run}
