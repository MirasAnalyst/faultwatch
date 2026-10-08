from . import component_condition, event_detection, labeled_faults, power_plant, rollout, run_to_failure, steady_state

EXPERIMENTS = {"steady_state": steady_state.run, "run_to_failure": run_to_failure.run,
               "labeled_faults": labeled_faults.run, "event_detection": event_detection.run,
               "component_condition": component_condition.run, "power_plant": power_plant.run,
               "rollout": rollout.run}
