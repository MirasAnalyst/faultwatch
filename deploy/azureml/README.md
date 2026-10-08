# Azure ML deployment

Not deployed from this repo (no subscription). These files follow the Azure ML v2 YAML schemas.

```bash
# register the MLflow model logged by run.py --mlflow (or point MLFLOW_TRACKING_URI at the workspace)
az ml model create --name faultwatch-turbofan_cmapss_fd001 --type mlflow_model \
    --path mlruns/1/models/<model-id>/artifacts
az ml online-endpoint create -f deploy/azureml/endpoint.yml
az ml online-deployment create -f deploy/azureml/deployment.yml --all-traffic
az ml online-endpoint invoke -n faultwatch-gt-fleet --request-file sample-request.json
```

`deployment-custom.yml` + `score.py` is the alternative when the scoring container needs
more than the MLflow pyfunc (for example to return the safety-risk register as well).
Blue/green: deploy `green` at 0% traffic, compare its scores with `blue` on mirrored
traffic, then shift.
