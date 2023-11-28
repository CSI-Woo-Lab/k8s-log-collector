kubectl delete jobs --all
for p in $(kubectl get pods | grep Terminating | awk '{print $1}'); do kubectl delete pod $p --grace-period=0 --force;done

rm -rf _tmp_job_*