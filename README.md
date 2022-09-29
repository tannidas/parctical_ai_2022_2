# parctical_ai_2022_2
This branch is for week 4 assignment on open source federated learning assigment
> 1. FedScale
> 2. FedBranch


**FedScale Testing**

Used dataset: FEMNIST

Steps to reproduce results:
> 1. Run fedscale_demo_server.py in one terminal
> 2. Run fedscale_demo_client.py in one terminal
> 3. Logs are found in .logs/demo_job/logs/aggregator/logs output file

`Correction fedscale_demo_server/client.py`

1. `from fedscale.core.logger.execution import args` will be chnaged to `from fedscale.core.config_parser import args`



