from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model
from threading import Thread
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice, AksWebservice
from azureml.core.runconfig import DockerConfiguration

ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws = Workspace.from_config(auth=ia)


aks_name = 'diabetes-aks-gpu'
try:
  aks_target = ComputeTarget(workspace=ws, name=aks_name)
  print("Found existing compute target")
except ComputeTargetException:
  print("Creating a new compute target...")

  prov_config = AksCompute.provisioning_configuration(vm_size='Standard_A2_v2')

  aks_target = ComputeTarget.create(
      workspace=ws, name=aks_name, provisioning_configuration=prov_config 
  )

  aks_target.wait_for_completion(show_output=True)


docker_config = DockerConfiguration(use_docker=True)

myenv = Environment.from_conda_specification(name='env', file_path='env.yml')

myenv.docker.base_image = 'amlwsacr.azurecr.io/azureml/azureml_413fe919029ed8c77aa5c2bd986ec719:latest'
myenv.inferencing_stack_version='latest'
inference_config = InferenceConfig(entry_script='score.py', environment=myenv)

# deploying for specific configuration
# in below code, didn't use autoscaling feature for that
#  please refer: https://docs.microsoft.com/en-us/azure/machine-learning/v1/how-to-deploy-azure-kubernetes-service?tabs=python#autoscaling
aks_config = AksWebservice.deploy_configuration(replica_max_concurrent_requests=500, auth_enabled=False,
                  autoscale_enabled=True, 
                  autoscale_target_utilization=30,
                  autoscale_min_replicas=1,
                  autoscale_max_replicas=4
                  )


aks_service_name='diabetes-aks'

model1 = Model(ws, "diabetes_model.pkl")

aks_service = Model.deploy(ws,
                        name=aks_service_name,
                        models=[model1],
                        deployment_config=aks_config,
                        deployment_target=aks_target,
                        inference_config=inference_config
                        )
 
aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)
print(aks_service.get_logs())
