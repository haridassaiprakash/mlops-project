from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model
from threading import Thread
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.model import InferenceConfig
from azureml.core.environment import Environment
from azureml.core.webservice import AciWebservice
from azureml.core.runconfig import DockerConfiguration

ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws = Workspace.from_config(auth=ia)
print(ws)

docker_config = DockerConfiguration(use_docker=True)

print('creating the environment ...')
myenv = Environment.from_conda_specification(name='env', file_path='env.yml')

myenv.docker.base_image = 'amlwsacr.azurecr.io/azureml/azureml_413fe919029ed8c77aa5c2bd986ec719:latest'
myenv.inferencing_stack_version='latest'
inference_config = InferenceConfig(entry_script='score.py', environment=myenv)
print('environment created!')

print('deploying the ACI service ...')
aci_config = AciWebservice.deploy_configuration(
                  cpu_cores=1,
                  memory_gb=1, auth_enabled=False
                  )


aci_service_name='diabetes-aci'

model1 = Model(ws, "diabetes_model.pkl")

aci_service = Model.deploy(ws, aci_service_name, [model1], inference_config, aci_config)

aci_service.wait_for_deployment(show_output=True)
print('ACI service deployed!')
print(aci_service.state)