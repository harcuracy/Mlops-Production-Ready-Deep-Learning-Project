# Mlops-Production-Ready-Deep-Learning-Project

## Workflows


1. Update config.yaml
2. Update params.yaml
3. Update the entity in src
4. Update the configuration manager in src config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the dvc.yaml

```bash
uv venv --python 3.10
```

```bash
.venv\Scripts\activate
```

```bash
uv pip install -r requirements.txt
```
## create iam user

#with specific access

1. EC2 access : It is virtual machine

2. ECR: Elastic Container registry to save your docker image in aws


#Description: About the deployment

1. Build docker image of the source code

2. Push your docker image to ECR

3. Launch Your EC2 

4. Pull Your image from ECR in EC2

5. Lauch your docker image in EC2

#Policy:

1. AmazonEC2ContainerRegistryFullAccess

2. AmazonEC2FullAccess