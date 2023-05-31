# bonfirehealth-ai
Bonfire Health Python AI

# Deployment
To deploy it on AWS Lambda, follow below steps.
1. Retrieve an authentication token and authenticate your Docker client to your registry.
```
aws ecr get-login-password --region ap-southeast-1 | docker login --username AWS --password-stdin 160177701951.dkr.ecr.ap-southeast-1.amazonaws.com
```
2. Build your Docker image using the following command.
```
docker build -t predict-vital-signs .
```
3. After the build completes, tag your image so you can push the image to this repository:
```
docker tag predict-vital-signs:latest 160177701951.dkr.ecr.ap-southeast-1.amazonaws.com/predict-vital-signs:latest
```
4. Run the following command to push this image to your newly created AWS repository:
```
docker push 160177701951.dkr.ecr.ap-southeast-1.amazonaws.com/predict-vital-signs:latest
```
5. Go to Lambda dashboard and deploy a function from this image.
6. After the fuctions is deployed, please add an environment variable named `VIDEO_BUCKET_NAME=bonfirehealthXXXXX-YYY` (X=randomly assigned numbers, Y=environment). We also have to change the memory size to 1024 MB.