#!/bin/bash

ECR_IMAGE_NAME=predict-vital-signs
ECR_URL=160177701951.dkr.ecr.ap-southeast-1.amazonaws.com
ECR_USERNAME=AWS
ECR_REGION=ap-southeast-1

echo "👷 Logging to AWS Elastic Container Registry..."
aws ecr get-login-password --region $ECR_REGION | docker login --username $ECR_USERNAME --password-stdin $ECR_URL

echo "👷 Building $ECR_IMAGE_NAME image..."
docker build -t $ECR_IMAGE_NAME .

echo "👷 Tagging $ECR_IMAGE_NAME image with $ECR_URL/$ECR_IMAGE_NAME.."
docker tag $ECR_IMAGE_NAME:latest $ECR_URL/$ECR_IMAGE_NAME:latest

echo "👷 Pushing $ECR_URL/$ECR_IMAGE_NAME.."
docker push $ECR_URL/$ECR_IMAGE_NAME:latest
