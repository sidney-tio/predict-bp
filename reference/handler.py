import json
import os
import time
from pathlib import Path

import boto3
from core.model import load_model
from core.utils import input_utils, processing
from settings import config

BUCKET_NAME = os.environ.get('VIDEO_BUCKET_NAME')
s3_client = boto3.client('s3')
start_time_model = time.time()
print(f'🔥 0 of 5: Load model ({config.model.name}).')
model = load_model(config.model.name, config.model.weights_path)
print("--- %s seconds ---" % (time.time() - start_time_model))


def handler(event, context):
    """
    """
    print('🔥 1 of 5: Retrieve request data from event body object.')
    start_time = time.time()
    # Retrieve video file from S3
    if 'body' in event:
        request_data = json.loads(event['body'])
    else:
        request_data = event

    input_url = None
    kwargs = {}
    is_delete_video = True
    if 'is_delete_video' in request_data:
        is_delete_video = request_data['is_delete_video']
    if 'video_url' in request_data:
        input_url = request_data['video_url']
    elif 'images_url' in request_data:
        input_url = request_data['images_url']
        fps = request_data.get('fps')
        if fps is None:
            print(f'❌ 5 of 5: fps not provided.')
            raise Exception(
                'You must provide fps when the input type is `images_url`')
        kwargs = {'fps': request_data['fps']}

    if input_url is None:
        print(f'❌ 5 of 5: input_url not valid.')
        return {
            'status': 'failed',
            'data': [], 'message': 'You must provide input_url',
            'event': json.dumps(event),
            'event_type': str(type(event))
        }

    print(
        f'🔥 2 of 5: Download video ({input_url}) from {BUCKET_NAME} S3 bucket.')
    video_content = s3_client.get_object(
        Bucket=BUCKET_NAME, Key=input_url
    )['Body'].read()
    print("--- %s seconds ---" % (time.time() - start_time))

    filename = Path(input_url).name
    print(f'🔥 3 of 5: Preprocess video ({input_url}) from {filename}.')
    inputs = input_utils.preprocess_video(video_content, filename, **kwargs)
    print("--- %s seconds ---" % (time.time() - start_time))

    print(f'🔥 4 of 5: Predict vital signs using model ({config.model.name}).')
    calculator = processing.VitalSignsCalculator(model)
    print("--- %s seconds ---" % (time.time() - start_time))
    inputs['user_info'] = request_data.get('user_info')
    try:
        result = calculator.predict(inputs)
        print(f'🔥 5 of 5: Results: {result}')

        print(f'🔥 5 of 5: Function completed.')
        print("--- %s seconds ---" % (time.time() - start_time))
        if is_delete_video:
            s3_client.delete_object(Bucket=BUCKET_NAME, Key=input_url)
        return {
            'status': 'success',
            'data': [{
                'bpm': result.bpm,
                'hrv': result.hrv,
                'si': result.si,
                'sns_index': result.sns_index,
                'sp': result.sp,
                'dp': result.dp,
                'resp_rate': result.resp_rate,
                'spo2': result.spo2,
                'o2': result.o2,
            }],
            'message': 'Predicted successfully'
        }
    except Exception as e:
        print(f'❌ 5 of 5: Function not completed.')
        print("--- %s seconds ---" % (time.time() - start_time))
        return {
            'status': 'failed',
            'data': [],
            'message': f'Got an error while predicting: {e}'
        }
