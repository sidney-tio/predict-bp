import os
import json
from pathlib import Path
import boto3

from core.model import load_model
from core.utils import input_utils, processing
from settings import config


BUCKET_NAME = os.environ.get('VIDEO_BUCKET_NAME')
s3_client = boto3.client('s3')


def handler(event, context):
    """
    """
    # Retrieve video file from S3
    if 'body' in event:
        request_data = json.loads(event['body'])
    else:
        request_data = event

    input_url = None
    kwargs = {}
    if 'video_url' in request_data:
        input_url = request_data['video_url']
    elif 'images_url' in request_data:
        input_url = request_data['images_url']
        fps = request_data.get('fps')
        if fps is None:
            raise Exception(
                'You must provide fps when the input type is `images_url`')
        kwargs = {'fps': request_data['fps']}

    if input_url is None:
        return {
            'status': 'failed',
            'data': [], 'message': 'You must provide input_url',
            'event': json.dumps(event),
            'event_type': str(type(event))
        }

    # video_content = s3_client.get_object(
    #     Bucket=BUCKET_NAME, Key=input_url
    # )['Body'].read()
    video_content = open(input_url, 'rb').read()

    # Preprocess the given video
    filename = Path(input_url).name
    inputs = input_utils.preprocess_video(video_content, filename, **kwargs)

    # Load model
    model = load_model(config.model.name, config.model.weights_path)

    # Predict
    calculator = processing.VitalSignsCalculator(model)
    inputs['user_info'] = request_data.get('user_info')
    try:
        result = calculator.predict(inputs)
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
        return {
            'status': 'failed',
            'data': [],
            'message': f'Got an error while predicting: {e}'
        }


event = {'images_url': '2209161046364010972.mp4', 'user_info': {
    'age': 40, 'height': 177, 'weight': 72, 'gender': 'male'}, 'fps': 30}
print(handler(event, None))
