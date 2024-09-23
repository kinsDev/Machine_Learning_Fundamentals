import os
import io
import boto3
import json
import base64
import csv

# grab environment variables
ENDPOINT_NAME = "latest-endpoint"
runtime= boto3.client('runtime.sagemaker')

def classify(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    event = event["body"]

    # Decode the image data
    image = base64.b64decode(event["image_data"])

    payload = json.dumps(event)

    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='image/png',
                                       Body=image)
    print(response)
    result = json.loads(response['Body'].read().decode())
    event["inferences"] = result
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }