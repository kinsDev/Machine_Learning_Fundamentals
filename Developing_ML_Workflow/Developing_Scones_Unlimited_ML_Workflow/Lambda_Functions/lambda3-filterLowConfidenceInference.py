import json

THRESHOLD = 0.90

def lambda_handler(event, context):
    # Parse the event body
    event = json.loads(event["body"])

    print(event)

    # Grab the inferences from the event
    inferences = event.get("inferences", [])

    # Ensure that there are at least two inferences
    if not inferences or len(inferences) < 2:
        raise ValueError("Inferences data is missing or invalid")

    # Check if any values in our inferences are above the THRESHOLD
    meets_threshold = (float(inferences[0]) >= THRESHOLD) or (float(inferences[1]) >= THRESHOLD)

    # If our threshold is met, pass the data back out of the Step Function
    if meets_threshold:
        pass
    else:
        # Raise an error if the threshold is not met
        raise ValueError("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }