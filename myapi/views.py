import json

from django.core import serializers
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from myapi.MLModel import predictor


@csrf_exempt  # To exempt from default requirement for CSRF tokens to use postman
def TheModelView(request):
    if (request.method == "POST"):
        # Turn the body into a dict
        body = json.loads(request.body.decode("utf-8"))
        symptoms = body['symptoms']

        # Turn the object to json to dict, put in array to avoid non-iterable error
        data = predictor(symptoms)
        # send json response with new object
        return JsonResponse(data, safe=False)
