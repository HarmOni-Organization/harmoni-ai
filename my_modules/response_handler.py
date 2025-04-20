"""
Response Handler Module

This module provides utilities for creating standardized API responses
across the application to ensure consistency in response format.
"""

import time
import logging
from flask import jsonify

# Get the logger
logger = logging.getLogger(__name__)

def create_response(status, message, data=None, status_code=200, start_time=None, request_info=None):
    """
    Creates a standardized API response.
    
    Args:
        status (bool): Indicates success or failure of the request.
        message (str): A message describing the response.
        data (dict, optional): The data to be included in the response. Defaults to None.
        status_code (int, optional): HTTP status code. Defaults to 200.
        start_time (float, optional): The time when request processing started. 
                                     Used to calculate response time. Defaults to None.
        request_info (dict, optional): Additional request information for logging. Defaults to None.
    
    Returns:
        tuple: A tuple containing the JSON response and the HTTP status code.
    """
    # Prepare response payload
    response_payload = {
        "status": status,
        "message": message
    }
    
    # Add data to payload if provided
    if data is not None:
        response_payload["data"] = data
    
    # Create JSON response
    response = jsonify(response_payload)
    response.status_code = status_code
    
    # Log response details if start_time is provided
    if start_time is not None:
        end_time = time.time()
        # Convert response time to milliseconds
        response_time_ms = (end_time - start_time) * 1000
        
        log_extra = {
            "status_code": status_code,
            "response_time_ms": response_time_ms
        }
        
        # Add request info to log if provided
        if request_info:
            log_extra.update(request_info)
        
        logger.info(
            f"Response Sent - Status Code:[{status_code}] - Response Time: {response_time_ms:.2f}ms",
            extra=log_extra
        )
    
    return response, status_code