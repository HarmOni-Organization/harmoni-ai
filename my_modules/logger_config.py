"""
Logger Configuration Module

This module provides a centralized configuration for application logging,
with different settings for production and development environments.
"""

import os
import logging
import logging.handlers
from flask import request, has_request_context
import json
from datetime import datetime

class RequestFormatter(logging.Formatter):
    """
    Custom formatter that adds request-specific information to log records when available.
    """
    
    def format(self, record):
        if has_request_context():
            record.url = request.url
            record.remote_addr = request.remote_addr
            record.method = request.method
            record.path = request.path
            
            # Add request headers and params if needed
            if not hasattr(record, 'request_data'):
                record.request_data = {
                    'params': dict(request.args),
                    'headers': {k: v for k, v in request.headers.items() 
                               if k.lower() not in ('authorization', 'cookie')}  # Exclude sensitive headers
                }
        else:
            record.url = None
            record.remote_addr = None
            record.method = None
            record.path = None
            
        return super().format(record)

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    
    def format(self, record):
        log_record = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add request context if available
        if hasattr(record, 'remote_addr') and record.remote_addr is not None:
            log_record.update({
                'url': record.url,
                'remote_addr': record.remote_addr,
                'method': record.method,
                'path': record.path,
            })
            
        # Add extra fields from the record
        if hasattr(record, 'request_data'):
            log_record['request_data'] = record.request_data
            
        # Include any extra attributes added in the log call
        for key, value in record.__dict__.items():
            if key not in ('args', 'asctime', 'created', 'exc_info', 'exc_text', 
                          'filename', 'funcName', 'id', 'levelname', 'levelno', 
                          'lineno', 'module', 'msecs', 'message', 'msg', 'name', 
                          'pathname', 'process', 'processName', 'relativeCreated', 
                          'stack_info', 'thread', 'threadName', 'request_data',
                          'url', 'remote_addr', 'method', 'path'):
                log_record[key] = value
                
        # Include exception info if available
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logging(app):
    """
    Configure application logging based on environment settings.
    
    Args:
        app: Flask application instance
    """
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    # Get configuration from environment
    log_level_name = os.getenv('LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    debug_mode = os.getenv('DEBUG', 'False').lower() in ('true', '1', 't')
    
    # Set the root logger level
    root_logger.setLevel(log_level)
    
    # Configure console handler for all environments
    console_handler = logging.StreamHandler()
    
    # Configure file handler for production
    if not debug_mode:
        # Use JSON formatter in production for structured logging
        console_handler.setFormatter(JsonFormatter())
        
        # Add rotating file handler for production
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'app.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=int(os.getenv('LOG_MAX_BYTES', 10485760)),  # Default 10MB
            backupCount=int(os.getenv('LOG_BACKUP_COUNT', 5))
        )
        file_handler.setFormatter(JsonFormatter())
        root_logger.addHandler(file_handler)
    else:
        # Use more readable format for development
        formatter = RequestFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | [%(remote_addr)s] %(method)s %(path)s | %(message)s'
        )
        console_handler.setFormatter(formatter)
    
    # Add console handler to root logger
    root_logger.addHandler(console_handler)
    
    # Configure Flask's logger
    app.logger.handlers = []
    app.logger.propagate = True
    
    # Ensure that third-party libraries don't override our configuration
    for logger_name in ('werkzeug', 'gunicorn.error', 'gunicorn.access'):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.propagate = True
        
    # Log startup information
    app.logger.info(f"Application logging configured. Level: {log_level_name}, Debug mode: {debug_mode}")