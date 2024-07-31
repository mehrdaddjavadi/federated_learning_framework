# federated_learning_framework/decorators.py
import functools

def federated_learning_decorator(uri):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            central_server = kwargs.get('central_server')
            client = kwargs.get('client')
            await central_server.run_server()
            await client.connect_to_central_server(uri)
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def encryption_decorator(context):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            client = kwargs.get('client')
            client.context = context
            return await func(*args, **kwargs)
        return wrapper
    return decorator
