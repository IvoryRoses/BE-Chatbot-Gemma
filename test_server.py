from fastapi.testclient import TestClient
from server import app
import pytest
import time
import multiprocessing
import uvicorn

client = TestClient(app)

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to complete")
        return result
    return wrapper

@timer_decorator
def test_chat_endpoint_response():
    """Test if chat endpoint returns valid response"""
    test_prompt = "What is 2+2?"
    response = client.post("/chat/", json={"prompt": test_prompt})
    
    # Test status code
    assert response.status_code == 200
    
    # Test response structure
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert len(data["response"]) > 0

@timer_decorator
def test_chat_endpoint_empty_prompt():
    """Test handling of empty prompt"""
    response = client.post("/chat/", json={"prompt": ""})
    assert response.status_code == 200

@timer_decorator
def test_chat_endpoint_invalid_request():
    """Test handling of invalid request"""
    response = client.post("/chat/", json={})
    assert response.status_code == 422

@timer_decorator
def test_model_loading():
    """Test if model loads correctly"""
    response = client.post("/chat/", json={"prompt": "Test"})
    assert response.status_code == 200

def run_server():
    """Function to run the FastAPI server"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="critical")

if __name__ == "__main__":
    print("Starting test suite...")
    
    # Start server in separate process
    server_process = multiprocessing.Process(target=run_server)
    server_process.start()
    
    # Wait for server to initialize
    print("Waiting for server to start...")
    time.sleep(2)
    
    try:
        print("Running tests...")
        start_total = time.time()
        
        test_chat_endpoint_response()
        test_chat_endpoint_empty_prompt()
        test_chat_endpoint_invalid_request()
        test_model_loading()
        
        end_total = time.time()
        print(f"\nTotal test execution time: {end_total - start_total:.2f} seconds")
        
    finally:
        # Cleanup: Stop the server
        print("Shutting down server...")
        server_process.terminate()
        server_process.join()
        print("Test suite completed.")