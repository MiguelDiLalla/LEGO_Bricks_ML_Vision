import logging
from train import setup_logging, export_logs

def test_export_logs():
    # Setup logging
    setup_logging("test_session")
    
    # Create some test logs
    logging.info("Test log entry 1")
    logging.info("Test log entry 2")
    
    # Test both parameter variations
    try:
        # Test with log_name parameter
        result1 = export_logs(log_name="test_session")
        print(f"Test 1 (log_name): {'✅ Passed' if result1 else '❌ Failed'}")
        
        # Test with default parameters
        result2 = export_logs()
        print(f"Test 2 (default): {'✅ Passed' if result2 else '❌ Failed'}")
        
    except TypeError as e:
        print(f"❌ Test failed with error: {e}")

if __name__ == "__main__":
    test_export_logs()