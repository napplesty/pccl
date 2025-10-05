#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pccl.api import create_collective_api

def main():
    print("PCCL Collective Communication Example")
    
    device_ids = [0, 1, 2, 3]
    data_size_gb = 1.0
    
    api = create_collective_api("standard")
    
    try:
        print("Executing AllReduce...")
        result = api.allreduce(device_ids, data_size_gb)
        
        if result.success:
            print(f"AllReduce completed successfully in {result.execution_time_ms:.2f}ms")
            print(f"Results: {result.results}")
        else:
            print(f"AllReduce failed: {result.error}")
        
        print("\nExecuting Broadcast...")
        result = api.broadcast(0, device_ids, data_size_gb)
        
        if result.success:
            print(f"Broadcast completed successfully in {result.execution_time_ms:.2f}ms")
        else:
            print(f"Broadcast failed: {result.error}")
    
    finally:
        api.shutdown()
        print("API shutdown completed")

if __name__ == "__main__":
    main()
