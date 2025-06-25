import pccl

def main():
    communicator = pccl.Communicator(0)
    endpoint = communicator.export_endpoint()
    print(endpoint)

if __name__ == '__main__':
    main()