import flwr as fl

def main():
    # Define the strategy (e.g., FedAvg)
    strategy = fl.server.strategy.FedAvg()

    # Start Flower server
    fl.server.start_server(server_address="localhost:8080", strategy=strategy)

if __name__ == "__main__":
    main()