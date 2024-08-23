import flwr as fl
import torch
from lora_model import create_lora_model
from data import load_data  # Implement this function to load your data

class FlowerLoraClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, data):
        self.model = model
        self.tokenizer = tokenizer
        self.data = data

    def get_parameters(self):
        return [param.detach().cpu().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.dtype).to(param.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        for batch in self.data:
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        return self.get_parameters(), len(self.data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        loss = 0.0
        for batch in self.data:
            inputs = self.tokenizer(batch, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            loss += outputs.loss.item()

        return float(loss / len(self.data)), len(self.data), {}

if __name__ == "__main__":
    model, tokenizer = create_lora_model()
    data = load_data()  # Replace with your data loading function
    client = FlowerLoraClient(model, tokenizer, data)
    fl.client.start_numpy_client("localhost:8080", client)