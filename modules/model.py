import joblib
import torch
import numpy as np
import yaml

from model_definitions import EnhancedTFT, EnhancedNBeats

class ModelManager:
    def __init__(self, symbol):
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        input_size = config.get("input_size", 7)
        self.scaler = joblib.load("scaler.pkl")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tft_model = EnhancedTFT(input_size).to(self.device)
        self.tft_model.load_state_dict(torch.load("tft_model.pth", map_location=self.device))
        self.tft_model.eval()
        self.nbeats_model = EnhancedNBeats(input_size).to(self.device)
        self.nbeats_model.load_state_dict(torch.load("nbeats_model.pth", map_location=self.device))
        self.nbeats_model.eval()

    def predict(self, features):
        # 优化：集成NBeats ensemble。处理nan，加sigmoid。
        x = np.nan_to_num(features)[None, :]
        x = self.scaler.transform(x)
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            tft_pred = torch.sigmoid(self.tft_model(x_tensor.unsqueeze(1))).item()  # 调整输入
            nbeats_pred = torch.sigmoid(self.nbeats_model(x_tensor)).item()
        pred = (tft_pred + nbeats_pred) / 2
        return pred, pred