import torch, torch.nn as nn

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.logT = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        T = torch.exp(self.logT) + 1e-8
        return logits / T

def fit_temperature(model, val_loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    scaler = TemperatureScaler().to(device)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)

    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            lg = model(xb)
            logits.append(lg.detach().cpu())
            labels.append(yb.detach().cpu())
    logits = torch.cat(logits, 0).to(device)
    labels = torch.cat(labels, 0).to(device)

    def closure():
        opt.zero_grad()
        loss = crit(scaler(logits), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return scaler
