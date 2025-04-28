import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
GPT로 생성한 베이스 코드 참고하여 이해 해보고자 함
LSTM과 Attention Mechanism에 대해 공부 필요
pytorch 공부 필요
'''

class TSPDataset(Dataset):
    def __init__(self, n_samples, n_cities):
        self.n_samples = n_samples
        self.n_cities = n_cities

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cities = torch.rand(self.n_cities, 2)
        target = torch.randperm(self.n_cities)
        return cities, target


# -------------------------
# Encoder: 양방향 LSTM으로 입력(도시 좌표) 임베딩
# Decoder: LSTMCell과 Attention 메커니즘을 이용하여 다음 방문할 도시 인덱스를 예측
# -------------------------
class PointerNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers=1):
        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 입력 임베딩 레이어
        self.embedding = nn.Linear(input_dim, embedding_dim)
        # Encoder: 양방향 LSTM (출력 차원 = hidden_dim*2)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        # Decoder: LSTMCell (입력은 임베딩 차원, 상태 크기는 hidden_dim*2)
        self.decoder = nn.LSTMCell(embedding_dim, hidden_dim * 2)

        # Attention
        self.W1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.W2 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.vt = nn.Linear(hidden_dim * 2, 1)

        # decoder의 초기 입력
        self.decoder_input0 = nn.Parameter(torch.FloatTensor(embedding_dim))
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs, targets=None):
        batch_size, seq_len, _ = inputs.size()
        embedded = self.embedding(inputs)

        encoder_outputs, _ = self.encoder(embedded)

        decoder_hidden = torch.zeros(batch_size, self.hidden_dim * 2, device=inputs.device)
        decoder_cell = torch.zeros(batch_size, self.hidden_dim * 2, device=inputs.device)
        # decoder의 첫 입력: 학습 가능한 초기 벡터 (모든 배치에 동일)
        decoder_input = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # 방문한 도시는 True
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=inputs.device)
        pointers = []  # 예측한 도시 인덱스 기록
        log_probs = []  # 각 단계의 로그 확률 기록

        # 각 decoding step마다 도시 하나씩 선택 (총 seq_len번)
        for _ in range(seq_len):
            decoder_hidden, decoder_cell = self.decoder(decoder_input, (decoder_hidden, decoder_cell))
            # decoder_hidden을 모든 encoder step에 대해 확장
            dec_hidden_exp = decoder_hidden.unsqueeze(1).expand(-1, seq_len, -1)
            # attention
            energy = self.vt(torch.tanh(self.W1(encoder_outputs) + self.W2(dec_hidden_exp))).squeeze(
                -1)  # (batch, seq_len)
            # 이미 방문한 도시는 -inf로 처리
            energy = energy.masked_fill(mask, float('-inf'))
            # softmax를 통해 (log) 확률 분포 계산
            log_p = F.log_softmax(energy, dim=1)  # (batch, seq_len)
            # greedy 방식으로 다음 도시 선택
            _, idx = log_p.max(dim=1)  # (batch,)
            pointers.append(idx)
            log_probs.append(log_p.gather(1, idx.unsqueeze(1)))
            # 선택된 도시를 방문 처리
            mask = mask.clone()
            mask[range(batch_size), idx] = True

            # 다음 decoder 입력으로 선택된 도시의 임베딩 사용
            decoder_input = embedded[range(batch_size), idx, :]

        # pointers: (batch, seq_len)
        pointers = torch.stack(pointers, dim=1)
        log_probs = torch.cat(log_probs, dim=1)  # (batch, seq_len)
        return pointers, log_probs


# -------------------------
# 간단한 학습 루프 (teacher forcing 없이 greedy decoding을 사용함)
# 실제 학습 시에는 teacher forcing이나 REINFORCE와 같은 기법을 적용해야 함.
# -------------------------
def train(model, dataloader, optimizer, n_epochs, device):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch  # inputs: (batch, seq_len, 2), targets: (batch, seq_len)
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            # 여기서는 teacher forcing 없이 모델의 greedy decoding 결과를 얻음
            pointers, log_probs = model(inputs)
            # 간단하게 타겟과 예측이 일치하는지 여부에 대해 negative log likelihood를 계산
            # (실제 학습에서는 타겟 순열에 따른 cross entropy loss 계산 등 정교한 방식 필요)
            loss = 0.0
            # 각 디코딩 스텝마다 예측 로그 확률을 타겟 인덱스와 비교하여 loss를 누적
            for i in range(targets.size(1)):
                loss += F.nll_loss(log_probs[:, i], targets[:, i])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# -------------------------
# 추론 및 평가 함수
# -------------------------
def evaluate(model, tsp_instance, device):
    model.eval()
    with torch.no_grad():
        # tsp_instance: (seq_len, 2) -> (1, seq_len, 2)
        inputs = tsp_instance.unsqueeze(0).to(device)
        pointers, _ = model(inputs)
        return pointers.squeeze(0).cpu().numpy()


def tour_length(cities, tour):
    # cities: numpy array of shape (n_cities, 2)
    # tour: 순서대로 방문하는 도시 인덱스 (배열)
    n = len(tour)
    length = 0.0
    for i in range(n):
        city1 = cities[tour[i]]
        city2 = cities[tour[(i + 1) % n]]
        length += np.linalg.norm(city1 - city2)
    return length


# -------------------------
# 메인 실행 코드
# -------------------------
if __name__ == '__main__':
    # 설정값
    n_samples = 1000  # 학습 샘플 수
    n_cities = 10  # 각 TSP 인스턴스의 도시 수
    batch_size = 128
    n_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 및 DataLoader 준비
    dataset = TSPDataset(n_samples, n_cities)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델, 옵티마이저 초기화
    model = PointerNet(input_dim=2, embedding_dim=128, hidden_dim=256, n_layers=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습
    train(model, dataloader, optimizer, n_epochs, device)

    # 새로운 TSP 인스턴스에 대해 추론
    tsp_instance, target = dataset[0]
    tsp_instance_np = tsp_instance.numpy()
    st = time.time()
    predicted_tour = evaluate(model, tsp_instance, device)
    print(f"예측하는 데에 걸리는 시간: {time.time()-st}")
    print("Predicted tour (도시 인덱스 순서):", predicted_tour)
    print("Tour length:", tour_length(tsp_instance_np, predicted_tour))
