Input Layer (Game State Representation)
│
├─ Card Embedding Layer (для представления карт)
│
├─ LSTM/Transformer Layer (для учета последовательности ходов)
│
├─ Dense Layers (для обработки состояния)
│
└─ Output Heads:
   ├─ Policy Head (вероятности действий)
   └─ Value Head (оценка позиции)