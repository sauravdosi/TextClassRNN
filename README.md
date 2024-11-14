# TextClassRNN

# TextClassRNN

A text classification project implementing various neural network architectures (RNN, LSTM, FFNN) for sentiment analysis on review data. The models are built using PyTorch and can classify reviews into 5 star ratings (1-5).

## Models Implemented

- **RNN (Recurrent Neural Network)**: Basic RNN implementation with tanh activation
- **LSTM (Long Short-Term Memory)**: Advanced RNN architecture with better gradient flow
- **FFNN (Feed-Forward Neural Network)**: Simple baseline model

## Requirements

All dependencies are listed in requirements.txt and can be installed using:

```bash
pip install -r requirements.txt
```

## Key Features

- Multiple neural network architectures for comparison
- GPU support for faster training
- Word embeddings using GloVe
- Comprehensive evaluation metrics
- Data visualization tools for analysis

## Project Structure

- `rnn.py`: Main RNN implementation
- `lstm.py`: LSTM model implementation
- `ffnn.py`: Feed-forward neural network implementation
- `eda.py`: Exploratory data analysis and visualization
- `analysis.py`: Training results analysis
- `rnn_gpu.py`: GPU-optimized RNN implementation

## Usage

```bash
python rnn.py --train_data path/to/training.json --val_data path/to/validation.json --hidden_dim 20 --epochs 30
```


## Results

The models are evaluated on both training and validation sets, with metrics including:
- Loss
- Training accuracy
- Validation accuracy

Results are saved to CSV files for further analysis and visualization.