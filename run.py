import json
from data_analyzer import DataAnalyzer


if __name__ == '__main__':
    with open('config.json') as f:
        config = json.load(f)
        batch_size = config['batch_size']
        loss_function = config['loss_function']
        learning_rate = config['learning_rate']
        early_stop_patience = config['early_stop_patience']
        optimizer = config['optimizer']
        epochs = config['epochs']
        input_dim = config['input_dim']
        seq_len = config['seq_len']
        embedding_dim = config['embedding_dim']

    auto_encoder = DataAnalyzer(datapath='./dataset2',
                                optimizer=optimizer,
                                epochs=epochs,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                early_stop_patience=early_stop_patience,
                                loss_function=loss_function,
                                input_dim=input_dim,
                                seq_len=seq_len,
                                embedding_dim=embedding_dim)
    """
    try:
        print('Load Model..')
        auto_encoder.load_model(path='./model/stacked_conv_autoencoder.pkl')
    except:
        auto_encoder.train()
        pass
        """
    auto_encoder.train()
    auto_encoder.reconstruct(path='./reconstruction')








