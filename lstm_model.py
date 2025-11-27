"""
LSTM Model for Question-Answer Matching
Trains an LSTM model on chunks for better retrieval
"""

import os
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


class LSTMModel:
    """LSTM model for question-answer matching and similarity"""
    
    def __init__(self, embedding_dim: int = 128, lstm_units: int = 64, max_length: int = 200):
        """
        Initialize LSTM model
        
        Args:
            embedding_dim: Dimension of input embeddings
            lstm_units: Number of LSTM units
            max_length: Maximum sequence length
        """
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.max_length = max_length
        self.model = None
        self.encoder_model = None
    
    def build_model(self, vocab_size: int):
        """
        Build improved LSTM model for similarity matching
        
        Args:
            vocab_size: Vocabulary size for embedding layer
        """
        # Input for question - improved architecture
        question_input = Input(shape=(self.max_length,), name='question_input')
        question_emb = keras.layers.Embedding(
            vocab_size, self.embedding_dim, 
            input_length=self.max_length, 
            name='question_embedding',
            mask_zero=True
        )(question_input)
        
        # Multi-layer LSTM with attention
        question_lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True), name='question_lstm1')(question_emb)
        question_dropout1 = Dropout(0.2, name='question_dropout1')(question_lstm1)
        question_lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=False), name='question_lstm2')(question_dropout1)
        question_dense1 = Dense(self.embedding_dim * 2, activation='relu', name='question_dense1')(question_lstm2)
        question_dropout2 = Dropout(0.3, name='question_dropout2')(question_dense1)
        question_dense2 = Dense(self.embedding_dim, activation='tanh', name='question_dense2')(question_dropout2)
        
        # Input for answer/chunk - same improved architecture
        answer_input = Input(shape=(self.max_length,), name='answer_input')
        answer_emb = keras.layers.Embedding(
            vocab_size, self.embedding_dim, 
            input_length=self.max_length, 
            name='answer_embedding',
            mask_zero=True
        )(answer_input)
        
        answer_lstm1 = Bidirectional(LSTM(self.lstm_units, return_sequences=True), name='answer_lstm1')(answer_emb)
        answer_dropout1 = Dropout(0.2, name='answer_dropout1')(answer_lstm1)
        answer_lstm2 = Bidirectional(LSTM(self.lstm_units, return_sequences=False), name='answer_lstm2')(answer_dropout1)
        answer_dense1 = Dense(self.embedding_dim * 2, activation='relu', name='answer_dense1')(answer_lstm2)
        answer_dropout2 = Dropout(0.3, name='answer_dropout2')(answer_dense1)
        answer_dense2 = Dense(self.embedding_dim, activation='tanh', name='answer_dense2')(answer_dropout2)
        
        # Combine using multiple methods for better similarity
        # Method 1: Concatenation
        combined = Concatenate(name='combined')([question_dense2, answer_dense2])
        # Method 2: Element-wise multiplication
        multiplied = keras.layers.Multiply(name='multiplied')([question_dense2, answer_dense2])
        # Method 3: Absolute difference
        diff = keras.layers.Lambda(lambda x: keras.backend.abs(x[0] - x[1]), name='difference')([question_dense2, answer_dense2])
        
        # Combine all methods
        all_features = Concatenate(name='all_features')([combined, multiplied, diff])
        
        # Deeper network for better learning
        combined_dense1 = Dense(self.embedding_dim * 3, activation='relu', name='combined_dense1')(all_features)
        combined_dropout1 = Dropout(0.4, name='combined_dropout1')(combined_dense1)
        combined_dense2 = Dense(self.embedding_dim * 2, activation='relu', name='combined_dense2')(combined_dropout1)
        combined_dropout2 = Dropout(0.3, name='combined_dropout2')(combined_dense2)
        combined_dense3 = Dense(self.embedding_dim, activation='relu', name='combined_dense3')(combined_dropout2)
        output = Dense(1, activation='sigmoid', name='similarity_score')(combined_dense3)
        
        # Full model for training
        self.model = Model(inputs=[question_input, answer_input], outputs=output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),  # Lower learning rate for better convergence
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Encoder model for creating embeddings (use question encoder)
        self.encoder_model = Model(inputs=question_input, outputs=question_dense2)
        
        # Also create answer encoder for better matching
        self.answer_encoder = Model(inputs=answer_input, outputs=answer_dense2)
        
        return self.model
    
    def create_training_data(self, chunks: List[Dict], embedder, num_negative: int = 5):
        """
        Create improved training data from chunks
        
        Args:
            chunks: List of chunk dictionaries
            embedder: TensorFlowEmbedder instance
            num_negative: Number of negative samples per positive sample
            
        Returns:
            Tuple of (X_question, X_answer, y)
        """
        texts = [chunk['text'] for chunk in chunks]
        sequences = embedder.texts_to_sequences(texts)
        
        X_question = []
        X_answer = []
        y = []
        
        print(f"Creating training data from {len(chunks)} chunks...")
        
        # Create positive pairs (chunk with itself) - stronger signal
        for i, seq in enumerate(sequences):
            X_question.append(seq)
            X_answer.append(seq)
            y.append(1.0)
        
        # Create positive pairs from adjacent chunks (they're likely related)
        for i in range(len(sequences) - 1):
            if chunks[i].get('source_file') == chunks[i+1].get('source_file'):
                # Adjacent chunks from same file are likely related
                X_question.append(sequences[i])
                X_answer.append(sequences[i+1])
                y.append(0.8)  # High similarity but not perfect
        
        # Create hard negative pairs (chunks from different files)
        np.random.seed(42)
        file_groups = {}
        for i, chunk in enumerate(chunks):
            file_name = chunk.get('source_file', 'unknown')
            if file_name not in file_groups:
                file_groups[file_name] = []
            file_groups[file_name].append(i)
        
        # Hard negatives: chunks from different files
        for i in range(len(sequences)):
            current_file = chunks[i].get('source_file', 'unknown')
            other_files = [f for f in file_groups.keys() if f != current_file]
            
            for _ in range(num_negative):
                if other_files:
                    # Pick from different file
                    other_file = np.random.choice(other_files)
                    j = np.random.choice(file_groups[other_file])
                else:
                    # Fallback to random
                    j = np.random.randint(0, len(sequences))
                    while j == i:
                        j = np.random.randint(0, len(sequences))
                
                X_question.append(sequences[i])
                X_answer.append(sequences[j])
                y.append(0.0)
        
        # Shuffle the data
        indices = np.arange(len(X_question))
        np.random.seed(42)
        np.random.shuffle(indices)
        
        X_question = np.array([X_question[i] for i in indices])
        X_answer = np.array([X_answer[i] for i in indices])
        y = np.array([y[i] for i in indices])
        
        print(f"Created {len(X_question)} training samples ({np.sum(y==1.0)} positive, {np.sum(y==0.0)} negative)")
        
        return X_question, X_answer, y
    
    def train(self, X_question: np.ndarray, X_answer: np.ndarray, y: np.ndarray,
              epochs: int = 10, batch_size: int = 32, validation_split: float = 0.2,
              save_path: Optional[str] = None):
        """
        Train the LSTM model
        
        Args:
            X_question: Question sequences
            X_answer: Answer sequences
            y: Labels (1 for similar, 0 for not similar)
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be built first")
        
        # Split data
        X_q_train, X_q_val, X_a_train, X_a_val, y_train, y_val = train_test_split(
            X_question, X_answer, y, test_size=validation_split, random_state=42
        )
        
        # Callbacks
        callbacks = []
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            callbacks.append(ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss'))
        callbacks.append(EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True))
        
        # Print model summary
        print("\nModel Architecture:")
        self.model.summary()
        
        print(f"\nTraining on {len(X_q_train)} samples, validating on {len(X_q_val)} samples")
        print(f"Positive samples: {np.sum(y_train == 1.0)} train, {np.sum(y_val == 1.0)} val")
        print(f"Negative samples: {np.sum(y_train == 0.0)} train, {np.sum(y_val == 0.0)} val")
        
        # Train with class weights to handle imbalance
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
        
        # Train
        history = self.model.fit(
            [X_q_train, X_a_train], y_train,
            validation_data=([X_q_val, X_a_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Print final metrics
        print("\nTraining completed!")
        print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return history
    
    def encode_question(self, question_sequence: np.ndarray) -> np.ndarray:
        """
        Encode a question using the LSTM encoder
        
        Args:
            question_sequence: Tokenized question sequence
            
        Returns:
            Encoded question embedding
        """
        if self.encoder_model is None:
            raise ValueError("Encoder model must be built first")
        
        if len(question_sequence.shape) == 1:
            question_sequence = question_sequence.reshape(1, -1)
        
        return self.encoder_model.predict(question_sequence, verbose=0)
    
    def compute_similarity(self, question_emb: np.ndarray, answer_emb: np.ndarray) -> float:
        """
        Compute similarity between question and answer embeddings
        
        Args:
            question_emb: Question embedding
            answer_emb: Answer embedding
            
        Returns:
            Similarity score (0-1)
        """
        # Use cosine similarity
        dot_product = np.dot(question_emb, answer_emb)
        norm_q = np.linalg.norm(question_emb)
        norm_a = np.linalg.norm(answer_emb)
        
        if norm_q == 0 or norm_a == 0:
            return 0.0
        
        similarity = dot_product / (norm_q * norm_a)
        # Normalize to 0-1 range
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    def save(self, directory: str):
        """Save the model"""
        os.makedirs(directory, exist_ok=True)
        
        if self.model:
            model_path = os.path.join(directory, 'lstm_model.h5')
            self.model.save(model_path)
        
        if self.encoder_model:
            encoder_path = os.path.join(directory, 'lstm_encoder.h5')
            self.encoder_model.save(encoder_path)
        
        config = {
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'max_length': self.max_length
        }
        config_path = os.path.join(directory, 'lstm_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(config, f)
        
        print(f"Saved LSTM model to {directory}")
    
    def load(self, directory: str, vocab_size: int):
        """Load the model"""
        config_path = os.path.join(directory, 'lstm_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            self.embedding_dim = config['embedding_dim']
            self.lstm_units = config['lstm_units']
            self.max_length = config['max_length']
        
        model_path = os.path.join(directory, 'lstm_model.h5')
        encoder_path = os.path.join(directory, 'lstm_encoder.h5')
        
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"Loaded LSTM model from {directory}")
        
        if os.path.exists(encoder_path):
            self.encoder_model = keras.models.load_model(encoder_path)
        elif self.model:
            # Rebuild encoder from full model
            try:
                self.encoder_model = Model(
                    inputs=self.model.get_layer('question_input').input,
                    outputs=self.model.get_layer('question_dropout').output
                )
            except Exception as e:
                print(f"Warning: Could not rebuild encoder from model: {e}")
                # Fallback: rebuild encoder
                if vocab_size > 0:
                    self.build_model(vocab_size)

