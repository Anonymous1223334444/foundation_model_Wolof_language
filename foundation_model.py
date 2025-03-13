import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Wav2Vec2Processor, 
    Wav2Vec2ForCTC,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
import pandas as pd
import numpy as np
import librosa
import logging
import wandb
from datasets import Dataset as HFDataset
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration class
class WolofModelConfig:
    def __init__(self):
        # General settings
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = "wolof_foundation_model"
        self.use_wandb = True
        
        # Text model settings
        self.text_data_path = "data/text/"
        self.text_model_type = "bert"  # Options: "bert", "gpt", "roberta", etc.
        self.text_model_name = "bert-base-multilingual-cased"  # Starting checkpoint
        self.tokenizer_max_length = 128
        self.mlm_probability = 0.15
        
        # Speech model settings
        self.audio_data_path = "data/audio/"
        self.audio_transcripts_path = "data/transcripts/"
        self.speech_model_type = "whisper"  # Options: "wav2vec2", "whisper"
        self.speech_model_name = "openai/whisper-small"  # Starting checkpoint
        self.sample_rate = 16000
        self.speech_max_length = 30  # seconds
        
        # Training settings
        self.text_batch_size = 32
        self.speech_batch_size = 16
        self.text_epochs = 10
        self.speech_epochs = 20
        self.text_learning_rate = 5e-5
        self.speech_learning_rate = 3e-5
        self.warmup_steps = 500
        self.eval_steps = 500
        self.save_steps = 1000

# Data preparation
class WolofTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0]
        }

class WolofAudioDataset(Dataset):
    def __init__(self, audio_paths, transcripts, processor, sample_rate, max_length):
        self.audio_paths = audio_paths
        self.transcripts = transcripts
        self.processor = processor
        self.sample_rate = sample_rate
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcript = self.transcripts[idx]
        
        # Load and preprocess audio
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        # Trim or pad audio to max_length
        max_samples = self.sample_rate * self.max_length
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        else:
            audio = np.pad(audio, (0, max_samples - len(audio)))
        
        # Process for the specific model
        if isinstance(self.processor, Wav2Vec2Processor):
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            input_values = inputs.input_values[0]
            
            with self.processor.as_target_processor():
                labels = self.processor(transcript).input_ids
                
            return {
                "input_values": input_values,
                "labels": torch.tensor(labels)
            }
        
        elif isinstance(self.processor, WhisperProcessor):
            inputs = self.processor(
                audio, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            )
            input_features = inputs.input_features[0]
            
            # Process transcript
            encoded_transcript = self.processor.tokenizer(transcript, return_tensors="pt")
            labels = encoded_transcript.input_ids[0]
            
            return {
                "input_features": input_features,
                "labels": labels
            }

# Data loading functions
def load_text_data(config):
    """Load and prepare text data for training"""
    logger.info("Loading text data...")
    
    all_texts = []
    text_files = []
    
    # Find all text files
    for root, _, files in os.walk(config.text_data_path):
        for file in files:
            if file.endswith('.txt'):
                text_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(text_files)} text files")
    
    # Read all text files
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Split text into paragraphs or sentences
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            all_texts.extend(paragraphs)
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
    
    logger.info(f"Loaded {len(all_texts)} text segments")
    
    # Split into train and validation
    train_texts, val_texts = train_test_split(
        all_texts, 
        test_size=0.1, 
        random_state=config.seed
    )
    
    return train_texts, val_texts

def load_audio_data(config):
    """Load and prepare audio data for training"""
    logger.info("Loading audio data...")
    
    audio_paths = []
    transcripts = []
    
    # Check if we have a metadata file with transcript info
    metadata_path = os.path.join(config.audio_transcripts_path, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            
        for item in metadata:
            audio_path = os.path.join(config.audio_data_path, item['audio_file'])
            if os.path.exists(audio_path):
                audio_paths.append(audio_path)
                transcripts.append(item['transcript'])
    else:
        # Try to match audio files with transcript files
        for root, _, files in os.walk(config.audio_data_path):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    
                    # Look for matching transcript file
                    transcript_path = os.path.join(
                        config.audio_transcripts_path, 
                        f"{base_name}.txt"
                    )
                    
                    if os.path.exists(transcript_path):
                        with open(transcript_path, 'r', encoding='utf-8') as f:
                            transcript = f.read().strip()
                        
                        audio_paths.append(audio_path)
                        transcripts.append(transcript)
    
    logger.info(f"Loaded {len(audio_paths)} audio files with transcripts")
    
    # Split into train and validation
    train_audio_paths, val_audio_paths, train_transcripts, val_transcripts = train_test_split(
        audio_paths, 
        transcripts,
        test_size=0.1, 
        random_state=config.seed
    )
    
    return train_audio_paths, val_audio_paths, train_transcripts, val_transcripts

# Model training functions
def train_text_model(config):
    """Train the text language model"""
    logger.info(f"Training text model using {config.text_model_type} architecture")
    
    # Load data
    train_texts, val_texts = load_text_data(config)
    
    # Initialize tokenizer
    if os.path.exists(os.path.join(config.output_dir, "tokenizer")):
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(config.output_dir, "tokenizer"))
        logger.info("Loaded saved tokenizer")
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)
        
        # Add Wolof-specific tokens if needed
        # tokenizer.add_tokens(["example_wolof_token"])
        
        # Save tokenizer
        os.makedirs(os.path.join(config.output_dir, "tokenizer"), exist_ok=True)
        tokenizer.save_pretrained(os.path.join(config.output_dir, "tokenizer"))
        logger.info("Initialized and saved tokenizer")
    
    # Prepare datasets
    train_dataset = HFDataset.from_dict({"text": train_texts})
    val_dataset = HFDataset.from_dict({"text": val_texts})
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.tokenizer_max_length
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Initialize model
    if config.text_model_type.lower() == "bert":
        model_class = AutoModelForMaskedLM
    elif config.text_model_type.lower() in ["gpt", "gpt2"]:
        model_class = AutoModelForCausalLM
    else:
        model_class = AutoModelForMaskedLM  # Default to MLM
    
    if os.path.exists(os.path.join(config.output_dir, "text_model")):
        model = model_class.from_pretrained(os.path.join(config.output_dir, "text_model"))
        logger.info("Loaded saved text model")
    else:
        model = model_class.from_pretrained(config.text_model_name)
        
        # Resize embedding layer if we added tokens
        # model.resize_token_embeddings(len(tokenizer))
        
        logger.info("Initialized text model")
    
    # Prepare data collator
    if config.text_model_type.lower() == "bert":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config.mlm_probability
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(project="wolof-foundation-model", name="text-model-training")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "text_model_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=config.text_epochs,
        per_device_train_batch_size=config.text_batch_size,
        per_device_eval_batch_size=config.text_batch_size,
        learning_rate=config.text_learning_rate,
        weight_decay=0.01,
        warmup_steps=config.warmup_steps,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train model
    logger.info("Starting text model training")
    trainer.train()
    
    # Save final model
    model.save_pretrained(os.path.join(config.output_dir, "text_model"))
    logger.info("Text model training complete")
    
    if config.use_wandb:
        wandb.finish()
    
    return model, tokenizer

def train_speech_model(config):
    """Train the speech recognition model"""
    logger.info(f"Training speech model using {config.speech_model_type} architecture")
    
    # Load data
    train_audio_paths, val_audio_paths, train_transcripts, val_transcripts = load_audio_data(config)
    
    # Initialize processor and model
    if config.speech_model_type.lower() == "wav2vec2":
        processor_class = Wav2Vec2Processor
        model_class = Wav2Vec2ForCTC
    elif config.speech_model_type.lower() == "whisper":
        processor_class = WhisperProcessor
        model_class = WhisperForConditionalGeneration
    else:
        raise ValueError(f"Unsupported speech model type: {config.speech_model_type}")
    
    if os.path.exists(os.path.join(config.output_dir, "speech_processor")):
        processor = processor_class.from_pretrained(os.path.join(config.output_dir, "speech_processor"))
        logger.info("Loaded saved speech processor")
    else:
        processor = processor_class.from_pretrained(config.speech_model_name)
        
        # Save processor
        os.makedirs(os.path.join(config.output_dir, "speech_processor"), exist_ok=True)
        processor.save_pretrained(os.path.join(config.output_dir, "speech_processor"))
        logger.info("Initialized and saved speech processor")
    
    if os.path.exists(os.path.join(config.output_dir, "speech_model")):
        model = model_class.from_pretrained(os.path.join(config.output_dir, "speech_model"))
        logger.info("Loaded saved speech model")
    else:
        model = model_class.from_pret
    if os.path.exists(os.path.join(config.output_dir, "speech_model")):
        model = model_class.from_pretrained(os.path.join(config.output_dir, "speech_model"))
        logger.info("Loaded saved speech model")
    else:
        model = model_class.from_pretrained(config.speech_model_name)
        logger.info("Initialized speech model")
    
    # Create datasets
    train_dataset = WolofAudioDataset(
        train_audio_paths, 
        train_transcripts, 
        processor, 
        config.sample_rate, 
        config.speech_max_length
    )
    
    val_dataset = WolofAudioDataset(
        val_audio_paths, 
        val_transcripts, 
        processor, 
        config.sample_rate, 
        config.speech_max_length
    )
    
    # Initialize wandb if enabled
    if config.use_wandb:
        wandb.init(project="wolof-foundation-model", name="speech-model-training")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "speech_model_checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=config.speech_epochs,
        per_device_train_batch_size=config.speech_batch_size,
        per_device_eval_batch_size=config.speech_batch_size,
        learning_rate=config.speech_learning_rate,
        weight_decay=0.01,
        warmup_steps=config.warmup_steps,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        report_to="wandb" if config.use_wandb else None,
        fp16=True,  # Use mixed precision training
    )
    
    # Create a data collator for batching
    def data_collator(batch):
        if config.speech_model_type.lower() == "wav2vec2":
            input_values = [item["input_values"] for item in batch]
            labels = [item["labels"] for item in batch]
            
            # Pad input values
            input_values = torch.nn.utils.rnn.pad_sequence(
                input_values, 
                batch_first=True
            )
            
            # Pad labels
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, 
                batch_first=True, 
                padding_value=-100
            )
            
            return {
                "input_values": input_values,
                "labels": labels
            }
        
        elif config.speech_model_type.lower() == "whisper":
            input_features = [item["input_features"] for item in batch]
            labels = [item["labels"] for item in batch]
            
            # Pad input features
            input_features = torch.nn.utils.rnn.pad_sequence(
                input_features, 
                batch_first=True
            )
            
            # Pad labels
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, 
                batch_first=True, 
                padding_value=-100
            )
            
            return {
                "input_features": input_features,
                "labels": labels,
                "return_loss": True
            }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # Train model
    logger.info("Starting speech model training")
    trainer.train()
    
    # Save final model
    model.save_pretrained(os.path.join(config.output_dir, "speech_model"))
    logger.info("Speech model training complete")
    
    if config.use_wandb:
        wandb.finish()
    
    return model, processor

def fine_tune_multilingual_model(config):
    """Fine-tune a pretrained multilingual model on Wolof data"""
    logger.info(f"Fine-tuning multilingual model {config.text_model_name} on Wolof data")
    
    # Load and train text model
    text_model, tokenizer = train_text_model(config)
    
    return text_model, tokenizer

def train_from_scratch(config):
    """Train a Wolof language model from scratch"""
    logger.info("Training Wolof language model from scratch")
    
    # For training from scratch, we would need to define a custom model architecture
    # and train it on the Wolof data. This is a more advanced approach.
    
    # Load data
    train_texts, val_texts = load_text_data(config)
    
    # Initialize a new tokenizer with Wolof vocabulary
    from tokenizers import ByteLevelBPETokenizer
    
    # First, create a temporary file with all texts
    with open('temp_wolof_corpus.txt', 'w', encoding='utf-8') as f:
        for text in train_texts:
            f.write(text + '\n')
    
    # Train a new tokenizer
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=['temp_wolof_corpus.txt'],
        vocab_size=52000,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    
    # Save the tokenizer
    os.makedirs(os.path.join(config.output_dir, "tokenizer_from_scratch"), exist_ok=True)
    tokenizer.save_model(os.path.join(config.output_dir, "tokenizer_from_scratch"))
    
    # Convert to HuggingFace tokenizer
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(config.output_dir, "tokenizer_from_scratch", "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    
    # Define model architecture
    # This is a simplified example; in practice, you would define a more sophisticated architecture
    class WolofTransformerModel(nn.Module):
        def __init__(self, vocab_size):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 512)
            self.transformer = nn.Transformer(
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                num_decoder_layers=6,
                dim_feedforward=2048
            )
            self.fc = nn.Linear(512, vocab_size)
            
        def forward(self, x):
            embedded = self.embedding(x)
            output = self.transformer(embedded, embedded)
            return self.fc(output)
    
    # Initialize model
    model = WolofTransformerModel(len(hf_tokenizer))
    
    # Train model (simplified)
    # In practice, you would use a more sophisticated training loop
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Model training from scratch not fully implemented")
    logger.info("Consider using the fine-tuning approach instead")
    
    return model, hf_tokenizer

def create_wolof_foundation_model(config):
    """Main function to create the Wolof foundation model"""
    logger.info("Starting Wolof foundation model creation")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Fine-tune a multilingual model for text
    text_model, tokenizer = fine_tune_multilingual_model(config)
    
    # Train speech model
    speech_model, processor = train_speech_model(config)
    
    logger.info("Wolof foundation model training complete")
    
    return {
        "text_model": text_model,
        "tokenizer": tokenizer,
        "speech_model": speech_model,
        "processor": processor
    }

# Evaluation functions
def evaluate_text_model(model, tokenizer, config):
    """Evaluate the text model performance"""
    logger.info("Evaluating text model performance")
    
    # Load test data
    _, test_texts = load_text_data(config)
    test_texts = test_texts[:100]  # Take a subset for evaluation
    
    # Prepare evaluation dataset
    test_dataset = HFDataset.from_dict({"text": test_texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.tokenizer_max_length
        )
    
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Initialize data collator
    if config.text_model_type.lower() == "bert":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=config.mlm_probability
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    
    # Setup evaluator
    eval_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "eval_results"),
        per_device_eval_batch_size=config.text_batch_size,
        remove_unused_columns=False,
    )
    
    evaluator = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )
    
    # Evaluate
    eval_results = evaluator.evaluate()
    
    logger.info(f"Text model evaluation results: {eval_results}")
    
    return eval_results

def evaluate_speech_model(model, processor, config):
    """Evaluate the speech model performance"""
    logger.info("Evaluating speech model performance")
    
    # Load test data
    _, test_audio_paths, _, test_transcripts = load_audio_data(config)
    test_audio_paths = test_audio_paths[:20]  # Take a subset for evaluation
    test_transcripts = test_transcripts[:20]
    
    # Create test dataset
    test_dataset = WolofAudioDataset(
        test_audio_paths, 
        test_transcripts, 
        processor, 
        config.sample_rate, 
        config.speech_max_length
    )
    
    # Define data collator
    def data_collator(batch):
        if config.speech_model_type.lower() == "wav2vec2":
            input_values = [item["input_values"] for item in batch]
            labels = [item["labels"] for item in batch]
            
            input_values = torch.nn.utils.rnn.pad_sequence(
                input_values, 
                batch_first=True
            )
            
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, 
                batch_first=True, 
                padding_value=-100
            )
            
            return {
                "input_values": input_values,
                "labels": labels
            }
        
        elif config.speech_model_type.lower() == "whisper":
            input_features = [item["input_features"] for item in batch]
            labels = [item["labels"] for item in batch]
            
            input_features = torch.nn.utils.rnn.pad_sequence(
                input_features, 
                batch_first=True
            )
            
            labels = torch.nn.utils.rnn.pad_sequence(
                labels, 
                batch_first=True, 
                padding_value=-100
            )
            
            return {
                "input_features": input_features,
                "labels": labels,
                "return_loss": True
            }
    
    # Setup evaluator
    eval_args = TrainingArguments(
        output_dir=os.path.join(config.output_dir, "speech_eval_results"),
        per_device_eval_batch_size=config.speech_batch_size,
        remove_unused_columns=False,
    )
    
    evaluator = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        eval_dataset=test_dataset,
    )
    
    # Evaluate
    eval_results = evaluator.evaluate()
    
    logger.info(f"Speech model evaluation results: {eval_results}")
    
    return eval_results

# Main function to run the pipeline
def main():
    """Main function to run the Wolof foundation model training pipeline"""
    # Initialize configuration
    config = WolofModelConfig()
    
    # Log system info
    logger.info(f"Using device: {config.device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create the foundation model
    models = create_wolof_foundation_model(config)
    
    # Evaluate models
    text_eval_results = evaluate_text_model(
        models["text_model"], 
        models["tokenizer"], 
        config
    )
    
    speech_eval_results = evaluate_speech_model(
        models["speech_model"], 
        models["processor"], 
        config
    )
    
    # Save evaluation results
    results = {
        "text_model_results": text_eval_results,
        "speech_model_results": speech_eval_results
    }
    
    with open(os.path.join(config.output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info("Wolof foundation model training pipeline completed successfully")

if __name__ == "__main__":
    main()