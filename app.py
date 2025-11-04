import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os
import pickle

st.set_page_config(page_title="MLP Text Generator", page_icon="📝", layout="wide")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model definition with configurable activation
class MLPTextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=1024, window_size=3, activation='relu'):
        super(MLPTextGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.input_dim = embedding_dim * window_size
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:  # gelu
            self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        embedded = self.embedding(x)
        embedded = embedded.view(batch_size, -1)
        h1 = self.dropout1(self.activation(self.fc1(embedded)))
        output = self.fc2(h1)
        return output

# Define model configurations (Dataset 2)
MODEL_CONFIGS = {
    'mlp_text_model_dataset1.pth': {
        'embedding_dim': 32, 'window_size': 3, 'activation': 'relu', 'name': 'Base Model'
    },
    'mlp_text_model_dataset1var1.pth': {
        'embedding_dim': 32, 'window_size': 3, 'activation': 'tanh', 'name': 'Variant 1'
    },
    'mlp_text_model_dataset1var2.pth': {
        'embedding_dim': 64, 'window_size': 3, 'activation': 'tanh', 'name': 'Variant 2'
    },
    'mlp_text_model_dataset1var3.pth': {
        'embedding_dim': 64, 'window_size': 5, 'activation': 'relu', 'name': 'Variant 3'
    },
    'mlp_text_model_dataset1var4.pth': {
        'embedding_dim': 64, 'window_size': 5, 'activation': 'tanh', 'name': 'Variant 4'
    },
    'mlp_text_model_dataset1var5.pth': {
        'embedding_dim': 32, 'window_size': 5, 'activation': 'relu', 'name': 'Variant 5'
    }
}

def get_model_path(embedding_dim, window_size, activation):
    """Get model path based on configuration"""
    for model_path, config in MODEL_CONFIGS.items():
        if (config['embedding_dim'] == embedding_dim and 
            config['window_size'] == window_size and 
            config['activation'] == activation):
            return model_path
    return None

def get_available_configs():
    """Get list of available model configurations"""
    available = []
    for model_path, config in MODEL_CONFIGS.items():
        if os.path.exists(os.path.join(SCRIPT_DIR, model_path)):
            available.append(config)
    return available

def _extract_vocab_from_container(obj):
    """
    Support:
    - word_to_idx/idx_to_word, stoi/itos, token_to_idx/idx_to_token
    - char_to_idx/idx_to_char
    - vocabulary (dict or list) as found in your pickle
    """
    w2i = (obj.get('word_to_idx') or obj.get('stoi') or
           obj.get('token_to_idx') or obj.get('char_to_idx'))
    i2w = (obj.get('idx_to_word') or obj.get('itos') or
           obj.get('idx_to_token') or obj.get('idx_to_char'))

    # Handle 'vocabulary' key from your file
    if not w2i and not i2w and 'vocabulary' in obj:
        vocab = obj['vocabulary']
        if isinstance(vocab, dict):
            # Detect mapping direction by key/value types
            if all(isinstance(k, str) and isinstance(v, int) for k, v in vocab.items()):
                w2i = vocab
                i2w = {v: k for k, v in vocab.items()}
            elif all(isinstance(k, int) and isinstance(v, str) for k, v in vocab.items()):
                i2w = vocab
                w2i = {v: k for k, v in vocab.items()}
        elif isinstance(vocab, list):
            i2w = {i: tok for i, tok in enumerate(vocab)}
            w2i = {tok: i for i, tok in enumerate(vocab)}

    # If i2w is a list, convert to dict
    if isinstance(i2w, list):
        i2w = {i: tok for i, tok in enumerate(i2w)}

    # If only one side exists, derive the other
    if w2i and not i2w:
        i2w = {int(v): str(k) for k, v in w2i.items()}
    if i2w and not w2i:
        w2i = {str(v): int(k) for k, v in i2w.items()}

    # Normalize
    if isinstance(w2i, dict):
        w2i = {str(k): int(v) for k, v in w2i.items()}
    if isinstance(i2w, dict):
        i2w = {int(k): str(v) for k, v in i2w.items()}

    return w2i or {}, i2w or {}

@st.cache_resource
def load_base_vocabulary():
    """Load vocabulary from base model or sidecar pickle - all models share this vocabulary"""
    # 1) Try from base model checkpoint
    base_model_path = os.path.join(SCRIPT_DIR, 'mlp_text_model_dataset1.pth')
    if os.path.exists(base_model_path):
        try:
            ckpt = torch.load(base_model_path, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict):
                w2i, i2w = _extract_vocab_from_container(ckpt)
                if w2i and i2w:
                    return w2i, i2w
        except Exception:
            pass

    # 2) Try sidecar vocab pickles (supports your 'vocabulary' schema)
    candidates = [
        os.path.join(SCRIPT_DIR, 'vocab_data_dataset1.pkl'),
        os.path.join(os.path.expanduser('~/Downloads'), 'vocab_data_dataset1.pkl'),
    ]
    for p in candidates:
        if not os.path.exists(p):
            continue
        try:
            with open(p, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, dict):
                w2i, i2w = _extract_vocab_from_container(obj)
            elif isinstance(obj, list):
                i2w = {i: tok for i, tok in enumerate(obj)}
                w2i = {tok: i for i, tok in enumerate(obj)}
            else:
                w2i, i2w = {}, {}
            if w2i and i2w:
                return w2i, i2w
        except Exception:
            continue

    return None, None

@st.cache_resource
def load_model(model_path, shared_word_to_idx, shared_idx_to_word):
    full_path = os.path.join(SCRIPT_DIR, model_path)
    if not os.path.exists(full_path):
        return None
    
    checkpoint = torch.load(full_path, map_location='cpu', weights_only=False)

    # Extract model state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model_state = checkpoint['state_dict']
    else:
        model_state = checkpoint

    # Get configuration from MODEL_CONFIGS or checkpoint
    config = MODEL_CONFIGS.get(model_path, {})
    
    # Try to get parameters from checkpoint first, then fall back to config
    vocab_size = len(shared_word_to_idx) if shared_word_to_idx else (checkpoint.get('vocab_size', None) if isinstance(checkpoint, dict) else None)
    if vocab_size is None and isinstance(model_state, dict) and 'embedding.weight' in model_state:
        vocab_size = model_state['embedding.weight'].shape[0]
    
    embedding_dim = (checkpoint.get('embedding_dim', config.get('embedding_dim', 32)) if isinstance(checkpoint, dict) else config.get('embedding_dim', 32))
    if isinstance(model_state, dict) and 'embedding.weight' in model_state:
        embedding_dim = model_state['embedding.weight'].shape[1]
    
    hidden_dim = (checkpoint.get('hidden_dim', 1024) if isinstance(checkpoint, dict) else 1024)
    window_size = (checkpoint.get('window_size', config.get('window_size', 3)) if isinstance(checkpoint, dict) else config.get('window_size', 3))
    activation = (checkpoint.get('activation', config.get('activation', 'relu')) if isinstance(checkpoint, dict) else config.get('activation', 'relu'))
    
    # Use shared vocabulary
    word_to_idx = shared_word_to_idx
    idx_to_word = shared_idx_to_word
    
    # Create and load model
    model = MLPTextGenerator(vocab_size, embedding_dim, hidden_dim, window_size, activation)
    model.load_state_dict(model_state)
    model.eval()

    return model, word_to_idx, idx_to_word, window_size, vocab_size, embedding_dim, hidden_dim, activation

def handle_unknown_words(text, word_to_idx, strategy='skip'):
    words = text.lower().split()
    oov_words = []

    if strategy == 'skip':
        known = [w for w in words if w in word_to_idx]
        oov_words = [w for w in words if w not in word_to_idx]
        return known, oov_words

def generate_text(model, seed_text, word_to_idx, idx_to_word, window_size,
                  num_words=20, temperature=1.0, seed=42, unknown_strategy='skip'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()

    processed_words, oov_words = handle_unknown_words(seed_text, word_to_idx, unknown_strategy)

    if len(processed_words) < window_size:
        oov_msg = f" (Out-of-vocabulary words: {', '.join(oov_words)})" if oov_words else ""
        return None, f"Need at least {window_size} known words. Got {len(processed_words)}: {' '.join(processed_words)}{oov_msg}"

    generated = processed_words.copy()

    with torch.no_grad():
        for _ in range(num_words):
            context = generated[-window_size:]

            try:
                context_indices = [word_to_idx[word] for word in context]
            except KeyError:
                break

            input_tensor = torch.LongTensor([context_indices])
            output = model(input_tensor)
            output = output / max(temperature, 1e-8)
            probabilities = torch.softmax(output, dim=1)
            predicted_idx = torch.multinomial(probabilities, 1).item()
            predicted_word = idx_to_word.get(predicted_idx, '<unk>')

            generated.append(predicted_word)

    return ' '.join(generated), oov_words

# Initialize session state for seed text
if 'seed_text' not in st.session_state:
    st.session_state.seed_text = "sherlock holmes was a detective"

# Main app
st.title("📝 MLP Text Generator")
st.markdown("Generate text using trained MLP models with different configurations.")

# Load shared vocabulary from base model (Dataset 2) or sidecar vocab
shared_word_to_idx, shared_idx_to_word = load_base_vocabulary()

if shared_word_to_idx is None or shared_idx_to_word is None or len(shared_word_to_idx) == 0:
    st.error("❌ Could not load vocabulary from base model or vocab_data_dataset2*.pkl.")
    st.stop()

st.sidebar.success(f"✅ Loaded vocabulary: {len(shared_word_to_idx):,} words")

# Check available models
available_configs = get_available_configs()
if not available_configs:
    st.error("❌ No model files found in current directory.")
    st.stop()

# Sidebar - Model Configuration Selection
st.sidebar.header("🎛️ Model Configuration")

available_embedding_dims = sorted(list(set(c['embedding_dim'] for c in available_configs)))
available_window_sizes = sorted(list(set(c['window_size'] for c in available_configs)))
available_activations = sorted(list(set(c['activation'] for c in available_configs)))

selected_embedding_dim = st.sidebar.selectbox(
    "Embedding Dimension",
    available_embedding_dims,
    help="Size of word embeddings"
)

selected_window_size = st.sidebar.selectbox(
    "Context Length (Window Size)",
    available_window_sizes,
    help="Number of previous words to consider"
)

selected_activation = st.sidebar.selectbox(
    "Activation Function",
    available_activations,
    help="Non-linear activation in hidden layer"
)

# Get corresponding model path
model_path = get_model_path(selected_embedding_dim, selected_window_size, selected_activation)
full_selected_path = os.path.join(SCRIPT_DIR, model_path) if model_path else None

if model_path is None or not os.path.exists(full_selected_path):
    st.sidebar.error("❌ No model found with this configuration!")
    st.error(f"❌ Model with Embedding={selected_embedding_dim}, Window={selected_window_size}, Activation={selected_activation} not found.")
    st.stop()

# Load selected model with shared vocabulary
model_result = load_model(model_path, shared_word_to_idx, shared_idx_to_word)
if model_result is None:
    st.error(f"❌ Failed to load model: {model_path}")
    st.stop()

model, word_to_idx, idx_to_word, window_size, vocab_size, embedding_dim, hidden_dim, activation = model_result
st.sidebar.success(f"✅ Loaded: {MODEL_CONFIGS[model_path]['name']}")

# Sidebar - Model Info
st.sidebar.markdown("---")
st.sidebar.header("📊 Model Information")
with st.sidebar.expander("Architecture Details", expanded=True):
    st.write(f"**Model File:** `{model_path}`")
    st.write(f"**Vocabulary Size:** {vocab_size:,}")
    st.write(f"**Embedding Dimension:** {embedding_dim}")
    st.write(f"**Hidden Dimension:** {hidden_dim}")
    st.write(f"**Context Length:** {window_size} words")
    st.write(f"**Activation Function:** {activation.upper()}")

st.sidebar.markdown("---")

# Sidebar - Generation Parameters
st.sidebar.header("⚙️ Generation Parameters")
temperature = st.sidebar.slider("🌡️ Temperature", 0.1, 2.0, 1.0, 0.1)
num_words = st.sidebar.slider("📏 Words to Generate", 5, 100, 20, 5)
random_seed = st.sidebar.number_input("🎲 Random Seed", 0, 10000, 42)
unknown_strategy = st.sidebar.selectbox("❓ Handle Unknown Words", ["skip"])

# Show available models
st.sidebar.markdown("---")
st.sidebar.header("📋 Available Models")
with st.sidebar.expander("View All Configurations", expanded=False):
    for path, config in MODEL_CONFIGS.items():
        if os.path.exists(os.path.join(SCRIPT_DIR, path)):
            st.write(f"**{config['name']}**")
            st.write(f"  - Embedding: {config['embedding_dim']}")
            st.write(f"  - Window: {config['window_size']}")
            st.write(f"  - Activation: {config['activation']}")
            st.write("---")

# Main area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 Input Text")
    seed_text = st.text_area(
        f"Enter seed text (minimum {window_size} words):",
        value=st.session_state.seed_text,
        height=100,
        placeholder="Type your seed text here...",
        key="text_input"
    )

with col2:
    st.subheader("💡 Tips")
    st.markdown(f"""
    **Context:** {window_size} words

    **Temperature:**
    - 0.5: Safe & predictable
    - 1.0: Balanced
    - 1.5: Creative & diverse

    **Unknown words:**
    - skip: Remove them
    """)

generate_button = st.button("🚀 Generate Text", type="primary", use_container_width=True)

if generate_button:
    if not seed_text.strip():
        st.warning("⚠️ Please enter some seed text.")
    else:
        with st.spinner("🔄 Generating text..."):
            generated, oov = generate_text(
                model, seed_text, word_to_idx, idx_to_word, window_size,
                num_words, temperature, random_seed, unknown_strategy
            )

        if generated is None:
            st.error(f"❌ {oov}")
        else:
            st.markdown("---")
            st.subheader("✨ Generated Text")
            st.markdown(f"""
            <div style="background-color: #1e1e1e; padding: 1.5rem; border-radius: 0.5rem;
                        border-left: 5px solid #00d4ff; font-size: 1.1rem; line-height: 1.6;
                        color: #ffffff; font-family: monospace;">
                {generated}
            </div>
            """, unsafe_allow_html=True)

            if oov:
                st.warning(f"⚠️ **Out-of-vocabulary words:** {', '.join(set(oov))}")

            st.markdown("---")
            with st.expander("📊 Text Statistics"):
                words_gen = generated.split()
                unique_words = len(set(words_gen))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Words", len(words_gen))
                c2.metric("Unique Words", unique_words)
                c3.metric("Diversity", f"{unique_words/len(words_gen):.1%}")
                c4.metric("Avg Word Length", f"{np.mean([len(w) for w in words_gen]):.1f}")

# Examples section (kept same as dataset1 for parity)
st.markdown("---")
st.subheader("💡 Try These Examples")
examples = [
    "sherlock holmes was a detective",
    "it was a dark and stormy night",
    "the man said to watson that he",
    "watson looked at the door and saw",
    "i have never seen such a strange"
]
cols = st.columns(5)
for idx, (col, example) in enumerate(zip(cols, examples)):
    with col:
        if st.button(f"Example {idx+1}", key=f"ex{idx}", use_container_width=True):
            st.session_state.seed_text = example
            st.rerun()

st.caption(f"💡 Click an example button to use it as seed text (Current window size: {window_size} words)")