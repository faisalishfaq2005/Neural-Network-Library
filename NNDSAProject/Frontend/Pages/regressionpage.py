import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from DataStructures.linklist import LinkList
from DataStructures.queueADT import Queue
from Layers.ActivationLayer import ActivationLayer
from Layers.DenseLayer import DenseLayer
from Layers.InputLayer import InputLayer
from Layers.ActivationFunctions import sigmoid,sigmoid_derivative,relu,relu_derivative,leaky_relu,leaky_relu_derivative
from Supporting_Functions.normalization import normalize,denormalize
from Layers.lossFunctions import binary_cross_entropy,mse_loss
from Supporting_Functions.batching import batch_split1,Batch
import plotly.graph_objects as go


# CSS for frontend styling
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
    }
    .btn-primary {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 18px;
        cursor: pointer;
        transition: transform 0.3s;
    }
    .btn-primary:hover {
        transform: scale(1.1);
        background-color: #45a049;
    }
    .animate-layer {
        transition: transform 0.3s, background-color 0.3s;
        margin: 10px;
        padding: 15px;
        border-radius: 10px;
        border: 2px dashed #4CAF50;
        background-color: rgba(76, 175, 80, 0.1);
        font-size: 18px;
        cursor: pointer;
    }
    .animate-layer:hover {
        transform: scale(1.1);
        background-color: rgba(76, 175, 80, 0.3);
    }
    .loss-box {
        background-color: rgba(255, 193, 7, 0.1);
        padding: 15px;
        border: 2px dashed #FFC107;
        margin: 10px;
        text-align: center;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title and Header
st.title("🧠 Neural Network Regression Trainer")
st.markdown("### Build, Train, and Evaluate a Neural Network with Intuitive Controls")

data=None
input_features=None
output_features=None
input_features_normalized=None
output_features_normalized=None
nn=LinkList()
nn_trained=LinkList()


# Step 1: Upload the dataset
st.markdown("#### Step 1: Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    input_features=data.iloc[:,:-1].values
    output_features=data.iloc[:,-1].values.reshape(-1,1)
    if input_features is not None:
        print(input_features.shape)



    st.write("### Uploaded Data", data.head())
else:
    st.info("Please upload a CSV file to proceed.")
    st.stop()

# Step 2: Normalization
st.markdown("#### Step 2: Data Normalization")
normalize_data = st.checkbox("Normalize Data?", value=True)
if normalize_data and input_features is not None and input_features.size > 0 and output_features is not None and output_features.size > 0:

    input_features_normalized, input_min, input_max = normalize(input_features)
    
    output_features_normalized, output_min, output_max = normalize(output_features)

    st.success("Data normalized successfully!")
    st.write("Normalized Data Preview:")
else:
    st.warning("Data will not be normalized.")

print(input_features_normalized)
print(output_features_normalized)

# Step 3: Define Neural Network Layers


if "nn_structure" not in st.session_state:
    st.session_state.nn_structure = []
if "nn_linklist" not in st.session_state:
    st.session_state.nn_linklist=[]
if "trained_model_layers" not in st.session_state:
    st.session_state.trained_model_layers=[]
if "prediction" not in st.session_state:
    st.session_state.prediction=[]


st.title("Neural Network Builder")



layer_type = st.selectbox("Select Layer Type:", ["Dense Layer", "Activation Layer"])

if layer_type=="Dense Layer":
    neurons = int(st.number_input("Number of Neurons:"))
elif layer_type=="Activation Layer":
    activation = st.selectbox("Activation Function:", ["ReLU", "Sigmoid"])



if st.button("Add Layer"):
    
    if layer_type == "Dense Layer":
        if nn.is_empty()==True:
            
            layer=(DenseLayer(len(input_features_normalized[0]),neurons))
            st.session_state.nn_linklist.append(layer)
            
            st.session_state.nn_structure.append({"type": "Dense", "neurons": neurons})
        else:
           
            layer=(DenseLayer(None,neurons))
            st.session_state.nn_linklist.append(layer)
            st.session_state.nn_structure.append({"type": "Dense", "neurons": neurons})
    elif layer_type == "Activation Layer":
        activation_functions={"ReLU":(relu,relu_derivative),"Sigmoid":(sigmoid,sigmoid_derivative)}
        
        activation_data=activation_functions.get(activation)
        if activation_data:
            act_function=activation_data[0]
            act_derivative=activation_data[1]
        layer=(ActivationLayer(act_function,act_derivative))
        st.session_state.nn_linklist.append(layer)
        
        st.session_state.nn_structure.append({"type": "Activation", "activation": activation})

# Display added layers
st.markdown("### Current Neural Network Structure:")
if st.session_state.nn_structure:
    for idx, layer in enumerate(st.session_state.nn_structure):
        if layer["type"] == "Dense":
            st.markdown(f"**Layer {idx + 1}: Dense Layer ({layer['neurons']} neurons)**")
        elif layer["type"] == "Activation":
            st.markdown(f"**Layer {idx + 1}: Activation Layer ({layer['activation']})**")
else:
    st.warning("No layers added yet.")

def visualize_nn_interactive(structure):
    """Interactive neural network visualization with full connections between layers."""
    fig = go.Figure()

    layer_spacing = 130  # Space between layers
    neuron_spacing = 50 # Reduced space between neurons within a layer
    neuron_radius = 70 # Radius for neurons

    positions = []  # Store (x, y) positions of neurons for connections

    # Draw layers
    for i, layer in enumerate(structure):
        x = i * layer_spacing
        neurons = 3  # Fixed number of neurons per layer for visualization
        y_positions = np.linspace(-neurons * neuron_spacing / 2, neurons * neuron_spacing / 2, neurons)
        positions.append([(x, y) for y in y_positions])

        for y in y_positions:
            # Add neuron circles
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=neuron_radius, color="white", line=dict(color="black", width=2)),
                text=layer.get("type", ""),  # Show the type of the layer
                textfont=dict(color="black", size=12),  # Darker text for labels
                textposition="bottom center",
                showlegend=False
            ))

    # Add connections between neurons in adjacent layers
    for layer_idx in range(len(positions) - 1):
        current_layer = positions[layer_idx]
        next_layer = positions[layer_idx + 1]

        for (x1, y1) in current_layer:
            for (x2, y2) in next_layer:
                # Add connection line
                fig.add_trace(go.Scatter(
                    x=[x1, x2], y=[y1, y2],
                    mode="lines",
                    line=dict(color="gray", width=1),
                    showlegend=False
                ))

    # Customize layout
    fig.update_layout(
        title="Interactive Neural Network Visualization",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        height=600,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig

# Render visualization if layers exist
if st.session_state.nn_structure:
    fig = visualize_nn_interactive(st.session_state.nn_structure)
    st.plotly_chart(fig, use_container_width=True)

# Step 4: Training Configuration
st.markdown("#### Step 4: Training Configuration")
batch_size = int(st.number_input("Batch Size:", value=32, min_value=1, step=1))
epochs = int(st.number_input("Number of Epochs:", value=1000, min_value=1, step=1))
learning_rate = st.number_input("Learning Rate:", value=0.01, min_value=0.0001, step=0.001, format="%.4f")



# Step 5: Train the Model
st.markdown("#### Step 5: Train Your Neural Network")
correct_predictions = 0
total_predictions = 0

if st.button("Start Training 🚀"):
    if st.session_state.nn_linklist:
        for layer in st.session_state.nn_linklist:
            nn.insert_node(layer)

    st.markdown("### Training in Progress...")
    progress_bar = st.progress(0)
    loss_display = st.empty()
    
    for i in range(epochs):

        batch_queue=batch_split1(input_features_normalized,output_features_normalized,batch_size)
        progress_bar.progress(min((epochs + 1) / epochs, 1.0))

        while batch_queue.is_empty() !=True:
            batch=batch_queue.dequeue()
            input_array=batch.feature_array
            output_array=batch.output_array
            predicted_output = nn.forward_propogation(input_array) 
            

            if i==epochs-1:
                st.session_state.prediction.append(predicted_output)
            
            error = output_array - predicted_output  

            loss = mse_loss(output_array, predicted_output) 
            
            threshold = 0.05  # Define acceptable error threshold
            correct_predictions += np.sum(np.abs(output_array - predicted_output) <= threshold)
            total_predictions += output_array.size

            gradients_stack = nn.backward_propagation(error) 
            nn.update_parameters(gradients_stack, learning_rate) 
        loss_display.markdown(f"<div class='loss-box'>Epoch {i+1} | Loss: {loss:.4f}</div>", unsafe_allow_html=True)
    st.success("Model trained successfully! 🎉")
    layers=nn.get_all_nodes()
    for layer in layers:
        st.session_state.trained_model_layers.append(layer)













    

# After training, options for evaluation
st.markdown("### What Would You Like to Do Next?")
action = st.selectbox("Choose Action:", ["Check Model Accuracy", "Make Predictions"])

if action == "Check Model Accuracy":

    #Use a loop or accumulate predictions to match the full dataset size.
    accuracy_percentage=0
    if correct_predictions>0 and total_predictions>0:
        accuracy_percentage = (correct_predictions / total_predictions) * 100
    st.write("### Model Accuracy:", accuracy_percentage)
elif action == "Make Predictions":

    for layer in st.session_state.trained_model_layers:
        nn_trained.insert_node_without_updates(layer)
    st.markdown("Enter new data to make predictions:")
    user_input = st.text_area("Enter input features (comma-separated):")
    if st.button("Predict"):
        inputs = np.array([float(i) for i in user_input.split(",")]).reshape(1,-1)
        
        if inputs.size>0:
            print(inputs.shape)
            print(inputs)
        inputs_normalized =(inputs - input_min) / (input_max - input_min)
        if inputs_normalized is not None:
            print(inputs_normalized.shape)
            print(inputs_normalized)

        predicted_output_normalized = nn_trained.forward_propogation(inputs_normalized)
        prediction=denormalize(predicted_output_normalized,output_min,output_max)

        
        st.success(f"**Prediction:** {prediction.flatten()[0]:.2f}")



        
