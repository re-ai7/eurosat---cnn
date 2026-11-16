# Shiny for Python (Express) app: upload a satellite image and get the EuroSAT class prediction.
# Run from the repository root with:  shiny run app.py

from pathlib import Path

from shiny import reactive
from shiny.express import input, render, ui
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd
from model import sat_CNN, inv_map, mean, std

# paths are resolved relative to this file, so the app runs from any machine
APP_DIR = Path(__file__).resolve().parent

# model instance
model = sat_CNN()
state_dict = torch.load(APP_DIR / 'assets' / 'weights' / 'best_model.pth', map_location=torch.device('cpu'))  # cpu is sufficient to compute the model quickly
_ = model.load_state_dict(state_dict)
_ = model.eval()


# shiny layout starts here
ui.page_opts(title="Satellite Image Classifier")

# retrieving model prediction for image
@reactive.calc
@reactive.event(input.predict) # reactive to button
def prediction():
    file = input.image() # returns list of uploaded images. each a dict with (name: , ..., datapath: )
    if file is None:
        return None

    img = Image.open(file[0]["datapath"]).convert("RGB")  # force 3 channels (e.g. PNGs with alpha)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # same as training!
    ])
    tensor = transform(img).unsqueeze(0)  # add batch dim (1, 3, 64, 64) for the model being able to process

    # pass image through model
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1) # logits through softmax to probabilities
        pred_idx = probabilities.argmax(dim=1).item()

        image_pred = inv_map[pred_idx]
        image_prob = probabilities.max(dim=1).values.item()

    return image_pred, image_prob, probabilities

# sidebar with image upload and model pass button
with ui.sidebar(title='Upload Image'):
    ui.input_file(id="image", label="Select satellite Image")
    ui.input_action_button(id='predict', label='Predict class')


# multi layout column
with ui.layout_columns():
    # display image
    with ui.card():
        ui.card_header("Image")
        @render.image
        def show_uploaded():
            file = input.image() # returns list of uploaded images. each a dict with (name: , ..., datapath: )
            if file is None:
                return None
            return {"src": file[0]["datapath"], 'width': '100%'} # get datapath of the uploaded image

    # render dataframe with classes and probabilities
    with ui.card():
        ui.card_header("Class Probabilities")
        @render.data_frame
        def img2():
            result = prediction()
            if result is None: # handling error when no image was uploaded
                return None
            _, _, probabilities = result
            probs_list = probabilities.squeeze().tolist()  # tensor 2d [1,10] to list size 10
            df = pd.DataFrame({'Class': inv_map.values(), 'Probability': probs_list})
            df['Probability'] = df['Probability'].round(4)
            return df


# spacing
ui.div(style="height: 1.5rem;")

# lower container with prediction
with ui.card():
    ui.card_header("Prediction")
    @render.text
    def pred_output():
        result = prediction()
        if result is None: # handling error when no image was uploaded
            return 'Upload image first!'
        image_pred, image_prob, _ = result
        return f'Predicted Class: {image_pred} (Confidence: {image_prob*100:.2f}% )'
