import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')
labels = learn.dls.vocab

def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}


title = "Is it Mia or Brian?"
description = "A Mia detector trained with fastai. Created as an exercise for fastbook."
article="<p style='text-align: center'><a href='https://levignelabs.com' target='_blank'>Lavigne Labs</a></p>"
examples = ['test images/b1.jpg','test images/b2.jpg','test images/m1.jpg']
interpretation='default'
enable_queue=True
gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch(share=True)



