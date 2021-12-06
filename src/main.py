import streamlit as st
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from thai2transformers.preprocess import process_transformers
from tokenizers import Tokenizer, AddedToken
import shap

model_path = './model'

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def hash_tokenizer(_):
    return model_path

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1, hash_funcs={Tokenizer: hash_tokenizer, AddedToken: hash_tokenizer})
def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")

    pipeline = TextClassificationPipeline(model = model, tokenizer = tokenizer)

    return pipeline


title = "Chula Course Faculty prediction demo"
st.title(title)


txt = st.text_area('Text to analyze', 'บทบาทของเทคโนโลยีสารสนเทศในการแก้ปัญหาทางธุรกิจ แนวคิดและพื้นฐานเทคนิคของศาสตร์เทคโนโลยีสารสนเทศ การวางแผน การพัฒนาและการจัดการของระบบประยุกต์สารสนเทศโดยคอมพิวเตอร์')
processed_text = process_transformers(txt)

pipeline = load_pipeline()

prediction = pipeline(processed_text)
st.json(prediction)

explainer = shap.Explainer(pipeline)

with st.spinner(text="Interpreting model by SHAP. we don't have GPU so this will take a 1-2 minutes"):
    shap_values = explainer([processed_text])
    st_shap(shap.text_plot(shap_values[0, :, 2], matplotlib=True))
st.success()

# st.text(shap_values)

# shap.text_plot(shap_values[0, :, int(enc.transform([['25']])[0][0])])