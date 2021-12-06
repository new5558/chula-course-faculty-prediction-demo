import streamlit as st
import streamlit.components.v1 as components
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from thai2transformers.preprocess import process_transformers
from tokenizers import Tokenizer, AddedToken
import shap

model_path = './model'
txt = ""

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.initjs()}</head><body>{plot}</body>"
    components.html(shap_html, height=height)

def hash_tokenizer(_):
    return model_path

@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1, hash_funcs={Tokenizer: hash_tokenizer, AddedToken: hash_tokenizer})
def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    tokenizer = AutoTokenizer.from_pretrained("./model")

    pipeline = TextClassificationPipeline(model = model, tokenizer = tokenizer)

    return pipeline


def main():
    title = "Chula Course Faculty prediction demo"
    st.set_page_config(page_title=title, page_icon="💎", layout="wide")
    _, center_column, c_ = st.columns((1, 2, 1))
    center_column.title(title)

    pipeline = load_pipeline()
    form = center_column.form("my_form")
    txt = form.text_area('Course description thai', 'บทบาทของเทคโนโลยีสารสนเทศในการแก้ปัญหาทางธุรกิจ แนวคิดและพื้นฐานเทคนิคของศาสตร์เทคโนโลยีสารสนเทศ การวางแผน การพัฒนาและการจัดการของระบบประยุกต์สารสนเทศโดยคอมพิวเตอร์')
    submit = form.form_submit_button(label="Predict Faculty")

    if submit:
        processed_text = process_transformers(txt)

        prediction = pipeline(processed_text)
        center_column.json(prediction)
        
        explainer = shap.Explainer(pipeline)
        shap_values = None
        with st.spinner(text="Interpreting model with SHAP. we don't have GPU so this will take a 1-2 minutes"):
            shap_values = explainer([processed_text])
        
        st_shap(shap.plots.text(shap_values[0, :, 2], display = False))

if __name__ == "__main__":
    main()


# st.text(shap_values)

# shap.text_plot(shap_values[0, :, int(enc.transform([['25']])[0][0])])