import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Загрузка модели и токенизатора
@st.cache_resource
def load_model():
    model_path = "models/llama3.2-testcase"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

# Функция предобработки ввода
def preprocess_input(scenario):
    return f"test_scenario:\n{scenario}\ntest_steps:\n"

# Генерация шагов тест-кейса
def generate_test_steps(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=128,
            num_return_sequences=1
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Создание интерфейса
st.title("Генератор шагов тест-кейсов на основе Llama3.2")

test_scenario = st.text_area("Введите описание тестового сценария:", height=150)
generate_button = st.button("Сгенерировать шаги тест-кейса")

if generate_button and test_scenario.strip():
    with st.spinner("Генерация шагов тест-кейса..."):
        processed_prompt = preprocess_input(test_scenario)
        generated_text = generate_test_steps(processed_prompt)
        
        # Извлекаем только сгенерированные шаги
        steps_start = generated_text.find("test_steps:") + len("test_steps:\n")
        test_steps = generated_text[steps_start:].strip()
        
        st.subheader("Сгенерированные шаги:")
        st.code(test_steps, language=None)
elif generate_button:
    st.warning("Пожалуйста, введите описание тестового сценария")