import streamlit as st
import requests
import os 


url = "your gpu ip"

st.set_page_config(layout="wide")
model_input = st.text_input("모델 입력", value="")

col1, col2 = st.columns(2)
show_gpu_data = False
gpu_data = None


def fetch_gpu_data():
    api_url = os.path.join(url, "gpu-info/")  
    response = requests.get(api_url)
    
    if response.status_code == 200:
        return response.json()['gpu_info']
    else:
        st.error("API 요청 실패: 데이터를 가져올 수 없습니다.")
        return []

            
def show_gpu(gpu_data):
    for i in range(0, len(gpu_data), 4):
        cols = st.columns(4)
        for j, col in enumerate(cols):
            gpu = gpu_data[i + j]
            with col:
                st.header(f"GPU {gpu['index']}")  
                st.text(f"모델: {gpu['name']}")  
                st.text(f"전체 메모리: {gpu['total_memory']} MB")  
                st.text(f"사용 메모리: {gpu['used_memory']} MB ({round(int(gpu['used_memory']) / float(gpu['total_memory']) * 100, 2)}%)")  # Used memory and utilization
                st.text(f"여유 메모리: {gpu['free_memory']} MB")  
                st.progress(int(gpu['used_memory']) / float(gpu['total_memory']))  

def calculate_gpus():
    gpu_data = fetch_gpu_data()
    gpus = []
    for gpu in gpu_data:
        if round(int(gpu['used_memory']) / float(gpu['total_memory']) * 100, 2) < 5 :
            gpus.append(gpu['index'])
    return gpus, gpu_data


with col1:
    if st.button("현재 사용량 보기"):
        show_gpu_data = True

with col2:
    if st.button("학습"):
        model_input = model_input.strip()
        if model_input:
            gpu_index, gpu_data = calculate_gpus()
            show_gpu_data = True
            gpus = ",".join(gpu_index)
            model_id = model_input
            st.write(f"'{model_input}' 모델로 학습을 시작합니다!") 
            st.write(f"'{gpus}' gpu로 학습을 시작합니다!") 
            api_url = os.path.join(url, "start-training/")  
            data = {
                'model_id': model_input,
                'gpus': gpus
            }
            response = requests.post(api_url, json=data)
            
            print(response.status_code)  # 상태 코드
            print(response.json())


if show_gpu_data:
    if gpu_data is None:
        gpu_data = fetch_gpu_data()
    show_gpu(gpu_data)