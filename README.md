# llm_infra

## CAI Tuning
+ Collective Constitutional AI: Aligning a Language Model with Public Input
+ https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input
+ data : [heegyu/hh-rlhf-ko](https://huggingface.co/datasets/heegyu/hh-rlhf-ko)
+ UI를 통한 학습 파이프라인
  + gpu_memory_streamlit.py : streamlit 기반 ui
  + main.py : 학습을 위한 fastapi 서버
  + ![image](https://github.com/user-attachments/assets/040ad829-a6cd-46ac-9ac8-fb511220fac8)
  + ![image](https://github.com/user-attachments/assets/0d8a8995-2df9-4e2b-bda4-a4d34cb38ad6)




## Fine Tuning
+ fsdp + lora 학습 코드 

## Interface Sercer
+ vllm OpenAI-Compatible Server 활용을 위한 클라우드 베이스 코드
  + 해당 코드를 기반으로 해서 RAG 및 로직 추가
 
## Server Text
+ locust GPU 부하테스트 코드
  + <img width="1564" alt="KakaoTalk_Photo_2024-12-11-06-38-58 001" src="https://github.com/user-attachments/assets/af11cca2-1144-43af-9722-240c6b557712" />


