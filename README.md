# llm_infra

## CAI Tuning
+ Collective Constitutional AI: Aligning a Language Model with Public Input
+ https://www.anthropic.com/research/collective-constitutional-ai-aligning-a-language-model-with-public-input
+ data : [heegyu/hh-rlhf-ko](https://huggingface.co/datasets/heegyu/hh-rlhf-ko)
+ UI를 통한 학습 파이프라인
  + gpu_memory_streamlit.py : streamlit 기반 ui
  + main.py : 학습을 위한 fastapi 서버    


## Fine Tuning
+ fsdp + lora 학습 코드 

## Interface Sercer
+ vllm OpenAI-Compatible Server 활용을 위한 클라우드 베이스 코드
  + 해당 코드를 기반으로 해서 RAG 및 로직 추가
 
## Server Text
+ locust GPU 부하테스트 코드
