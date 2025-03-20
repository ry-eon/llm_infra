from locust import HttpUser, task, between
import json
import random
import logging
import time
from openai import OpenAI


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("locust_log.txt", mode="a", encoding="utf-8")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(console_handler)


class VLLMUser(HttpUser):
    host = "http://0.0.0.0:8000"
    wait_time = between(20, 25)

    prompts = [
        "안녕하세요, 반갑습니다.",
        "오늘 날씨가 어떤가요?",
        "인공지능에 대해 설명해주세요.",
        "맛있는 음식 추천해주세요.",
        "좋아하는 취미가 뭔가요?"
    ]

    def on_start(self):
        """ 테스트 시작 시 OpenAI API 클라이언트 초기화 """
        self.client = OpenAI(
            api_key="dummy-key",
            base_url=self.host + "/v1"
        )

    def get_random_prompt(self):
        """ 랜덤 프롬프트 선택 """
        return random.choice(self.prompts)

    @task
    def generate_text(self):
        start_time = time.time()
        try:
            response = self.client.completions.create(
                model="olympiad",
                prompt=self.get_random_prompt(),
                max_tokens=2048,
                temperature=0.7,
                top_p=0.95,
                stream=False
            )

            response_time = time.time() - start_time

            success_detail = (
                f"Success with completion ID: {response.id}, "
                f"Generated text length: {len(response.choices[0].text)}, "
                f"Response time: {response_time:.2f}s, "
                f"Total tokens: {response.usage.total_tokens}"
            )

            logger.info(success_detail)

            self.environment.events.request.fire(
                request_type="POST",
                name="/v1/completions",
                response_time=response_time * 1000,
                response_length=len(response.choices[0].text),
                exception=None,
            )

        except Exception as e:
            error_msg = f"Request failed: {type(e).__name__}, {str(e)}"
            logger.error(error_msg)

            self.environment.events.request.fire(
                request_type="POST",
                name="/v1/completions",
                response_time=time.time() - start_time,
                response_length=0,
                exception=e,
            )
