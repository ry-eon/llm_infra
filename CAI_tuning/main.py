from fastapi import FastAPI
import subprocess
import uvicorn
from dto import (
   ReqQuery,  
)


def get_gpu_info():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True
    )
    gpu_info = result.stdout.strip().split("\n")
    gpu_data = []
    for gpu in gpu_info:
        index, name, total_mem, used_mem, free_mem, utilization = gpu.split(", ")
        gpu_data.append({
            "index": index,
            "name": name,
            "total_memory": total_mem,
            "used_memory": used_mem,
            "free_memory": free_mem,
            "gpu_utilization": utilization
        })
        print(gpu.split(", "))
    return gpu_data

app = FastAPI()


@app.get("/")
def hello():
    return "hello fastapi"


@app.get("/gpu-info")
def gpu_info():
    gpu_data = get_gpu_info()
    return {"gpu_info": gpu_data}


@app.post("/start-training")
def start_training(req_query: ReqQuery) :
    model_id = req_query.model_id
    gpus = req_query.gpus
    print(gpus)
    print(f"CUDA_VISIBLE_DEVICES={gpus}")
    try:
        command = [
            f"CUDA_VISIBLE_DEVICES={gpus}",
            "torchrun",
            f"--nproc_per_node={len(gpus.split(','))}",
            "train_cai.py",
            f"--model_id={model_id}",
        ]
        
        subprocess.Popen(" ".join(command), shell=True)
        return {"status": "학습이 시작되었습니다.", "command": " ".join(command)}
    except Exception as e:
        return {"status": "에러 발생", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7811)