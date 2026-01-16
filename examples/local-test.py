"""
本地測試腳本 - 使用 .env 檔案儲存 HF_TOKEN

使用方式：
1. 複製 .env.example 到 .env
2. 在 .env 中填入你的 HF_TOKEN
3. 執行: python examples/local-test.py
"""

import os
import sys
import time
import psutil
import shutil
import warnings
from datetime import datetime
from pathlib import Path

# 抑制 transformers 的警告訊息
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# 顏色代碼
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 進度顯示工具函數
def print_step(step, message, **kwargs):
    """顯示步驟進度"""
    timestamp = kwargs.get('timestamp', False)
    memory = kwargs.get('memory', False)
    disk = kwargs.get('disk', False)

    prefix = f"{Colors.BLUE}[{step}]{Colors.NC}"

    # 時間戳記
    time_str = ""
    if timestamp:
        time_str = f" {Colors.CYAN}({datetime.now().strftime('%H:%M:%S')}){Colors.NC}"

    print(f"{prefix} {message}{time_str}")

    # 記憶體使用
    if memory:
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_percent = mem.percent
        print(f"   💾 記憶體: {mem_used_gb:.1f}GB / {mem_total_gb:.1f}GB ({mem_percent}%)")

    # 磁碟空間
    if disk:
        disk_usage = shutil.disk_usage("/")
        disk_free_gb = disk_usage.free / (1024**3)
        disk_total_gb = disk_usage.total / (1024**3)
        disk_percent = (disk_usage.used / disk_usage.total) * 100
        print(f"   💿 磁碟空間: {disk_free_gb:.1f}GB 可用 / {disk_total_gb:.1f}GB 總計 ({100-disk_percent:.1f}% 可用)")

def print_success(message, detail=None):
    """顯示成功訊息"""
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
    if detail:
        print(f"   {Colors.CYAN}{detail}{Colors.NC}")

def print_error(message, detail=None):
    """顯示錯誤訊息"""
    print(f"{Colors.RED}❌ {message}{Colors.NC}")
    if detail:
        print(f"   {detail}")

def print_warning(message):
    """顯示警告訊息"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")

def check_model_cache(model_id):
    """檢查模型是否已經下載到快取"""
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()

        # 尋找模型
        for repo in cache_info.repos:
            if model_id in repo.repo_id:
                # 計算模型大小
                total_size = sum(revision.size_on_disk for revision in repo.revisions)
                size_gb = total_size / (1024**3)

                # 檢查是否有未完成的下載
                incomplete_files = []
                cache_path = Path.home() / ".cache" / "huggingface" / "hub"
                model_cache_dir = cache_path / f"models--{model_id.replace('/', '--')}"

                if model_cache_dir.exists():
                    incomplete_files = list(model_cache_dir.rglob("*.incomplete"))

                if incomplete_files:
                    return {
                        "cached": False,
                        "partial": True,
                        "size_gb": size_gb,
                        "incomplete_count": len(incomplete_files)
                    }
                else:
                    return {
                        "cached": True,
                        "partial": False,
                        "size_gb": size_gb
                    }

        # 模型未找到
        return {"cached": False, "partial": False, "size_gb": 0}

    except Exception as e:
        # 如果檢查失敗，假設未下載
        return {"cached": False, "partial": False, "size_gb": 0, "error": str(e)}

def load_env():
    """載入 .env 檔案"""
    env_file = project_root / ".env"

    if not env_file.exists():
        print("❌ .env 檔案不存在")
        print()
        print("請執行以下步驟：")
        print("1. 複製 .env.example 到 .env:")
        print("   cp .env.example .env")
        print()
        print("2. 編輯 .env 並填入你的 HF_TOKEN:")
        print("   # 從 https://huggingface.co/settings/tokens 取得")
        print("   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        return False

    # 讀取 .env 檔案
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

    return True

def test_token():
    """測試 HF_TOKEN 是否有效"""
    token = os.getenv("HF_TOKEN")

    if not token or token.startswith("hf_xxx"):
        print("❌ HF_TOKEN 未設定或使用預設值")
        print("請在 .env 檔案中設定有效的 HF_TOKEN")
        return False

    print("✅ HF_TOKEN 已設定")
    print(f"   Token: {token[:10]}...{token[-5:]}")

    try:
        from huggingface_hub import login
        login(token=token)
        print("✅ Hugging Face 認證成功")
        return True
    except Exception as e:
        print(f"❌ Hugging Face 認證失敗: {e}")
        return False

def test_model_access():
    """測試模型存取"""
    print("\n測試 TranslateGemma 模型存取...")

    try:
        from transformers import AutoTokenizer

        MODEL_ID = os.getenv("MODEL_ID", "google/translategemma-4b-it")
        print(f"載入 tokenizer: {MODEL_ID}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print(f"✅ Tokenizer 載入成功")
        print(f"   詞彙表大小: {len(tokenizer)}")
        return True

    except Exception as e:
        print(f"❌ 模型存取失敗: {e}")
        print()
        print("可能的原因：")
        print("1. 尚未申請 TranslateGemma 存取權限")
        print("   前往: https://huggingface.co/google/translategemma-4b-it")
        print("2. Token 權限不足（需要 Read 權限）")
        return False

def test_translation():
    """測試翻譯功能"""
    print("\n" + "="*80)
    print_step("4/4", "開始翻譯功能測試", timestamp=True, memory=True, disk=True)
    print("="*80)

    start_time = time.time()

    try:
        # 設定 transformers 日誌等級，減少不必要的警告
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        # 額外設定 logging level
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        MODEL_ID = os.getenv("MODEL_ID", "google/translategemma-4b-it")

        # 步驟 1: 載入 Tokenizer
        print()
        print_step("4.1", f"載入 Tokenizer: {MODEL_ID}", timestamp=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        print_success("Tokenizer 載入成功", f"詞彙表大小: {len(tokenizer):,}")

        # 步驟 2: 檢查模型快取並載入
        print()
        print_step("4.2", "檢查模型快取狀態", timestamp=True)

        cache_status = check_model_cache(MODEL_ID)

        if cache_status.get("cached"):
            # 模型已完整下載
            print_success("模型已在快取中",
                         f"大小: {cache_status['size_gb']:.1f} GB, 無需重新下載")
        elif cache_status.get("partial"):
            # 有未完成的下載
            print_warning(f"發現 {cache_status['incomplete_count']} 個未完成的下載")
            print(f"   目前已下載: {cache_status['size_gb']:.1f} GB")
            print(f"   建議先執行清理: ./run-examples.sh cleanup")
        else:
            # 模型未下載
            print_warning("模型未下載，將從 Hugging Face 下載（約 8.6 GB）")

            # 檢查磁碟空間是否足夠
            disk = shutil.disk_usage("/")
            free_gb = disk.free / (1024**3)

            if free_gb < 10:
                print_error(f"磁碟空間不足！僅剩 {free_gb:.1f} GB",
                           "建議至少有 12 GB 可用空間")
                print(f"   請執行: ./run-examples.sh cleanup")
                return False

        print()
        print_step("4.2b", "載入模型到記憶體（可能需要幾分鐘）",
                   timestamp=True, memory=True, disk=True)

        # 檢查空間：即使模型已下載，載入時仍需要臨時空間
        disk = shutil.disk_usage("/")
        free_gb = disk.free / (1024**3)

        # 計算所需空間
        if cache_status.get("cached"):
            # 模型已下載，只需要較少的臨時空間（2-3 GB）
            required_gb = 3.0
            space_purpose = "載入臨時空間"
        else:
            # 需要下載模型（8.6 GB）+ 臨時空間（2-3 GB）
            required_gb = 12.0
            space_purpose = "下載 + 載入"

        if free_gb < required_gb:
            print_error(
                f"磁碟空間不足！僅剩 {free_gb:.1f} GB",
                f"建議至少有 {required_gb:.1f} GB 可用空間（{space_purpose}）"
            )
            print()
            print(f"{Colors.CYAN}解決方案：{Colors.NC}")
            print(f"   1. 執行清理: ./run-examples.sh cleanup")
            print(f"   2. 清理系統暫存: sudo rm -rf /private/var/tmp/*")
            print(f"   3. 改用 Colab（推薦）")
            return False

        if not cache_status.get("cached"):
            print_warning("首次下載時 Hugging Face 會顯示進度條")
            print()

        model_start_time = time.time()

        # 檢查可用記憶體，決定載入策略
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)

        if available_mem_gb < 10:
            # 記憶體不足，使用 CPU-only 模式（較慢但更穩定）
            print_warning(f"可用記憶體不足 ({available_mem_gb:.1f}GB < 10GB)")
            print(f"   {Colors.CYAN}使用 CPU-only 模式（較慢但更穩定）{Colors.NC}")
            device_map = "cpu"
            torch_dtype = torch.float32  # CPU 不支援 bfloat16
        else:
            # 記憶體充足，使用 auto (MPS 或 CUDA)
            device_map = "auto"
            torch_dtype = torch.bfloat16

        print(f"   載入配置: device_map={device_map}, dtype={torch_dtype}")
        print()

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True  # 減少臨時檔案和記憶體使用
        )
        model_load_time = time.time() - model_start_time

        print()
        device_info = f"device: {model.device}, dtype: {model.dtype}"
        print_success("模型載入成功",
                     f"{device_info}, 耗時: {model_load_time:.1f} 秒")

        # 顯示模型載入後的記憶體狀態
        mem = psutil.virtual_memory()
        print(f"   💾 當前記憶體使用: {mem.used / (1024**3):.1f}GB ({mem.percent}%)")

        # 步驟 3: 準備翻譯
        print()
        print_step("4.3", "準備翻譯測試（英文→繁體中文）", timestamp=True)
        text = "Hello, world!"
        print(f"   原文: {text}")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "source_lang_code": "en",
                        "target_lang_code": "zh-TW"
                    }
                ]
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(model.device)

        print(f"   輸入 tokens: {inputs.shape[1]}")

        # 步驟 4: 執行翻譯
        print()
        print_step("4.4", "執行翻譯推理", timestamp=True)

        gen_start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                # 明確設定為 None 以避免警告
                top_p=None,
                top_k=None
            )
        gen_time = time.time() - gen_start_time

        # 只解碼新生成的 tokens（不包括輸入 prompt）
        # 這樣就不會顯示 system prompt
        generated_tokens = outputs[0][inputs.shape[1]:]
        translation = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 如果需要看完整輸出（除錯用）
        if os.getenv("DEBUG_TRANSLATION"):
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n{Colors.YELLOW}[DEBUG] 完整輸出:{Colors.NC}")
            print(full_output)

        print()
        print(f"{Colors.BOLD}翻譯結果:{Colors.NC}")
        print(f"   原文: {text}")
        print(f"   譯文: {translation}")
        print(f"   推理時間: {gen_time:.2f} 秒")
        print(f"   生成 tokens: {len(generated_tokens)} (總計: {outputs.shape[1]})")
        print(f"   生成速度: {len(generated_tokens) / gen_time:.1f} tokens/秒")

        # 總結
        total_time = time.time() - start_time
        print()
        print("="*80)
        print_success("翻譯測試完成", f"總耗時: {total_time:.1f} 秒")
        print("="*80)

        return True

    except Exception as e:
        print()
        print_error("翻譯測試失敗", str(e))
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print()
    print("="*80)
    print(f"{Colors.BOLD}TranslateGemma 本地測試{Colors.NC}")
    print("="*80)
    print()

    # 顯示系統資訊
    mem = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    print(f"💻 系統資訊:")
    print(f"   記憶體: {mem.total / (1024**3):.1f}GB 總計, {mem.available / (1024**3):.1f}GB 可用")
    print(f"   磁碟: {disk.free / (1024**3):.1f}GB 可用 / {disk.total / (1024**3):.1f}GB 總計")
    print()

    # 1. 載入 .env
    print_step("1/4", "載入 .env 檔案", timestamp=True)
    if not load_env():
        return 1
    print_success(".env 檔案載入成功")
    print()

    # 2. 測試 token
    print_step("2/4", "測試 HF_TOKEN", timestamp=True)
    if not test_token():
        return 1
    print()

    # 3. 測試模型存取
    print_step("3/4", "測試模型存取", timestamp=True)
    if not test_model_access():
        return 1
    print()

    # 4. 測試翻譯（可選，因為載入模型需要較長時間）
    print(f"{Colors.YELLOW}{'='*80}{Colors.NC}")
    print(f"{Colors.YELLOW}步驟 4/4: 翻譯功能測試（可選）{Colors.NC}")
    print(f"{Colors.YELLOW}{'='*80}{Colors.NC}")
    print()

    # 先檢查模型是否已下載
    MODEL_ID = os.getenv("MODEL_ID", "google/translategemma-4b-it")
    cache_status = check_model_cache(MODEL_ID)

    # 根據模型狀態顯示不同訊息
    if cache_status.get("cached"):
        # 模型已下載
        print_success(f"模型已在快取中（{cache_status['size_gb']:.1f} GB）")
        print(f"   {Colors.CYAN}只需要載入到記憶體，無需重新下載{Colors.NC}")
        required_disk = 3.0  # 只需要臨時空間
        required_mem = 10.0
        print(f"   建議至少有 {Colors.BOLD}{required_disk:.0f} GB 可用磁碟空間{Colors.NC}（載入臨時空間）")
        print(f"   和 {Colors.BOLD}{required_mem:.0f} GB 可用記憶體{Colors.NC}")
    else:
        # 模型未下載
        print_warning("模型尚未下載，此步驟會下載完整模型（約 8-9 GB）並載入到記憶體")
        required_disk = 12.0  # 需要下載 + 臨時空間
        required_mem = 10.0
        print(f"   建議至少有 {Colors.BOLD}{required_disk:.0f} GB 可用磁碟空間{Colors.NC}")
        print(f"   和 {Colors.BOLD}{required_mem:.0f} GB 可用記憶體{Colors.NC}")

    print()

    # 檢查空間是否足夠（使用動態需求）
    free_disk_gb = disk.free / (1024**3)
    free_mem_gb = mem.available / (1024**3)

    if free_disk_gb < required_disk:
        print_warning(f"磁碟空間不足（僅 {free_disk_gb:.1f}GB），可能會失敗")
        print(f"   {Colors.CYAN}建議執行: ./run-examples.sh cleanup{Colors.NC}")
    if free_mem_gb < required_mem:
        print_warning(f"可用記憶體不足（僅 {free_mem_gb:.1f}GB），可能會失敗")

    print()
    response = input("是否執行翻譯測試？[y/N]: ")
    if response.lower() == 'y':
        if not test_translation():
            return 1
    else:
        print()
        print_warning("跳過翻譯測試")

    print()
    print("="*80)
    print_success("所有測試完成！")
    print("="*80)
    print()
    print(f"{Colors.CYAN}下一步：{Colors.NC}")
    print(f"   1. 在 Colab 中開啟 translategemma-colab.ipynb")
    print(f"   2. 或部署到 Cloud Run: cd cloudrun && ./deploy.sh")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
