# ROCm 7.2 WSL2 설치 가이드

> 환경: Ubuntu 24.04 (Noble) / WSL2 / AMD RX 9070 XT (gfx1201)  
> 작성일: 2026-03-27

---

## 1. 사전 확인

### 현재 ROCm 버전 확인
```bash
cat /opt/rocm/.info/version
```

### Windows 드라이버 버전 확인 (PowerShell)
```powershell
Get-WmiObject Win32_VideoController | Select Name, DriverVersion
```
> ROCm 7.2는 `/dev/kfd` 장치를 사용합니다. AMD Adrenalin **25.x 이상** 드라이버가 필요합니다.

---

## 2. APT 리포지토리 변경

### 기존 리포 (6.4.2)
```
deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.4.2 noble main
```

### 7.2로 변경
```bash
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main' \
  | sudo tee /etc/apt/sources.list.d/rocm.list
```

### 패키지 목록 갱신
```bash
sudo apt update -o Dir::Etc::sourcelist="sources.list.d/rocm.list" \
                -o Dir::Etc::sourceparts="-" \
                -o APT::Get::List-Cleanup="0"
```

---

## 3. 충돌 패키지 제거

ROCm 7.2의 표준 `hsa-rocr`과 WSL2 전용 `hsa-runtime-rocr4wsl-amdgpu`는 **Conflicts** 관계입니다.

| 패키지 | 설명 |
|---|---|
| `hsa-runtime-rocr4wsl-amdgpu 25.10` | 구형 WSL2 전용 HSA (Provides: hsa-rocr = **1.15.0**) |
| `hsa-rocr 1.18.0.70200` | ROCm 7.2 표준 HSA (Provides: hsa-rocr = **1.18.0**) |

```bash
# 구형 WSL2 전용 패키지 제거 및 ROCm 7.2 전체 업그레이드 (동시 진행)
sudo apt remove -y hsa-runtime-rocr4wsl-amdgpu
```

> APT가 자동으로 `hsa-rocr 1.18.0` 설치 및 ROCm 전체 업그레이드를 수행합니다.  
> 다운로드 용량: 약 **6.5 GB**, 추가 디스크 사용: 약 **3.5 GB**

---

## 4. 설치 확인

```bash
cat /opt/rocm/.info/version
# 7.2.0

rocminfo 2>&1 | head -5
```

### ROCm 7.2부터 WSL2 변경사항

| 항목 | 6.4.x (구형) | 7.2.0 (신형) |
|---|---|---|
| GPU 접근 장치 | `/dev/dxg` | `/dev/kfd` + `/dev/dri/` |
| HSA 패키지 | `hsa-runtime-rocr4wsl-amdgpu` | `hsa-rocr` (표준) |
| Windows 드라이버 요구 | Adrenalin 24.x | Adrenalin **25.x 이상** |

> **중요**: `/dev/kfd`가 없으면 `rocminfo`가 `hsa_init Failed`를 출력합니다.  
> Windows AMD 드라이버를 최신 버전으로 업데이트 후 `wsl --shutdown` → WSL 재시작 필요.

---

## 5. PyTorch ROCm 7.2 wheel 설치

### pyproject.toml 변경사항

```toml
# 변경 전
[[tool.uv.index]]
name = "pytorch-rocm64"
url = "https://download.pytorch.org/whl/rocm6.4"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-rocm64" }
torchvision = { index = "pytorch-rocm64" }
pytorch-triton-rocm = { index = "pytorch-rocm64" }  # ← 구 패키지명
```

```toml
# 변경 후
[[tool.uv.index]]
name = "pytorch-rocm72"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-rocm72" }
torchvision = { index = "pytorch-rocm72" }
triton-rocm = { index = "pytorch-rocm72" }  # ← ROCm 7.2부터 패키지명 변경
```

> ROCm 7.2부터 `pytorch-triton-rocm` → **`triton-rocm`** 으로 패키지명이 변경됐습니다.

### dependencies 변경
```toml
# 변경 전
"pytorch-triton-rocm>=3.4.0; sys_platform == 'linux'"

# 변경 후
"triton-rocm>=3.6.0; sys_platform == 'linux'"
```

### 설치
```bash
rm -f uv.lock
uv sync
```

### 설치 결과
```
- torch==2.9.1+rocm6.4      →  + torch==2.11.0+rocm7.2
- torchvision==0.24.1+rocm6.4 →  + torchvision==0.26.0+rocm7.2
- pytorch-triton-rocm==3.5.1 →  + triton-rocm==3.6.0
```

---

## 6. GPU 동작 테스트

```bash
HSA_OVERRIDE_GFX_VERSION=12.0.1 uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'HIP: {torch.version.hip}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    t = torch.tensor([1.0]).cuda()
    print(f'Tensor: {t}  ← GPU 정상 동작!')
"
```

> `HSA_OVERRIDE_GFX_VERSION=12.0.1` — RX 9070 XT (gfx1201) 강제 지정  
> `PYTORCH_ROCM_ARCH=gfx1201` — 빌드 아키텍처 지정 (필요시)

---

## 7. 최종 환경 요약

| 항목 | 버전 |
|---|---|
| OS | Ubuntu 24.04 Noble (WSL2) |
| WSL2 커널 | 6.6.87.2-microsoft-standard-WSL2 |
| ROCm | **7.2.0** |
| hsa-rocr | 1.18.0.70200 |
| PyTorch | **2.11.0+rocm7.2** |
| torchvision | 0.26.0+rocm7.2 |
| triton-rocm | 3.6.0 |
| GPU | AMD RX 9070 XT (gfx1201) |

---

## 8. 트러블슈팅

### `/dev/kfd` 없음 (WSL2)
```
WSL environment detected.
hsa_init Failed, possibly no supported GPU devices
```
**해결**: Windows에서 AMD Adrenalin 드라이버 25.x 이상으로 업데이트 후 WSL 재시작
```powershell
# PowerShell (관리자)
wsl --shutdown
wsl
```

### `triton-rocm` 패키지를 찾을 수 없음
ROCm 7.2 이상에서는 `pytorch-triton-rocm` 대신 `triton-rocm`을 사용합니다.  
의존성과 `[tool.uv.sources]` 모두 변경해야 합니다.

### uv sync 의존성 해결 실패
```
No solution found: torch==2.11.0+rocm7.2 depends on triton-rocm==3.6.0
```
`pyproject.toml`에서 `pytorch-triton-rocm` → `triton-rocm`으로 변경 후 재시도.
