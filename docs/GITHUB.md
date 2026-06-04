# GitHub에 올리는 방법

현재 이 저장소는 이미 GitHub 원격이 연결되어 있습니다.

```text
origin  https://github.com/jhch1113/P-project.git
브랜치  Concatenated-Jetson (예시)
```

새 저장소를 만들거나, 기존 `P-project`에 푸시하면 됩니다.

---

## 1. 올리면 안 되는 것 (필수)

| 항목 | 이유 |
|------|------|
| **`.env`** | Muse MAC 등 개인 설정·비밀 (`.gitignore`에 포함됨) |
| **`*.keras` / `models/`** | 용량 큼(수십 MB~). GitHub 단일 파일 권장 100MB |
| **가상환경** `.venv/` | 재현 불필요, 용량만 증가 |

대신 **`.env.example`** 을 커밋합니다(값 없는 템플릿).

모델은 README/DEPLOYMENT에 “별도 다운로드 또는 `models/`에 복사”로 안내합니다.

---

## 2. 새 GitHub 저장소를 쓰는 경우

1. [github.com/new](https://github.com/new) 에서 저장소 생성 (Private 권장).
2. 로컬에서 원격 변경:

```bash
cd /path/to/drowsiness_project
git remote rename origin old-origin   # 기존 origin 유지 시 (선택)
git remote add origin https://github.com/<YOUR_USER>/<REPO>.git
```

또는 기존 `P-project`를 그대로 쓰면 `git remote -v`만 확인 후 push.

---

## 3. 커밋 & 푸시 (일반 절차)

```bash
cd drowsiness_project

# 상태 확인
git status

# 올릴 파일 스테이징 (.env는 자동 제외됨)
git add .
git status   # .env / *.keras 가 없는지 다시 확인

git commit -m "docs: 배포 가이드 및 Docker CPU compose 추가"

# 첫 푸시 또는 upstream 설정
git push -u origin Concatenated-Jetson
# main 브랜치를 쓰는 경우: git push -u origin main
```

GitHub 로그인:

- HTTPS: Personal Access Token (비밀번호 대신)
- SSH: `git@github.com:<user>/<repo>.git` + SSH 키 등록

---

## 4. 용량이 큰 파일 (이미 저장소에 있는 경우)

`deprecated/pp_nrsc/*.csv` 는 합쳐 **약 120MB+** 입니다. 이미 Git에 들어가 있으면 push는 되지만, **앞으로 제거**하려면:

```bash
git rm --cached deprecated/pp_nrsc/*.csv
echo "deprecated/**/*.csv" >> .gitignore
git commit -m "chore: 학습용 CSV는 Git에서 제외"
```

`third_party/librealsense` 도 용량이 큽니다. 팀 정책에 따라:

- 그대로 유지(현재처럼 submodule/전체 포함), 또는
- Jetson에만 `apt`/바이너리 설치하고 소스는 제외

---

## 5. 모델 파일을 GitHub에 넣고 싶다면

**권장: Git LFS**

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/MUSE_activity_model.keras
git commit -m "model: LFS로 Keras 모델 추가"
git push
```

**대안:** GitHub Releases에 `.keras` 첨부 → `DEPLOYMENT.md`에 다운로드 URL 기재.

---

## 6. 다른 사람이 clone 후 실행

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
cp .env.example .env
# models/MUSE_activity_model.keras 배치
docker compose --env-file .env up --build -d
```

상세: [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 7. 체크리스트

- [ ] `.env` 커밋 안 함
- [ ] `.env.example` 커밋함
- [ ] `MUSE_ADDRESS` 등 실제 MAC이 README/compose 기본값에 없음
- [ ] 모델: LFS·Release·수동 복사 중 하나로 문서화
- [ ] `git push` 후 GitHub에서 Actions/용량 경고 없음
