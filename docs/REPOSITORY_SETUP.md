# Thiết lập repository trên GitHub

## 1. Đẩy bộ khung lần đầu

```bash
git init
git add .
git commit -m "chore(repo): initialize project workflow"
git branch -M main
git remote add origin <repository-url>
git push -u origin main

git checkout -b dev
git push -u origin dev
```

## 2. Cấu hình owner

1. Thay `@member-1` ... `@member-6` trong `CODEOWNERS.example` bằng username thật.
2. Copy file đã sửa thành `.github/CODEOWNERS`.
3. Commit bằng `chore(repo): configure code owners`.

## 3. Branch protection đề xuất

Cho `main`:

- Require a pull request before merging.
- Require at least 1 approval.
- Require status check `validate`.
- Require conversation resolution.
- Block force pushes and deletions.

Cho `dev`:

- Require a pull request before merging.
- Require at least 1 approval.
- Require status check `validate`.
- Require conversation resolution.

## 4. Labels đề xuất

```text
workstream:data
workstream:baseline
workstream:padim
workstream:patchcore
workstream:evaluation
workstream:demo-report
type:experiment
type:bug
priority:high
status:blocked
```

## 5. Quy tắc artifact

- Dataset không đưa lên GitHub.
- Checkpoint, PaDiM statistics và PatchCore memory bank không commit trực tiếp.
- Có thể dùng GitHub Release, Git LFS hoặc storage dùng chung sau khi nhóm thống nhất giới hạn dung lượng.
- Mỗi artifact phải có experiment id, category, model, config và commit SHA tương ứng.

