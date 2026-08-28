# Quy tắc đóng góp

## Branch

- Không commit trực tiếp vào `main` hoặc `develop`.
- Tên branch: `feature/p<so-nguoi>-<mo-ta>`, `fix/<mo-ta>`, `docs/<mo-ta>`.
- Rebase/merge `develop` mới nhất trước khi yêu cầu review.

## Commit

Dùng Conventional Commits:

```text
feat(data): add deterministic MVTec split manifest
feat(patchcore): add coreset subsampling
fix(eval): prevent test threshold tuning
docs(protocol): lock validation rules
test(data): detect split overlap
```

Không commit dataset, secret, checkpoint lớn, memory bank lớn hoặc output có thể tái tạo lại.

## Pull request

PR phải:

- liên kết issue;
- mô tả thay đổi và cách chạy kiểm tra;
- có config/seed cho experiment;
- đính kèm metric hoặc output mẫu nếu thay đổi model;
- xác nhận không data leakage;
- có ít nhất một reviewer ngoài tác giả;
- vượt qua CI trước khi merge.

## Quyền merge

- Module owner phê duyệt nội dung chuyên môn.
- Cross-reviewer kiểm tra code/kết quả.
- Leader kiểm tra dependency, protocol và quyết định merge vào `develop`.
- `main` chỉ nhận release đã chạy end-to-end.

