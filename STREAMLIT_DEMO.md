# Streamlit Demo

這個展示程式會把競賽 artifacts 當成唯讀輸入：

- `01_best.pt`
- `08_best.pt`
- `v3_01_WBF_outputcsv.ipynb`
- `v5_08_NMS_outputcsv.ipynb`
- `sample_submission.csv`
- `images/`

程式不會覆寫、搬移、改名或修改以上檔案。

## Setup

安裝展示程式依賴：

```powershell
pip install -r requirements_streamlit.txt
```

把 YOLOv7 程式碼放在 `streamlit_app.py` 同層的 `yolov7/`：

```powershell
git clone https://github.com/ws6125/yolov7.git
```

如果目前環境沒有 `git`，可以從 GitHub 下載 ZIP，解壓縮後把資料夾改名成 `yolov7`。

啟動展示頁：

```powershell
streamlit run streamlit_app.py
```

## Notes

- Implementation 01 使用 `01_best.pt` 搭配 WBF post-processing。
- Implementation 08 使用 `08_best.pt` 搭配 NMS post-processing。
- 程式會在記憶體內注入 Implementation 08 需要的 CBAM class，不會編輯 YOLOv7 檔案。
- 下載的標註圖片與單列 CSV preview 都是展示程式產生的新輸出。
