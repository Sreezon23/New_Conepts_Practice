from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from advanced_customer_churn_predictor import run_analysis_pipeline

app = FastAPI(title="Customer Churn Analysis API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EXCEL_PATH = BASE_DIR / "analysis_results.xlsx"


def _load_analysis_payload(path: Path = DEFAULT_EXCEL_PATH) -> Dict[str, Any]:
    if not path.exists():
        run_analysis_pipeline(output_path=str(path))

    excel_file = pd.ExcelFile(path)
    sheet_map = {}
    for sheet_name in ["Summary", "Customers", "Risky Customers", "Clusters", "Products", "Attention Products", "Age Segments"]:
        if sheet_name in excel_file.sheet_names:
            sheet_map[sheet_name] = pd.read_excel(path, sheet_name=sheet_name)

    if "Summary" not in sheet_map:
        raise FileNotFoundError("The analysis Excel file does not contain a Summary sheet")

    summary = sheet_map["Summary"].iloc[0].to_dict()
    return {
        "summary": {
            key: (int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value)
            for key, value in summary.items()
        },
        "customers": sheet_map.get("Customers", pd.DataFrame()).to_dict(orient="records"),
        "risky_customers": sheet_map.get("Risky Customers", pd.DataFrame()).to_dict(orient="records"),
        "clusters": sheet_map.get("Clusters", pd.DataFrame()).to_dict(orient="records"),
        "products": sheet_map.get("Products", pd.DataFrame()).to_dict(orient="records"),
        "attention_products": sheet_map.get("Attention Products", pd.DataFrame()).to_dict(orient="records"),
        "age_segments": sheet_map.get("Age Segments", pd.DataFrame()).to_dict(orient="records"),
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/analysis/summary")
def get_analysis_summary() -> Dict[str, Any]:
    try:
        return _load_analysis_payload()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/upload")
def upload_excel(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Please upload an Excel or CSV file")

    upload_dir = BASE_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    save_path = upload_dir / file.filename

    with save_path.open("wb") as handle:
        handle.write(file.file.read())

    if save_path.suffix.lower() == ".csv":
        frame = pd.read_csv(save_path)
    else:
        frame = pd.read_excel(save_path)

    return {
        "message": "File uploaded",
        "filename": file.filename,
        "path": str(save_path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
