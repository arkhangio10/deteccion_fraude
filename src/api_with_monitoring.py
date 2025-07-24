
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import sys
import os
import time
from datetime import datetime

# Agregar el directorio src al path
sys.path.append('../src')
from data_generation import BedrockClient
from monitoring import CreditRiskMonitor

app = FastAPI(
    title="Credit Risk Detection API",
    description="API para predicción de riesgo crediticio con monitoreo en tiempo real",
    version="1.0.0"
)

# Inicializar componentes
monitor = CreditRiskMonitor()

# Cargar modelos
try:
    rf_model = joblib.load('../models/local_rf_model.pkl')
    label_encoders = joblib.load('../models/label_encoders.pkl')
    bedrock_client = BedrockClient()
    monitor.logger.info("✅ Modelos cargados exitosamente")
except Exception as e:
    monitor.log_error("MODEL_LOADING", str(e))
    rf_model = None
    label_encoders = None
    bedrock_client = None

class CreditRequest(BaseModel):
    age: int
    sex: str
    job: int
    housing: str
    saving_accounts: str = None
    checking_account: str = None
    credit_amount: float
    duration: int
    purpose: str

class PredictionResponse(BaseModel):
    ml_prediction: str
    ml_probability: float
    bedrock_prediction: str
    bedrock_confidence: float
    bedrock_reasoning: str
    recommendation: str
    response_time: float
    timestamp: str

@app.get("/")
async def root():
    return {
        "message": "Credit Risk Detection API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(request: CreditRequest):
    start_time = time.time()
    request_data = request.dict()

    try:
        if rf_model is None or bedrock_client is None:
            monitor.log_error("MODEL_UNAVAILABLE", "Modelos no disponibles", request_data)
            raise HTTPException(status_code=500, detail="Modelos no disponibles")

        # Preparar datos para ML
        data = request_data.copy()

        # Encoding de variables categóricas
        categorical_columns = ['sex', 'housing', 'saving_accounts', 'checking_account', 'purpose']
        for col in categorical_columns:
            if col in data and col in label_encoders:
                value = data[col] if data[col] is not None else 'unknown'
                try:
                    data[f'{col}_encoded'] = label_encoders[col].transform([value])[0]
                except:
                    data[f'{col}_encoded'] = 0  # Valor desconocido

        # Crear features para ML
        features = [
            data['age'], data['credit_amount'], data['duration'],
            data.get('sex_encoded', 0), data.get('job', 0), 
            data.get('housing_encoded', 0), data.get('saving_accounts_encoded', 0),
            data.get('checking_account_encoded', 0), data.get('purpose_encoded', 0)
        ]

        # Predicción ML
        ml_proba = rf_model.predict_proba([features])[0][1]
        ml_pred = "bad" if ml_proba > 0.5 else "good"

        # Predicción Bedrock
        customer_data = {
            "age": request.age,
            "sex": request.sex,
            "job": request.job,
            "housing": request.housing,
            "credit_amount": request.credit_amount,
            "duration": request.duration,
            "purpose": request.purpose
        }

        # Generar descripción con Bedrock
        description = bedrock_client.generate_credit_description(customer_data)

        # Clasificar con Bedrock
        bedrock_result = bedrock_client.classify_credit_risk(customer_data, description)

        # Recomendación final
        if ml_pred == "bad" and bedrock_result['prediction'] == "bad":
            recommendation = "RECHAZAR - Ambos modelos predicen alto riesgo"
        elif ml_pred == "good" and bedrock_result['prediction'] == "good":
            recommendation = "APROBAR - Ambos modelos predicen bajo riesgo"
        else:
            recommendation = "REVISAR MANUALMENTE - Modelos discrepan"

        # Calcular tiempo de respuesta
        response_time = time.time() - start_time

        # Registrar en monitoreo
        monitor.log_prediction(
            request_data, ml_pred, ml_proba,
            bedrock_result['prediction'], bedrock_result['confidence'],
            response_time
        )

        return PredictionResponse(
            ml_prediction=ml_pred,
            ml_probability=float(ml_proba),
            bedrock_prediction=bedrock_result['prediction'],
            bedrock_confidence=bedrock_result['confidence'],
            bedrock_reasoning=bedrock_result['reasoning'],
            recommendation=recommendation,
            response_time=response_time,
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        response_time = time.time() - start_time
        monitor.log_error("PREDICTION_ERROR", str(e), request_data)
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "ml_model": rf_model is not None,
            "bedrock_client": bedrock_client is not None
        },
        "system_status": monitor._get_system_status()
    }

@app.get("/metrics")
async def get_metrics():
    """Endpoint para obtener métricas del sistema en formato JSON"""
    return monitor.get_metrics_summary()

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Página web con dashboard de métricas"""
    try:
        with open("templates/dashboard.html", "r") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        # Fallback - dashboard HTML embebido
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Credit Risk API - Dashboard de Métricas</title>
            <meta charset="UTF-8">
            <meta http-equiv="refresh" content="10">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                h1 { color: #333; }
                .container { max-width: 800px; margin: 0 auto; }
                .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                pre { background: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Dashboard de Métricas - Credit Risk API</h1>
                <div class="card">
                    <h2>Datos en tiempo real</h2>
                    <p>Los datos se actualizan cada 10 segundos</p>
                    <pre id="metrics-data">Cargando datos...</pre>
                </div>
            </div>
            <script>
                function loadMetrics() {
                    fetch('/metrics')
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById('metrics-data').textContent = JSON.stringify(data, null, 2);
                        })
                        .catch(error => {
                            console.error('Error cargando métricas:', error);
                            document.getElementById('metrics-data').textContent = 'Error cargando métricas: ' + error.message;
                        });
                }
                
                // Cargar métricas al iniciar
                window.onload = function() {
                    loadMetrics();
                    // Auto-refrescar cada 10 segundos
                    setInterval(loadMetrics, 10000);
                };
            </script>
        </body>
        </html>
        """
        return html_content

if __name__ == "__main__":
    import uvicorn
    import socket
    import sys
    
    # Puerto predeterminado
    port = 8000
    
    # Verificar si el puerto está disponible, si no probar con puertos alternativos
    for attempt_port in range(8000, 8010):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('0.0.0.0', attempt_port))
            sock.close()
            port = attempt_port
            break
        except OSError:
            monitor.logger.warning(f"Puerto {attempt_port} no disponible, probando otro...")
    
    monitor.logger.info(f"🚀 Iniciando Credit Risk API con monitoreo en puerto {port}")
    print(f"\n💻 Dashboard disponible en: http://localhost:{port}/dashboard")
    print(f"🔍 API docs disponible en: http://localhost:{port}/docs")
    print(f"📊 Métricas JSON en: http://localhost:{port}/metrics\n")
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        monitor.logger.error(f"Error al iniciar servidor: {e}")
        print(f"❌ Error al iniciar servidor: {e}")
        sys.exit(1)
