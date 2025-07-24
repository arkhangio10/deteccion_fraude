# 🚀 Sistema de Detección de Riesgo Crediticio

Sistema avanzado de evaluación de riesgo crediticio que combina **Machine Learning tradicional** con **IA Generativa** (AWS Bedrock) para proporcionar predicciones precisas y explicaciones detalladas.

## 🎯 Características Principales

- **Doble Predicción**: Modelo ML (Random Forest) + IA Generativa (Claude 3 Haiku)
- **API REST Completa**: FastAPI con documentación automática
- **Dashboard en Tiempo Real**: Monitoreo de métricas y performance
- **Sistema de Alertas**: Detección automática de degradación del modelo
- **Logging Completo**: Registro de todas las predicciones y errores

## 📁 Estructura del Proyecto FINAL

```
credit-risk-detection/
├── 📊 data/                    # Datos de entrenamiento y prueba
│   ├── credir_risk_reto.xlsx   # Dataset original
│   ├── credit_risk_reto.csv    # Dataset procesado
│   └── credit_risk_sample_enriched.csv  # Datos enriquecidos con Bedrock
├── 📝 docs/                    # Documentación técnica
├── 📈 logs/                    # Logs del sistema y predicciones
├── 🤖 models/                  # Modelos entrenados y encoders
├── 📔 notebooks/               # Jupyter notebooks de análisis
│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio
│   ├── 02_bedrock_generation.ipynb      # Generación con Bedrock
│   └── 03_sagemaker_training.ipynb      # Entrenamiento ML
├── 🛠️ scripts/                 # Scripts de automatización (LIMPIADOS)
│   ├── restart_api.bat         # Reiniciar API con detección de puerto
│   ├── test_api.py            # Tests automáticos de la API
│   └── configure_aws_env.ps1  # Configuración AWS
└── 💻 src/                     # Código fuente principal (LIMPIADO)
    ├── api_with_monitoring.py  # API principal con monitoreo
    ├── config.py              # Configuración del sistema
    ├── data_generation.py     # Cliente AWS Bedrock
    ├── monitoring.py          # Sistema de monitoreo y métricas
    └── templates/
        └── dashboard.html      # Dashboard web de métricas
```


## 🚀 ARCHIVOS PRINCIPALES 

### 🔥 Archivos Core (NO TOCAR):
1. **`src/api_with_monitoring.py`** - API principal con dashboard
2. **`src/data_generation.py`** - Cliente Bedrock funcionando
3. **`src/monitoring.py`** - Sistema de métricas reparado
4. **`src/templates/dashboard.html`** - Dashboard web
5. **`scripts/restart_api.bat`** - Script de inicio con detección de puerto
6. **`scripts/test_api.py`** - Tests automáticos

## 🚀 Inicio Rápido

### 1. Ejecutar el Sistema
```bash
cd scripts
.\restart_api.bat
```
> ✅ **Auto-detecta puerto disponible** (8000, 8001, 8002...)

### 2. Acceder a las Interfaces
- 💻 **Dashboard**: http://localhost:8001/dashboard
- 📊 **API Docs**: http://localhost:8001/docs
- 🔍 **Métricas JSON**: http://localhost:8001/metrics

### 3. Probar Predicciones
```bash
python test_api.py
```
> ✅ **Auto-detecta puerto de la API**

## 📊 Estado del Sistema

### ✅ Funcionando Correctamente:
- API REST con FastAPI ✅
- Integración AWS Bedrock (Claude 3 Haiku) ✅
- Modelo ML Random Forest ✅
- Dashboard de métricas en tiempo real ✅
- Sistema de monitoreo y alertas ✅
- Auto-detección de puertos ✅
- Tests automáticos ✅


### 🔧 Archivos y Configuración:
- `requirements.txt` - Dependencias Python ✅
- Notebooks ejecutados y funcionando ✅

#### ⚡ Configuración de Credenciales AWS

> **Importante:** En este proyecto, las credenciales de AWS **NO** se cargaron desde un archivo `.env`, sino que se configuraron directamente usando el comando oficial de AWS CLI:

```bash
aws configure
```

Esto almacena las claves en `~/.aws/credentials` y es el método recomendado para proyectos productivos en AWS. No es necesario crear ni mantener un archivo `.env` para las claves de AWS.

## 🎉 PROYECTO LIMPIO Y LISTO

El proyecto ahora tiene una estructura limpia sin archivos duplicados:

- **4 archivos Python core** en `src/`
- **3 scripts esenciales** en `scripts/`
- **1 dashboard HTML** funcional
- **Sin archivos vacíos o duplicados**
- **Sin caché de Python**

¡Sistema profesional listo para producción! 🚀
