#!/bin/bash
# Script para configurar um ambiente virtual Python para o projeto

# Criar um ambiente virtual
echo "Criando ambiente virtual..."
python -m venv venv

# Ativar o ambiente virtual
echo "Ativando ambiente virtual..."
source venv/bin/activate

# Instalar as dependências necessárias
echo "Instalando dependências..."
pip install pandas numpy matplotlib seaborn scikit-learn jupyter nbformat

# Verificar instalação
echo "Verificando instalação..."
python -c "import pandas; import numpy; import matplotlib; import seaborn; import sklearn; import jupyter; import nbformat; print('Todas as bibliotecas foram instaladas com sucesso!')"

echo ""
echo "Ambiente configurado com sucesso!"
echo "Para ativar o ambiente virtual, execute: source venv/bin/activate"
echo "Para iniciar o Jupyter Notebook, execute: jupyter notebook"
