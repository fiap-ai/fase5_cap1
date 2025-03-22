# Documentação do Script `merge_notebooks.py`

## Visão Geral

O script `merge_notebooks.py` é uma ferramenta para mesclar múltiplos notebooks Jupyter em um único arquivo. Isso é útil para combinar notebooks de diferentes fases de um projeto em um notebook final consolidado.

## Funcionalidades

- Mescla múltiplos notebooks Jupyter (.ipynb) em um único arquivo
- Mantém a formatação e estrutura original de cada notebook
- Adiciona células de markdown como separadores entre os notebooks
- Preserva todas as células (código, markdown, saídas) dos notebooks originais

## Requisitos

- Python 3.6 ou superior
- Biblioteca `nbformat` (geralmente instalada com o Jupyter)

## Como Funciona

O script funciona da seguinte maneira:

1. Lê cada notebook de entrada na ordem em que são fornecidos como argumentos
2. Usa o primeiro notebook como base para o notebook mesclado
3. Para cada notebook subsequente, adiciona uma célula de markdown como separador
4. Adiciona todas as células do notebook subsequente ao notebook mesclado
5. Salva o notebook mesclado no arquivo de saída especificado

## Uso

```bash
python merge_notebooks.py <arquivo_saida.ipynb> <arquivo_entrada1.ipynb> <arquivo_entrada2.ipynb> ...
```

### Parâmetros

- `<arquivo_saida.ipynb>`: Nome do arquivo de notebook de saída (será criado ou sobrescrito)
- `<arquivo_entrada1.ipynb> <arquivo_entrada2.ipynb> ...`: Lista de arquivos de notebook de entrada a serem mesclados

### Exemplo

```bash
python merge_notebooks.py notebooks/notebook_final.ipynb notebooks/fase1_preparo_ambiente.ipynb notebooks/fase2_analise_exploratoria.ipynb notebooks/fase3_clusterizacao.ipynb notebooks/fase4_modelagem_preditiva.ipynb
```

Este comando mesclará os notebooks das fases 1 a 4 em um único arquivo chamado `notebook_final.ipynb`.

## Importante: Ordem dos Notebooks

O script mescla os notebooks **na ordem exata em que são fornecidos na linha de comando**. Não há lógica interna para determinar a ordem correta dos notebooks.

Para garantir que o conteúdo seja mesclado na sequência lógica do projeto, é essencial listar os notebooks na ordem desejada. Por exemplo, para um projeto com fases sequenciais, a ordem correta seria:

1. fase1_preparo_ambiente.ipynb
2. fase2_analise_exploratoria.ipynb
3. fase3_clusterizacao.ipynb
4. fase4_modelagem_preditiva.ipynb

## Saída do Script

Durante a execução, o script exibe informações sobre o processo de mesclagem:

```
Mesclando 4 notebooks...
Processando 1/4: notebooks/fase1_preparo_ambiente.ipynb
  Usando notebooks/fase1_preparo_ambiente.ipynb como notebook base
Processando 2/4: notebooks/fase2_analise_exploratoria.ipynb
  Adicionando 24 células de notebooks/fase2_analise_exploratoria.ipynb
Processando 3/4: notebooks/fase3_clusterizacao.ipynb
  Adicionando 28 células de notebooks/fase3_clusterizacao.ipynb
Processando 4/4: notebooks/fase4_modelagem_preditiva.ipynb
  Adicionando 16 células de notebooks/fase4_modelagem_preditiva.ipynb

Notebooks mesclados com sucesso em: notebook_final.ipynb
Total de células no notebook final: 83
```

## Limitações

- O script não verifica a compatibilidade entre os notebooks (por exemplo, dependências de variáveis)
- Não há opção para excluir células específicas durante a mesclagem
- Não há validação do conteúdo dos notebooks (é responsabilidade do usuário garantir que os notebooks sejam válidos)

## Dicas de Uso

1. **Verifique os notebooks antes da mesclagem**: Certifique-se de que cada notebook esteja funcionando corretamente antes de mesclá-los.

2. **Renomeie variáveis conflitantes**: Se houver variáveis com o mesmo nome em diferentes notebooks, considere renomeá-las para evitar conflitos.

3. **Adicione células de transição**: Após a mesclagem, pode ser útil adicionar células de markdown adicionais para melhorar a transição entre os notebooks.

4. **Execute o notebook mesclado**: Após a mesclagem, execute todas as células do notebook mesclado para garantir que tudo funcione corretamente.

5. **Faça backup dos notebooks originais**: Sempre mantenha cópias dos notebooks originais, caso precise fazer alterações.
