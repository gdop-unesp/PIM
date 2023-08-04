# Preparação do Ambiente

Para executar o programa, é necessário garantir que todas as bibliotecas do ambiente virtual estejam atualizadas. Caso o ambiente virtual não tenha sido criado, basta executar o comando abaixo.

```bash
python -m venv venv
```
Uma vez que o ambiente foi criado, as bibliotecas podem ser carregadas utilizando os comandos abaixo.

```bash
source venv/bin/activate  # Inicia o ambiente virtual

pip install -r requirements.txt  # Carrega as bibliotecas
```

# Versionamento do Código (Git)

Para o versionamento do código, foi utilizado o software [Git](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F). As principais funções oferecidas por ele são de:

1. Baixar as atualizações mais recentes do repositório remoto
```bash
git pull origin main
```
2. Criar uma nova versão local
```bash
# Inclui arquivo ou diretório para nova versão
git add <nome_do_arquivo/pasta>

# Cria nova versão com arquivos incluidos anteriormente
git commit -m "Comentário da nova versão"
```

3. Subir as versões locais para o repositório remoto
```bash
git push origin main
```
