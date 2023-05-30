## 1. Instalar Bibliotecas Necessárias

Para instalar as bibliotecas necessárias, utilize o seguinte comando:

```bash
pip install -r requirements.txt
```

## 2. Utilizar o Git

### 2.1 Linkar Repositório Local com Repositório Remoto

Para linkar o repositório local com o repositório remoto, utilize o seguinte comando:

```bash
git remote add origin {link do github}
```

### 2.2 Baixar Repositório Remoto para sua Máquina

Para baixar o repositório remoto para sua máquina, utilize o seguinte comando:

```bash
git clone {link}
```

### 2.3 Baixar Atualizações do Git

Para baixar as atualizações do Git antes de começar a fazer alterações, utilize o seguinte comando:

```bash
git pull origin {nome da branch}
```

### 2.4 Incluir Arquivos na Lista de Modificados

Para incluir arquivos na lista de modificados, utilize o seguinte comando:

```bash
git add {nome do arquivo/pasta ou .}
```

### 2.5 Commit (Incluir uma Nova Versão)

Para incluir uma nova versão, utilize o seguinte comando:

```bash
git commit -m "{mensagem}"
```

### 2.6 Atualizar Repositório Remoto

1. Para atualizar o repositório remoto, utilize o seguinte comando:

```bash
git push origin {nome da branch}
```
    
2. Baixar as atualizações do git (primeira coisa a se fazer no dia antes de começar as alterações):

```bash
    git pull origin {nome da branch}
```

3. Incluir arquivos na lista de modificados:
```bash
    git add {nome do arquivo/pasta ou .} 
```

4. Commit (Incluir uma nova versão):
```bash
    git commit -m "{mensagem}"
```

5. Atualizar repositorio remoto:
```bash
    git push origin {nome da branch}
```

# 3. Acessar o Ambiente
```bash
    source AmbientePIM-env/bin/activate
```