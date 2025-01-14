# Comparativo de Algoritmos de Inteligência Artificial

## Aplicação no problema das Junções de emenda (uniões) das Sequências de Genes (DNA) de Primatas

Junções de emenda (uniões) são pontos em uma sequência de DNA nos quais o DNA **supérfluo** é removido durante o processo de criação de proteínas em organismos superiores. O problema proposto neste conjunto de dados é **reconhecer, dada uma sequência de DNA, os limites entre exons (as partes da sequência de DNA retidas após o emendamento, ou junção) e íntrons (as partes da sequência de DNA que são emendadas)**. Este problema consiste em duas subtarefas: reconhecer limites de exon/íntron (chamados de sítios EI) e reconhecer limites de íntron/éxon (sítios IE). (Na comunidade biológica, os limites de IE são chamados de **acceptors**, enquanto os limites de EI são chamados de **donors**.)

Este conjunto de dados **(que vamos utilizar nessas availiações)** foi desenvolvido para ajudar a avaliar um algoritmo de aprendizado "híbrido" (KBANN) que usa exemplos para refinar indutivamente o conhecimento preexistente. Usando uma metodologia de "validação cruzada" de dez vezes em 1000 exemplos selecionados aleatoriamente do conjunto completo de 3190, as seguintes taxas de erro foram produzidas por vários algoritmos de ML - Machine Learning (todos os experimentos foram realizados na Universidade de Wisconsin, às vezes com implementações locais de algoritmos publicados).

| System |Neither|  EI | IE  |
|:-----:|------:|-----:|-----:|
| **KBANN**  |  **4.62** |  **7.56** |  **8.47** |
|BACKPROP| 5.29 |  5.74| 10.75|
|PEBLS | 6.86| 8.18| 7.55|
|PERCEPTRON| 3.99| 16.32| 17.41|
|ID3 |  8.84| 10.58| 13.99|
|COBWEB| 11.80| 15.04| 9.46|
|Near. Neighbor| 31.11| 11.65| 9.09|


>[!Important]
>Dos algoritmos acima, somente iremos utilizar nas comparações o **KBANN (Knowledge-Based Artificial Neural Networks)**, demais algoritmos da comparação estarão descritos na sequência deste documento.

Fonte: https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences, último acesso em 01/01/2025

### Descrição da Teoria Subjacente

A ***Teoria de Domínio Sequencial***, também conhecida como ***Teoria da Domínio do Gene***, é um conceito na biologia molecular e genética que se refere a como as proteínas e outros produtos gênicos são organizados e expressos de maneira funcional em organismos. Essa teoria é particularmente relevante para a compreensão das sequências de aminoácidos que constituem as proteínas e a maneira como essas sequências se relacionam com a função biológica.

A teoria postula que proteinas são compostas de várias sequências ou ***domínios funcionais***, que podem ser consideradas como "blocos de construção" dos quais as proteínas são montadas. Cada domínio pode ter uma função específica e, muitas vezes, pode ser encontrado em diversas proteínas em diferentes contextos. 

Os domínios são regiões da sequência polipeptídica que apresentam aspectos estruturais e funcionais característicos, muitas vezes funcionando independentemente de outras partes da proteína. Essas regiões também podem se combinar de diferentes formas para executar diferentes tarefas.

Pode-se dizer que a proteína é um componente modular, que pode ser agrupado e classificado de acordo com os domínios que possui. Muitos desses domínios não apenas formam estruturas estáveis dobradas em solução, mas frequentemente retêm parte da função bioquímica da proteína maior da qual são derivados. 

### Importância dos Domínios Sequenciais

**Função Proteica**: Os domínios proteicos são responsáveis por diferentes funções biológicas. Por exemplo, um domínio pode ser responsável pela ligação ao DNA, enquanto outro pode participar de interações com outras proteínas.

**Evolução Proteica**: Os domínios podem ser reutilizados em várias proteínas, o que explica a diversidade funcional das proteínas em um organismo. Isso também implica que mudanças em domínios específicos podem levar a novas funções sem a necessidade de recriar a proteína a partir do zero.

**Predição de Função**: Conhecer a estrutura e as sequências de domínios pode ajudar os cientistas a prever as funções de proteínas desconhecidas baseadas em informações de sequências conhecidas.

**Exemplo Prático**

Um exemplo de domínio sequencial pode ser visto nos fatores de transcrição, que muitas vezes possuem um domínio de ligação ao DNA e um domínio de ativação. Esses fatores podem interagir com sequências específicas de DNA, iniciando a transcrição de genes alvos.

**Exemplo de Domínios em Proteínas**

O gene da **β-globina** em humanos é um caso famoso que ilustra a teoria de domínio sequencial. Ele codifica uma proteína que parte do grupo das globinas (proteínas transportadoras de oxigênio) e contém domínios que se ligam à hemoglobina e que facilitam a ligação ao oxigênio.

**Ferramentas e Recursos Práticos**

**BLAST (Basic Local Alignment Search Tool)**: Usada para encontrar regiões semelhantes entre sequências, ajudando a identificar domínios conservados.

**Pfam**: Um banco de dados de domínios proteicos que fornece informações sobre a função e estrutura de domínios sequenciais. (https://pfam.xfam.org/)

**Fontes Teóricas e Referências**

**Livros**: 
        "Molecular Biology of the Cell" por Alberts et al. Explica a estrutura e função das proteínas, incluindo como domínios sequenciais funcionam.
        "Biochemistry" por Berg, Tymoczko e Stryer. Este livro aborda as funções das proteínas em detalhe.

**Artigos de Pesquisa**: 
        "The evolution of protein domains" - Este artigo analisa como os domínios proteicos evoluíram e o seu impacto na funcionalidade das proteínas.
        "Protein Domain Organization: Insights from the Public Domain" - Explora a organização e a função dos domínios nas proteínas.

**Recursos Online**:

  - O site do NCBI oferece acesso a uma variedade de artigos de pesquisa e revisões sobre domínios sequenciais. (https://www.ncbi.nlm.nih.gov)
  - UniProt: Um recurso extensivo para informações de proteínas que inclui dados sobre domínios funcionais (https://www.uniprot.org/).
  - Domínio proteico: Wikepedia, acesso em 30/12/2024 (https://pt.wikipedia.org/wiki/Dom%C3%ADnio_proteico)
        

**Conclusão sobre a Teoria de Domínio Sequencial**

A Teoria de Domínio Sequencial é fundamental para a biologia e genética modernas, elucidando como as proteínas são organizadas e evoluem ao longo do tempo. Compreender essa teoria ajuda os pesquisadores em muitos campos científicos, incluindo biologia celular, bioquímica e biotecnologia. 



## Desenvolvimento do Trabalho

Comparativo de algoritmos aplicados no problema das ***Junções de emenda (uniões) das Sequências de Genes (DNA) de Primatas*** é um exercício teórico/prático relevante para nossa compreensão do funcionamento dos algoritmos e dos resultados que emergem de sua aplicação no problema descrito no contexto.

Os objetivos principais neste trabalho é apresentar perspectivas de aplicação de ***diferentes*** algoritmos de Inteligência Artificial, no problema acima descrito, o qual caracterizado como um problema de **classificação**. Mas não somente a utilização dos algoritmos, como também a escolha e justificativas dos Hiperparâmetros escolhidos, os quais sabidamente afetam os resultados obtidos, e que podem inviabilizar as comparações entre eles. Deste modo elencamos os seguintes algoritmos para tentar resolver o problema de classificação das sequências de DNA:
  
  - [Rede Neural KBANN (Knowledge-Based Artificial Neural Networks)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.inf.ufrgs.br/~engel/data/media/file/cmp121/kbann-artigo.pdf&ved=2ahUKEwiFn9yCqdWKAxX4BrkGHWFMEq8QFnoECBUQAQ&usg=AOvVaw2yzjp752bnqkN2fu16XcdW)
  - [RandomForestClassifier](https://link.springer.com/article/10.1023/a:1010933404324)
  - [MLP Classifier - Multi Layer Perceptron](https://www.semanticscholar.org/paper/Learning-representations-by-back-propagating-errors-Rumelhart-Hinton/052b1d8ce63b07fec3de9dbb583772d860b7c769)

Importante lembrar que as teorias aqui apresentadas, especificamente sobre o sequenciamento de DNA e suas interpretações nas questões do Domínio Sequencial, não estão num formalismo acadêmico, mas foram adicionadas para contextualizar o problema que foi objeto para a aplicação dos algoritmos, e ajustes dos seus respectivos hiperparâmetros. Entretanto colocamos os links para os temas abordadas, caso haja interesse para melhor entendimento e esclarecimentos.

Os ajustes dos hiperparâmetros foram realizados e discutidos nos códigos em Python, dentro das funções que executam cada algoritmo individual, e seus resultados foram apresentados de forma gráfica e também discutidos e apresentados no item Resultados abaixo.

A comparação entre algoritmos foi realizada não levando-se em consideração o tempo de processamento, mas sim os resultados práticos e qualitativos como a **acurácia** gerada por cada algoritmos, assim como as **taxas de erros percentuais** geradas para cada classe avaliada, no caso as classe **EI, IE e N**.

## Resultados

**Sobre a Rede Neural KBANN**

Este algoritmo combina redes neurais com conhecimento pré-existente. Na prática, não tendo uma implementação padrão do KBANN nas bibliotecas mais comuns de Python como TensorFlow ou PyTorch. A implementação geralmente envolve uma estrutura manual para utilizar conhecimento aprendido e previamente estabelecido na arquitetura da rede.

Mais detalhes e características dessa Rede Neural, podem ser obtidas no documento adicional elaborado para o presente trabalho, disponível na pasta [docs/Algoritmos_IA_Descricoes.pdf](https://github.com/luizantoniopaula/comp_algoritmos/blob/main/docs/Algoritmos_IA_Descricoes.pdf)

Ao utilizar a rede neural KBANN, foram necessários inúmeros ajustes dos hiperparâmetros para que conseguissemos nos aproximar dos valores apresentados na base de dados compartilhados para uso em pesquisas com IA/Machine Learning, da [Universidade de Irvine/CA.](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences) e também descritos no inicio deste texto.

Os seguintes hiperparâmetros foram diversas vezes alterados de forma sistêmica, mantendo-se valores pré-definidos, e então procedendo alterações dos valores para maior ou menor, sempre acompanhando a acurácia e os erros percentuais por classe analisada:

| Hiperparâmetro | Valor Final | Descrição |
|:--------------:|:-----------:|:---------:|
| Dense(64...)   | 64  e 32        | Define o tamanho da camada da rede neural,oculta para 64 e 32 perceptrons |
| activation     | relu        | Função de ativação utilizada nas camadas ocultas, no caso foi a ReLU (Rectified Linear Unit), não foi alterada |
| Dropout        | 0.5 | Desativa aleatóriamente 50% dos neurônidos da camada anterior, no treinamento ajuda prevenir overfitting|individuais, não alterado.|
|activation na saída    | softmax     | Função de ativação na camada de saída, 'softmax' transforma as saídas em probabilidades que somam a 1, para classificação multi-classe|
| Epoch          | 50   | Número de iterações completas através do conjunto de dados de treinamento. Cada época permite que o modelo aprenda mais sobre os dados. |
| Batch          | 8    | Número de amostras usadas antes de atualizar os pesos no modelo. Um tamanho de lote menor evita sobrecarga da memória, mas pode tornar o treinamento mais ruidoso.
| fold           | 20   | Indica que, o conjunto de dados é dividido em 20 partes iguais. Cada **(fold)** será usada como conjunto de validação uma vez,o restante dos dados é usado para treinamento.|

![image](https://github.com/user-attachments/assets/3597cfe3-c2cc-4c7b-a75e-a7d556c0801b)

Nos resultados obtidos, podemos perceber que não adiantou criar uma estrutura de redes neurais muito grande, pois a variação nos erros percentuais nas séries investigadas foi muito pequena. Se compararmos uma rede com 256 x 128 x 64 x 32, tendo 4 camadas ocultas, gerou uma taxa de erros de 10.33% na série EI, contra o melhor resultado de 8,11% na mesma série, agora obtido com uma rede de 64 x 32 neurônios nas camadas internas.

A mudança desses valores foi seguida por ajustes nos tamanhos das partições das validações cruzadas (folds), que foram de **10 fold** no primeiro caso, subindo até **20 fold** no melhor caso. Também impactante verificar que o número dos parâmetros Epoch e Batch foram reduzidos drasticamente, de **300 epoch** e **192 batch**, para **50 epoch** e **8 batch**.

O número de iterações completas através do conjunto de dados de treinamento, ou o parâmetro **Epoch**, permite que o modelo aprenda mais sobre os dados. Porém o grande valor inicial estava superestimado, pois verificou-se que sua redução foi mais impactante na redução dos erros percentuais por séries, da mesma forma que ocorreu com o número das redes neurais ocultas

Juntamente com o parâmetro **Epoch**, o parâmetro **Batch** também veio sendo reduzido em conjunto, tendo seu resultado final em **8 Batch**. Esse parâmetro é o número de amostras usadas antes de atualizar os pesos no modelo. Um tamanho de lote menor evita sobrecarga da memória, mas pode tornar o treinamento mais ruidoso. Entretanto, nos resultados obtidos, foi justamente a redução neste parâmetro que também contribuiu para redução nos erros percentuais nas séries. 

Os parâmetros que ficaram fixos, na verdade foram objeto de testes, onde também alteramos seus valores, como o **Dropout de 0.5 para 0.3 e 0.7**, mas não houveram mudanças significativas nos resultados.
Deste modo observamos que os hiperparâmetros ajustados foram os mais relevantes para a mudança nos resultados, reduzindo os erros %, mas manteve-se em todos os casos, a acurácia em 0.96.

Concluindo, podemos observar que, os melhores resultados nos erros % das série **EI, IE e NEITHER** com esta rede KBANN, respectivamente **8,11%, 10,34% e 5,2%**, ainda estão longe dos resultados apresentados no site que hospeda a base de dados, os quais foram **7.56%, 8.47%, 4.62%**, respectivamente para as suas séries EI, IE e NEITHER.  
Porem na Universidade de Irvine, usaram a metodologia de "validação cruzada de dez vezes" em 1000 exemplos selecionados aleatoriamente do conjunto completo de 3190. E nossa abordagem, utilizamos as amostras em validações cruzadas (folds) que variaram de 5, 10 e 20, porém em toda a base de dados e seu conjunto de 3190 registros.

-----
**Sobre a Árvore de Decisão Random Forest**

Este algoritmo é uma **Árvore de Decisão Aleatória** e consiste de múltiplas árvores de decisão (**Forest**). Cada árvore é treinada em um subconjunto diferente dos dados e suas previsões são combinadas (geralmente pela média ou votação majoritária) para produzir o resultado final.

Os seguintes hiperparâmetros foram alterados de forma sistêmica, mantendo-se valores pré-definidos, e então procedendo alterações dos valores para maior ou menor, sempre acompanhando a acurácia e os erros percentuais por classe analisada:

| Hiperparâmetro | Valor Final | Descrição |
|:--------------:|:-----------:|:---------:|
| n_estimators   | 400         |  Especifica o número de árvores na floresta. Um número maior de árvores geralmente melhora o desempenho, pois reduz o sobreajuste e varia da estimativa final. Porém, um número maior aumenta o tempo de treinamento e requisitos de memória |
| n_splits       | 20   | Indica que, o conjunto de dados é dividido em 20 partes iguais. Cada **(fold)** será usada como conjunto de validação uma vez,o restante dos dados é usado para treinamento.|

![image](https://github.com/user-attachments/assets/b6ff13fc-0970-4ddc-8cca-95dec6222886)


Nos resultados acimpa percebemos que o aumento no número de árvores aleatórias (**n_estimators**) não melhorou o resultado, pois a variação nos erros percentuais nas séries investigadas foi muito pequena. Se compararmos **400 n_estimators e 20 n_splits**, com **800 n_estimators e 20 n_splits**, veremos que ao dobrar o número de árvores, não tivemos variação significativa no sentido da diminuição das taxas de erros nas sequências em análise. Mantendo-se os números de n_splits constantes, vemos que ao reduzir o número de árvores, os valores de erros foram reduzindo. (**melhorando**)

Porém, observando-se a planilha como um todo, temos um aumento no número de árvores (**100 a 400**) e mantendo-se os **n_splits constantes (20)**, os valores foram caindo. A partir desse ponto, com 400 árvores e 20 conjunto de dados para treinamento e validação cruzada, os valores dos erros % por série, foram aumentados. Em contrapartida, os resultados foram piorando, ficando evidente que encontramos um valor médio nos parâmetros de ajuste (hiperparâmetros), onde obtivemos os melhores índices de erros e acurácia média do modelo.

Concluindo, podemos observar que os melhores resultados nos **erros % das série EI, IE e NEITHER** com o **Random Forest**, foram respectivamente **7,09%, 9,58% e 4,36%**. Estes resultados **são melhores**, em parte, que os apresentados no site que hospeda a base de dados, os quais foram **7.56%, 8.47%, 4.62%**, respectivamente para as suas séries EI, IE e NEITHER.

Novamente lembramos que na Universidade de Irvine, usaram a metodologia de "validação cruzada de dez vezes" em 1000 exemplos selecionados aleatoriamente do conjunto completo de 3190. E nossa abordagem, utilizamos as amostras em validações cruzadas (folds) que variaram de **10 a 30**, porém em toda a base de dados e seu conjunto de 3190 registros.

Mesmo tendo **reduzido** a taxa de erros nas séries **EI para 7,09% e NEITHER para 4,36%, na série IE com 9,58%** a **acurácia** não foi reduzida com uso do **Random Forest**. Porém, além do fato da redução de hiperparâmetros para serem configurados e analisados, e também do tempo de execução em máquina, este algoritmo se mostra superior ao KBANN.

Em resumo, mais rápido, mais fácil de configuração e com melhores resultados, ao menos neste modelo de dados em análise.

----

**Sobre a Rede Neural MLP - Multi Layer Perceptron**

Este algoritmo é um tipo de rede neural feedforward composta por pelo menos três camadas de nós: uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída. Cada nó (neurônio) em uma camada usa uma função de ativação não linear, exceto pelos nós de entrada. MLPs são capazes de aprender representações não lineares complexas, tornando-os adequados para tarefas de classificação e regressão.

Os seguintes hiperparâmetros foram alterados de forma sistêmica, mantendo-se valores pré-definidos, e então procedendo alterações dos valores para maior ou menor, sempre acompanhando a acurácia e os erros percentuais por classe analisada:

| Hiperparâmetro | Valor Final | Descrição |
|:--------------:|:-----------:|:---------:|
|hidden_layer_sizes| 128 x 64  | Define o tamanho da camada da rede neural,oculta para 128 e 64 perceptrons |
| max_iter   | 200   | Número de iterações completas através do conjunto de dados de treinamento. Cada época permite que o modelo aprenda mais sobre os dados. |
| n_splits           | 20   | Indica que, o conjunto de dados é dividido em 20 partes iguais. Cada **(split)** será usada como conjunto de validação uma vez,o restante dos dados é usado para treinamento.|

![image](https://github.com/user-attachments/assets/758ce7e1-3384-484c-b29b-ee682951011b)

Os resultados obtidos com esse algoritmo MLP foram os piores, em relação aos demais modelos. Notadamente a parametrização (hiperparâmetros) para este algoritmo é bem limitada, restringindo aos valores das camadas ocultas (**hidden_layer_sizes**), o número de iterações (**max_iter**) e as o número de conjuntos de dados para validação cruzada (**n_splits**). Não temos mais parâmetros como as funções de ativação, parâmetros Batch, funções perda (loss), Dropout, dentre outros parâmetros.

Assim percebemos que, de fato é um algoritmo um pouco limitado, mas não houve tempo para procurar mais detalhes e talvez fazer ou utilizar versões mais aprimoradas do mesmo, entretanto para efeitos de comparação com os demais, podemos creditar a limitação de hiperparâmetros como um fator de diminui as possibilidades de acurácia deste algoritmo, em contraponto, por exemplo, com as redes KBANN.

Os melhores resultados obtidos nesta rede neural foram: **10,96%, 12,22%, 6,50%**, respectivamente para as suas séries **EI, IE e NEITHER**, tendo-se uma acurácia média de **0,95**, ou seja, a menor em todos os algoritmos aqui testados.

A planilha acima apresenta mais resultados deste algoritmo e facilita a visualização das opções de modificação nos hiperparâmetros e fos resultados alcançados, onde podemos ver que as evoluções foram muito pequenas, pelas variações de parâmetros.

----
### Conclusão

Enfim, de todos os algoritmos testados, pelos resultados e simplicidade de parametrização, além do tempo de execução, o **Random Forest** apresentou resultados levementes superiores à rede neural **KBANN**. Entretanto, esta última é de complexidade mais elevada de parametrização e exige mais tempo para testes e treinamento dos parâmetros. Devido ao tempo de execução, pode se tornar uma tarefa bem exaustiva e difícil implementação, uma vez que a mudança de vários parâmetros pode produzir resultados inesperados, se não for realizada de maneira sistêmica e com acompanhamento e verificação dos resultados.

É possível perceber que, a parametrização de algoritmos torna-se uma especialidade necessária, na medida que o emprego/utilização dessas ferramentas, de forma mais corriqueira nas empresas, vai impondo a necessidade desse conhecimento, e como consequência temos a necessidade de atualização quase constante no conhecimento dos algoritmos, seus hiperparâmetros e características fundamentais. Pois não devemos esquecer que, o tipo de problema para o qual devemos aplicar os algoritmos, são de extrema importância para que os mesmos consigam apresentar resultados satisfatórios e condizentes com seus objetivos primordiais.

Em síntese, tais algorimtos devem ser empregados corretamente nas tarefas correspondentes, como Classificação, Agrupamento ou Regressão, ou seja, sempre há um algoritmo mais adequado para cada problema a ser resolvido ou respondido.

Esta atividade cada dia que passa, pode se tornar uma arte!

----
### Execução do programa Proc_compAlg.py

Para executar o código comparativo, basta baixar o arquivo do programa em linguagem Python, disponível neste repositório, com o nome **Proc_compAlg.py**.
Uma vez baixado, abrir o arquivo com alguma IDE para Python, como o **Pycharm**, o **VSCode** ou qualquer outra de sua escolha. Verificar que existem vários imports de bibliotecas, no início do arquivo do programa.

Convém lembrar que, cada linha de import deve ser instalada via comando **"pip install nome_bibblioteca"**. Se por ventura seu ambiente já tiver a biblioteca, o instalador somente relata que há versão instalada, caso não tenha ele a bilbioteca será instalada em seu ambiente.

Uma vez instaladas a bilbiotecas, vá para o final o arquivo ondem existem as chamadas das funções que executam, individualmente, cada algoritmo descrito neste trabalho. As linhas das chamadas das funções são as seguintes:

**Exemplos de uso**:
- #**kbann_exe()**  #  Caso deseje executar a rede neural KBANN
- #**random_forest()**  #  Caso deseje executar o classificador RandoForest
- #svm_exec(X, y)  # Caso deseje executar a rede SVM Suport Vector Machine --> Não executar, pois não está funcionando ainda...
- #**mlp_exec()**  # Caso deseje executar a rede MLP - Multilayer Perceptron

Basta então escolher o algoritmos e **remover o caracter "#"** que comenta a linha, assim basta rodar o programa. Em suas execuções, irão  ser impressos os resultados de cada passagem nos Folds de testes, com os resultados individuais por Fold. Ao final será printado no terminal os resultados e os gráficos da matriz de confusão e dos percentuais de acurácia e erros% por série, este último na forma de um gráfico de barras.

Peço a gentileza se caso não executar, reportar ao meu email para que eu possa corrigir e melhorar detalhes, se for possível.

---
## Sobre o Trabalho

O presente trabalho foi desenvolvido como exercício final para a disciplina de **CA006NC - Inteligência Artificial**, do curso de Mestrado em Inteligência Artificial, dentro do PPGI - Programa de Pós-Graduação em Informática, da UTFPR Cornélio Procópio/PR

Elaboração: **Luiz Antonio Silva de Paula**
[email](mailto:luiz.paula@alunos.utfpr.edu.br)
