# Comparativo de Algoritmos de Inteligência Artificial

## Aplicação no problema das Junções de emenda (uniões) das Sequências de Genes (DNA) de Primatas

Junções de emenda (uniões) são pontos em uma sequência de DNA nos quais o DNA **supérfluo** é removido durante o processo de criação de proteínas em organismos superiores. O problema proposto neste conjunto de dados é **reconhecer, dada uma sequência de DNA, os limites entre exons (as partes da sequência de DNA retidas após o emendamento, ou junção) e íntrons (as partes da sequência de DNA que são emendadas)**. Este problema consiste em duas subtarefas: reconhecer limites de exon/íntron (chamados de sítios EI) e reconhecer limites de íntron/éxon (sítios IE). (Na comunidade biológica, os limites de IE são chamados de **acceptors**, enquanto os limites de EI são chamados de **donors**.)

Este conjunto de dados **(que vamos utilizar nessas availiações)** foi desenvolvido para ajudar a avaliar um algoritmo de aprendizado "híbrido" (KBANN) que usa exemplos para refinar indutivamente o conhecimento preexistente. Usando uma metodologia de "validação cruzada" de dez vezes em 1000 exemplos selecionados aleatoriamente do conjunto completo de 3190, as seguintes taxas de erro foram produzidas por vários algoritmos de ML - Machine Learning (todos os experimentos foram realizados na Universidade de Wisconsin, às vezes com implementações locais de algoritmos publicados).

| System |Neither|  EI | IE  |
|:-----:|------:|-----:|-----:|
|KBANN  |  4.62 |  7.56|  8.47|
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

Comparativo de algoritmos aplicados no problema das ***Junções de emenda (uniões) das Sequências de Genes (DNA) de Primatas*** é um exercício teórico relevante para nossa compreensão do funcionamento dos algoritmos e dos resultados que emergem de sua aplicação no problema descrito no contexto.

Os objetivos principais neste trabalho é apresentar perspectivas de aplicação de ***diferentes*** algoritmos de Inteligência Artificial, no problema acima descrito, o qual caracterizado como um problema de **classificação**. Mas não somente a utilização dos algoritmos, como também a escolha e justificativas dos Hiperparâmetros escolhidos, os quais sabidamente afetam os resultados obtidos, e que podem inviabilizar as comparações entre eles. Deste modo elencamos os seguintes algoritmos para tentar resolver o problema de classificação das sequências de DNA:
  
  - [Rede Neural KBANN (Knowledge-Based Artificial Neural Networks)](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://www.inf.ufrgs.br/~engel/data/media/file/cmp121/kbann-artigo.pdf&ved=2ahUKEwiFn9yCqdWKAxX4BrkGHWFMEq8QFnoECBUQAQ&usg=AOvVaw2yzjp752bnqkN2fu16XcdW)
  - [RandomForestClassifier](https://link.springer.com/article/10.1023/a:1010933404324)
  - [SVM Classifier](https://link.springer.com/article/10.1007/BF00994018)
  - [MLP Classifier - Multi Layer Perceptron](https://www.semanticscholar.org/paper/Learning-representations-by-back-propagating-errors-Rumelhart-Hinton/052b1d8ce63b07fec3de9dbb583772d860b7c769)

Importante lembrar que as teorias aqui apresentadas, especificamente sobre o sequenciamento de DNA e suas interpretações nas questões do Domínio Sequencial, não estão num formalismo acadêmico, mas foram adicionadas para contextualizar o problema que foi objeto para a aplicação dos algoritmos, e ajustes dos seus respectivos hiperparâmetros. Entretanto colocamos os links para os temas abordadas, caso haja interesse para melhor entendimento e esclarecimentos.

Os ajustes dos hiperparâmetros foram realizados e discutidos nos códigos em Python, dentro das funções que executam cada algoritmo individual, e seus resultados foram apresentados de forma gráfica e também discutidos e apresentados no item Resultados abaixo.

## Resultados

**Sobre a Rede Neural KBANN**

Este algoritmo combina redes neurais com conhecimento pré-existente. Na prática, não tendo uma implementação padrão do KBANN nas bibliotecas mais comuns de Python como TensorFlow ou PyTorch. A implementação geralmente envolve uma estrutura manual para utilizar conhecimento aprendido e previamente estabelecido na arquitetura da rede.


## Sobre o Trabalho

O presente trabalho foi desenvolvido como exercício final para a disciplina de **CA006NC - Inteligência Artificial**, do curso de Mestrado em Inteligência Artificial, dentro do PPGI - Programa de Pós-Graduação em Informática, da UTFPR Cornélio Procópio/PR

Elaboração: **Luiz Antonio Silva de Paula**
