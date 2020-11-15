## Establishing Data Mining Goals

The first step in data mining requires you to set up goals for the exercise. Obviously, you must identy the key questions that need to be answered. However, going beyond identifying the key questions are the concerns about the costs and benefits of the exercise. Futhermore, you must determine, in advance, the expected level of accuracy and usefulness of the results obtained from data mining. If Money were no object, you cold throw as many funds as necessary to get the answers required. However, the cost benefit trade-off is always instrumental in determining the goals and scope of the data mining exercise. The level of the accuracy expected from the results also influences the costs. High levels of accuracy, you do not gain much from the exercise, given the diminishing returns. Thus, the cost benefit trade-offs for the desired level of accuracy  are importante considerations for data mining goals.

## Estabelecendo metas de mineração de dados

A primeira etapa na mineração de dados exige que você configure metas para o exercício. Obviamente, você deve identificar as principais perguntas que precisam ser respondidas. No entanto, além de identificar as questões-chave, estão as preocupações com os custos e benefícios do exercício. Além disso, você deve determinar com antecedência o nível esperado de precisão e utilidade dos resultados obtidos na mineração de dados. Se o dinheiro não fosse um objeto, você jogaria a frio quantos fundos fossem necessários para obter as respostas necessárias. No entanto, o trade-off de custo-benefício é sempre fundamental para determinar as metas e o escopo do exercício de mineração de dados. O nível de precisão esperado dos resultados também influencia os custos. Elevados níveis de precisão, você não ganha muito com o exercício, dados os retornos decrescentes. Assim, as compensações de custo-benefício para o nível de precisão desejado são considerações importantes para as metas de mineração de dados.

## Selecting Data

The output of a data mining exercise largely depends upon the quality of data being used. At times, data are readily avaiable for futher processing. For instance, retailers often possess large databases of customer purchases na demographics. On the other hand, data may not be readily available for data mining. In such cases, you must identify other sources of data or even plan new data collection initiatives, including surveys. The type of data, its size, and frequency of collection have a direct bearing on the cost of data mining exercise. Thereforce, identifying the right kind of data needed for data mining that could answer the questions at reasonable costs is critical.

## Selecionando Dados

A saída de um exercício de mineração de dados depende muito da qualidade dos dados que estão sendo usados. Às vezes, os dados estão prontamente disponíveis para processamento posterior. Por exemplo, os varejistas geralmente possuem grandes bancos de dados de compras de clientes na demografia. Por outro lado, os dados podem não estar prontamente disponíveis para mineração de dados. Nesses casos, você deve identificar outras fontes de dados ou até planejar novas iniciativas de coleta de dados, incluindo pesquisas. O tipo de dados, seu tamanho e frequência de coleta afetam diretamente o custo do exercício de mineração de dados. Portanto, é fundamental identificar o tipo certo de dados necessários para a mineração de dados que possa responder às perguntas a custos razoáveis.


## Preprocessing Data
Preprocessing data is na importante step in data mining. Often raw data are messy, containing erroneous or irrelevant data. In addition, even with relevant data, information is sometimes missing. In the preprocessing  stage, you identify the irrelevant atributes of data and expunge such atributes from further consideration. At the same time, identifying the erroneous aspects of the data set and flagging them as such is necessary. For instance, humam error might lead to inadvertent merging or incorrect parsing od information between columns. Data should be subject to checks to ensure integrity. Lastly, you must develop a formal method os dealing  with missing data and determine whether the data are missing randomly or systematically.

If the data were missing randomly,a simple set of solutions would suffice. However, when data are missing in a systematic way, you must determine the impact of missing data on the results. For instance, a particular subset of individuals in a large data set may have refused to diclose their income. Findings relying on na individual’s income as na input would exclude details of those individuals whose income was not reported. This would lead to systematic biases in the analysis. Therefore, you must consider in advance if observations or variables containing missing data be excluded from the entire analysis or parts of it. 


## Pré-processamento de Dados

O pré-processamento de dados é uma etapa importante na mineração de dados. Muitas vezes, os dados brutos são confusos, contendo dados errados ou irrelevantes. Além disso, mesmo com dados relevantes, algumas vezes, as informações estão ausentes. No estágio de pré-processamento, você identifica os atributos irrelevantes dos dados e expurga esses atributos de uma análise mais aprofundada. Ao mesmo tempo, é necessário identificar os aspectos incorretos do conjunto de dados e sinalizá-los como tal. Por exemplo, um erro humamo pode levar a uma fusão inadvertida ou a uma análise incorreta de informações entre colunas. Os dados devem estar sujeitos a verificações para garantir a integridade. Por fim, você deve desenvolver um método formal para lidar com dados ausentes e determinar se os dados estão ausentes aleatoriamente ou sistematicamente.

Se os dados estivessem ausentes aleatoriamente, um conjunto simples de soluções seria suficiente. No entanto, quando os dados estão ausentes de maneira sistemática, você deve determinar o impacto dos dados ausentes nos resultados. Por exemplo, um subconjunto específico de indivíduos em um grande conjunto de dados pode ter se recusado a definir sua renda. As descobertas que dependem da renda de um indivíduo como uma entrada excluiriam detalhes daqueles indivíduos cuja renda não foi informada. Isso levaria a vieses sistemáticos na análise. Portanto, você deve considerar com antecedência se observações ou variáveis ​​contendo dados ausentes serão excluídas de toda a análise ou de partes dela.

## Transforming Data

After the relevant attributesod data have been retained, the next step is to determine the appropriate format in which data must be stored. An important consideration in datat mining is to reduce the number of atributes needed to explainthe phenomena. This may required transforming data. Data reduction algorithms, such as Principal Component Analisys (demonstrated and explained later in the chapter), can reduce the number of atributes without a significant loss in information. In addition, variables may need to be transformed to help explain the phenomenon being studied. For instnace, na individual’s income may be recorded in the datat set as wage income; income from other sources, such as rental properties; support payments from the government, and the like. Aggregating income from all sources will develop a representative indicator for the individual income.

Often you need to transformation variables from one type to another. It may be prudente to transform the continuous variable for income intro a categorical variable where each record in the database is identified as low, médium, and high-income individual. This could help capture the non-linearities in the underlying behaviors.

## Transformando Dados

Após a retenção dos dados dos atributos relevantes, a próxima etapa é determinar o formato apropriado no qual os dados devem ser armazenados. Uma consideração importante na mineração de dados é reduzir o número de atributos necessários para explicar os fenômenos. Isso pode exigir a transformação de dados. Algoritmos de redução de dados, como a Análise de componentes principais (demonstrada e explicada mais adiante neste capítulo), podem reduzir o número de atributos sem uma perda significativa de informações. Além disso, as variáveis podem precisar ser transformadas para ajudar a explicar o fenômeno que está sendo estudado. Por exemplo, a renda de um indivíduo pode ser registrada nos dados definidos como renda salarial; renda de outras fontes, como imóveis para aluguel; apoiar pagamentos do governo e assim por diante. A agregação de renda de todas as fontes desenvolverá um indicador representativo para a renda individual.

Geralmente, você precisa transformar variáveis de um tipo para outro. Pode ser prudente transformar a variável contínua para renda em uma variável categórica em que cada registro no banco de dados seja identificado como indivíduo de baixa, média e alta renda. Isso poderia ajudar a capturar as não linearidades nos comportamentos subjacentes.

## Storing Data

The transformed data must  be stored in a format that makes it conductive for data mining. The data must be stored in a format that gives unrestricted na immediate read/write privileges to the data scientist. During data mining, new variables are created, which are written back to the original database, which is why the data storage scheme should facilitate efficiently Reading from and writing to the database. It i also important to store data on servers or storage media that keeps the data secure and also prevents the data mining algorithm from unnecessarily searching for pieces of data scattered on diferente servers or storage media. Data safety and provacy should be a prime concern for storing data.

## Armazenando Dados

Os dados transformados devem ser armazenados em um formato que os torne condutores para a mineração de dados. Os dados devem ser armazenados em um formato que ofereça privilégios de leitura / gravação sem restrições ao cientista de dados. Durante a mineração de dados, novas variáveis são criadas, que são gravadas de volta no banco de dados original, e é por isso que o esquema de armazenamento de dados deve facilitar com eficiência a leitura e a gravação no banco de dados. Também é importante armazenar dados em servidores ou mídia de armazenamento que os mantenha seguros e também evite que o algoritmo de mineração de dados procure desnecessariamente pedaços de dados espalhados em diferentes servidores ou mídias de armazenamento. A segurança e a provisão de dados devem ser uma das principais preocupações para o armazenamento de dados.

## Mining Data

After data is appropriately processed, transformed, and stored, it is subject to data mining. This step covers data analysis methods, including parametric and non-parametric methods and machine-learning algorithms. A good starting point for data mining is data visualization. Multidimensional views of the data using the advanced graphing capabilities of datat mining software are very hepful in developing a preliminar understanding of the trends hidden in the data set.

Later sections in this chapter detail data mining algorithms and methods.

## Mineração de Dados

Depois que os dados são processados, transformados e armazenados adequadamente, eles ficam sujeitos à mineração de dados. Esta etapa abrange métodos de análise de dados, incluindo métodos paramétricos e não paramétricos e algoritmos de aprendizado de máquina. Um bom ponto de partida para a mineração de dados é a visualização de dados. As visualizações multidimensionais dos dados usando os recursos avançados de gráficos do software de mineração de dados são muito úteis no desenvolvimento de um entendimento preliminar das tendências ocultas no conjunto de dados.

As seções posteriores deste capítulo detalham algoritmos e métodos de mineração de dados.

## Evaluting Mining Results

After results have been extracted froms data mining, you do a formal evaluation of the results. Formal evaluation could include testing teh predictive capabilities of the models on observed data to sees how effective and eficcient the algorithms have been in reproducing data. This is know as na in-sample forecast. In addition, the results are shared with the key stakeholders for feedback, which is then incorporated in the later interations of data mining to improve the process.

Data mining and evaluating the results decomes na interative process such that the analysts use better and improved algorithms to improve the quality of results generated in light of the feedback received from the key stakeholders.

## Avaliação de resultados de mineração

Depois que os resultados são extraídos da mineração de dados, você faz uma avaliação formal dos resultados. A avaliação formal pode incluir o teste dos recursos preditivos dos modelos nos dados observados para ver quão eficazes e eficientes os algoritmos têm sido na reprodução de dados. Isso é conhecido como uma previsão dentro da amostra. Além disso, os resultados são compartilhados com as principais partes interessadas para feedback, que são incorporados nas interações posteriores da mineração de dados para melhorar o processo.

A mineração de dados e a avaliação dos resultados decodificam um processo interativo para que os analistas usem algoritmos cada vez melhores para aprimorar a qualidade dos resultados gerados à luz do feedback recebido dos principais interessados.

## Anotações

Aprendemos que que a Ciência de Dados ajuda os médicos a fornecer o melhor tratamento para seus pacientes e ajuda os meteorologistas a prever a extensão dos eventos climáticos locais e pode até ajudar a prever desastres naturais como terremotos e tornados.

Que as empresas podem começar sua jornada de ciência de dados capturando dados. Depois de terem dados, eles podem começar a analisá-los.
Algumas maneiras pelas quais os dados são gerados pelos consumidores.

Como empresas como Netflix, Amazon, UPs, Google e Apple usam os dados gerados por seus consumidores e funcionários.

O objetivo da entrega final de um projeto de Ciência de Dados é comunicar novas informações e insights da análise de dados aos principais tomadores de decisão.