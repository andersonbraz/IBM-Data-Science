## Chapter 7. Why Tall Parentes Don't Have Even Taller Children

*You mitgh noticed that taller parents often have tall children who are not necessarily taller than their parents - and that's a god thing. This is not to suggest that children born to tall parents are not necessarily taller than the rest. That may be the case, but they are not necessarily taller than their  own "tall" parents. Why I think this to be a good thing requires a simple mental simulation. Imagine is every successive generation born to all parents were taller than their parents, in a matter of couple of millennia, human beings would become uncomfortably tall for their own good, requiring even bigger furniture, cars, and planes.*

Sir Frances Galton in 1886 studied the same question and landed upon a statistical technique we today know as *regression models*. This chapter explores the workings of regression models, which have become the workhorse of statistical analisys. In almost all empirical pursuits of research, either in the academic or professional fields, the use of regression models, or their variants, is ubiquitous. In medical science, regression models are being used to develop more effective medicines, improve the methods for operations, and optimize resources for small and large hispitals. In tehe business world, regression models are at the forefront of analyzing consumer behavior, firm productivity, and competitiveness of public and private sector entities.

I would like to introduce regression models by narrating a story about my Master's thesis. I believe that this story can help explain the utility of regression models.

---

## Capítulo 7. Por que pessoas de grande estatura não têm filhos ainda mais altos

*Você percebeu que os pais mais altos costumam ter filhos altos que não são necessariamente mais altos que os pais - e isso é uma coisa divina. Isso não sugere que crianças nascidas de pais altos não sejam necessariamente mais altas que as demais. Pode ser esse o caso, mas eles não são necessariamente mais altos que seus próprios pais "altos". Por que acho que isso é uma coisa boa, requer uma simulação mental simples. Imagine que toda geração sucessiva nascida de todos os pais era mais alta que seus pais; em questão de dois milênios, os seres humanos se tornariam desconfortavelmente altos para o seu próprio bem, exigindo móveis, carros e aviões ainda maiores.*

Sir Frances Galton, em 1886, estudou a mesma questão e aterrou em uma técnica estatística que hoje conhecemos como *modelos de regressão*. Este capítulo explora o funcionamento dos modelos de regressão, que se tornaram o cavalo de batalha da análise estatística. Em quase todas as pesquisas empíricas de pesquisa, tanto no campo acadêmico quanto no profissional, o uso de modelos de regressão, ou suas variantes, é onipresente. Na ciência médica, modelos de regressão estão sendo usados ​​para desenvolver medicamentos mais eficazes, melhorar os métodos de operações e otimizar recursos para pequenos e grandes hospitais. No mundo dos negócios, os modelos de regressão estão na vanguarda da análise do comportamento do consumidor, da produtividade da empresa e da competitividade de entidades do setor público e privado.

Gostaria de introduzir modelos de regressão narrando uma história sobre minha tese de mestrado. Acredito que essa história possa ajudar a explicar a utilidade dos modelos de regressão.

---

## The Department of Obvious Conclusions

In 1999, I finished my Master's research on developing hedonic price models for residencial real estate properties. It took me there years to complete the project involving 500,000 real state transactions. As I was getting ready for the defense, my wife generously offered to drive me to the university. While we were on our way, she asked, "Tell me, what have you found in your research?" I was delighted to be finally asked to explain what I have been  up to for the past three years. "Well, I have been studing the dereteminants ps housing prices. I have found that larger homes sell for more than smaller homes," I told my wife with a triumphantlook on my face as I held the draft of the thesis in my hands.

We were approaching the on-ramp fora highway. AS soon  as I finished the sentence, my wife suddenly turned the car to the shoulder, and applied brakes. As the car stopped, she turned to me and said: "I can't believe that they are giving you a Master's degree for finding just that. I could have told you that larger homes sell for more than smaller homes.""

At that very moment, I felt like a professor who taught at the department of obvious conclusions. How can I blame her for being shocked that what is commonly know about housing prices will earn me a Master's degree from a university of high repute.

I requested my wife to resume driving so that I could take the next ten minutes to explain her the intricacies of my research. She gave me five nminutes instead, thinking this may not require even that. I settled for five and spent the next minute collecting my thoughts. I explained to her that my research has not just found the correlation between housing prices and size of hounsing units, but I have also discovered the magnitude of those relationships. For instance, I found that *all else being equal,* a term that I explain later in this chapter, an additional washroom adds more to the housing price than an additional bedroom. Stated otherwise, the marginal increasein the price of a house is higher for an additional washroon than for an additional bedroom. I found later that teh real estate brokens in Toronto indeed appreciated this finding.

I also explained to my wife that proximity to transport infrastructure, such as subways, resulted in higher housing prices. For instance, houses situated closer to subways sold for more than did those situated farther away. However, houses near freeways or highways sold for less than others did. Similary, I also discovered that proximity to large shopping centers sold for less than the rest. However, houses located *closer* (less than 5 km, but more than 2.5 km) to th shopping center sold for more than more than did those located farther away. I also found that the housing values in Toronto declined with distance from downtown.

As I explained my contribuitions to the study of housing markets, I noticed that my wife was midly impressed. The likely reason for her lukewarm reception was taht my findings confirmed what we already knew from our everyday experience. However, the real value added by the research rested in quantifying the magnitude of those relationships.

---

Em 1999, terminei minha pesquisa de mestrado no desenvolvimento de modelos de preços hedônicos para imóveis residenciais. Levei anos para concluir o projeto envolvendo 500.000 transações imobiliárias. Enquanto me preparava para a defesa, minha esposa se ofereceu generosamente para me levar para a universidade. Enquanto estávamos a caminho, ela perguntou: "Diga-me, o que você encontrou em sua pesquisa?" Fiquei encantado por finalmente ser convidado a explicar o que tenho feito nos últimos três anos. "Bem, estudei os preços das casas dos depreciativos. Descobri que casas maiores são vendidas por mais do que casas menores", disse a minha esposa com um olhar triunfante enquanto segurava o rascunho da tese em minhas mãos.

Estávamos nos aproximando da rampa da rodovia. Assim que terminei a frase, minha esposa de repente virou o carro para o ombro e freou. Quando o carro parou, ela se virou para mim e disse: "Não acredito que eles estão fazendo um mestrado por encontrar exatamente isso. Eu poderia ter lhe dito que casas maiores são vendidas por mais que casas menores".

Naquele exato momento, eu me senti como um professor que ensinava no departamento conclusões óbvias. Como posso culpá-la por estar chocada com o fato de o que geralmente se sabe sobre os preços das moradias me render um mestrado em uma universidade de alta reputação.

Solicitei que minha esposa voltasse a dirigir para que eu pudesse dedicar os próximos dez minutos para explicar os meandros de minha pesquisa. Ela me deu cinco minutos em vez disso, pensando que isso pode não exigir nem isso. Eu me acomodei por cinco e passei o minuto seguinte coletando meus pensamentos. Expliquei a ela que minha pesquisa não apenas encontrou a correlação entre os preços e o tamanho das unidades, mas também descobri a magnitude desses relacionamentos. Por exemplo, descobri que * todo o resto é igual *, um termo que explico mais adiante neste capítulo, um banheiro adicional acrescenta mais ao preço da moradia do que um quarto adicional. Dito de outra forma, o aumento marginal no preço de uma casa é mais alto para um lavatório adicional do que para um quarto adicional. Descobri mais tarde que os corretores imobiliários em Toronto realmente apreciavam essa descoberta.

Também expliquei à minha esposa que a proximidade com a infraestrutura de transporte, como metrôs, resultou em preços mais altos da habitação. Por exemplo, casas situadas mais próximas aos metrôs são vendidas por mais do que aquelas situadas mais longe. No entanto, casas perto de rodovias ou rodovias são vendidas por menos do que outras. Da mesma forma, descobri também a proximidade com grandes shopping centers vendidos por menos do que o resto. No entanto, as casas localizadas *mais próximas* (a menos de 5 km, mas mais de 2,5 km) do shopping foram vendidas por muito mais do que as localizadas mais longe. Também descobri que os valores da habitação em Toronto diminuíam com a distância do centro da cidade.

Ao explicar minhas contribuições para o estudo do mercado imobiliário, notei que minha esposa ficou impressionada. A provável razão de sua recepção morna foi que minhas descobertas confirmaram o que já sabíamos de nossa experiência cotidiana. No entanto, o verdadeiro valor agregado pela pesquisa residia na quantificação da magnitude desses relacionamentos.

---

## Why Regress?

A whole host of questions could be put to regression analysis. Some examples of questions that regression (hedonic) models could address include:

* How much more can a house sell for an additional bedroom?
* What is the impact of lot size on housing price?
* Do homes with bricks exterior sell for less than homes with stone exterior?
* How much does a finished basement contribute to the price of a housing unit?
* Do houses located near high-voltage power lines sell for more or less than the rest?


---

## Por que regressão?

Uma série de perguntas pode ser colocada na análise de regressão. Alguns exemplos de perguntas que os modelos de regressão (hedônicos) poderiam abordar incluem:

* Quanto mais uma casa pode vender por um quarto adicional?
* Qual o impacto do tamanho do lote no preço da moradia?
* As casas com exterior de tijolos são vendidas por menos do que as casas com exterior em pedra?
* Quanto um porão acabado contribui para o preço de uma unidade habitacional?
* As casas localizadas perto de linhas de alta tensão são vendidas por mais ou menos que o resto?

