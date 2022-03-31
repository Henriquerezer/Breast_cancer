# Breast Cancer
Neste repositório foi utilizado a biblioteca Keras para o desenvolvimento de redes neurais, foi utilizado o dataset de breast cancer, que está incluido no repositório, o mesmo é dividido em Entradas e saídas ( previsores e classes) respectivamente. 

Foi desenvolvido de modo simples com uma rede neural de apenas uma camada, e com o desenvolvimento das aulas, foi sendo aperfeiçoando o modelo de classificação, como por exemplo utilizando validação cruzada, utilização de dropout para diminiuir o overfiting, e por fim foi feito o tuning (ajuste) do modelo de classificação, utilizando gridsearch para encontrar os melhores parâmetros.

Após foi demonstrado como salvar uma rede neural (com os parâmetros fornecidos pelo Gridsearch) no formato de arquivo json, para que não seja preciso rodar o classificador a todo o momento e após o mesmo modelo salvo foi carregado em outro arquivo.
Foi realizado uma nova previsão, utilizando uma nova variavel que o modelo não conhecia, foi escolhido dados aleatórios e realizado a classificação dos dados. O processo simula a inclusão de dados de um novo paciente pelo médico.

Apenas para fins de aprendizado foi incluido novamente o dataset breast cancer, para fazer a classificação. Não é recomendado fazer este tipo de teste em seu modelo, pelo fato do modelo ter sido treinado pela mesma base de dados, então os resultados obtidos não possuem importância.


<div align="center">
<img src="https://user-images.githubusercontent.com/87787728/161064424-2c11e3a4-1570-461a-8d67-b2ad18a1e20f.jpg" width="600px" />
</div>


Este repositório é resultado das atividades desenvolvidas do curso de " Deep learning com python de A a Z - Curso completo" da IA expert Academy, segue o link de todos os cursos ofertados https://iaexpert.academy/
