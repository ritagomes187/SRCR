%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Trabalho prático SRCR 20/21 - Grupo 17
%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% PROLOG: Declaracoes iniciais
:- set_prolog_flag(discontiguous_warnings,off).
:- set_prolog_flag(single_var_warnings,off).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% PROLOG: definicoes iniciais
:- op(900,xfy,'::').
:- dynamic utente/10.
:- dynamic centro_saude/5.
:- dynamic staff/4.
:- dynamic vacinacao_Covid/6.
:- dynamic consulta/5.
:- dynamic tratamento/5.
:- dynamic receita/5.
:- discontiguous (::)/2.
:- discontiguous (-)/1.

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado utente: #Idutente, Nº Segurança_Social, Nome, Data_Nasc, Email, Telefone, Morada,
%                                           Profissão, [Doenças_Crónicas], #CentroSaúde -> {V,F}
utente(0, '970530380', 'Joao Tadeu'      , '10-01-1999', 'joaoTadeu99@gmail.com' , '939827435', 'Rua Perto de Braga'     , 'Estudante'  , [], 0).
utente(1, '988877885', 'Jose Reis'       , '10-04-1999', 'josereis@hotmail.com'  , '910753465', 'Rua Santa Eugenia'      , 'Enfermeiro' , ['Asma'], 1).
utente(2, '606028459', 'Jose Manso'      , '24-11-2000', 'josemansinho@yahoo.com', '935581435', 'Rua Mais Perto de Braga', 'Medico'     , ['Parkinson'], 1).
utente(3, '722084637', 'Rita Gomes'      , '31-05-2000', 'ritagomees@gmail.com'  , '921293006', 'Rua em Braga'           , 'Estudante'  , [], 2).
utente(4, '415345342', 'Pelo Nome'       , '25-04-1930', 'semEmail@none.pt'      , '963582458', 'Rua um bocado longe'    , 'Reformado'  , ['Asma', 'Colesterol', 'Hipertensao'], 2).
utente(5,  null      , 'Luciana Abreu'   , '25-05-1985', 'floribela@sic.pt'      , '984310706', 'Rua do Professor'       , 'Atriz'      , [], 3).
utente(6, '214583755', 'Alberto Pereira' , '01-01-1945', 'betinho@yahoo.com'     , '979382414',  interdito               , 'Reformado'  , ['Alzheimer', 'Colesterol', 'Hipertensao'], 3).
utente(7, '868975797', 'Ricardo Quaresma', '26-09-1983', 'rq@vsc.pt'             , '988370144', 'Avenida Sao Goncalo'    , 'Futebolista', [], 3).
% Conhecimento negativo de utente
-utente(I,S,N,D,E,Nr,M,P,D,C) :- nao(utente(I,S,N,D,E,Nr,M,P,D,C)),
                                 nao(excecao(utente(I,S,N,D,E,Nr,M,P,D,C))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado centro_saude: #Idcentro, Nome, Morada, Telefone, Email -> {V,F}
centro_saude(0, 'Centro de Barceladas Catitas'    , 'Casa do Rafa nº 22'        , 252144785, 'cbc@dgs.gmail.com').
centro_saude(1, 'Clinica dos Cowboys'             , 'Rua Cringe, nº 11'         , 253412238, 'cowboys@dgs.outlook.pt').
centro_saude(2, 'Centro de Saude Pires & Chávenas', 'Avenida do 9 e meio'       , 254961222,  null).
centro_saude(3, 'Centro de Saude Mansos Lda.'     , 'Praca dos merges conflicts', 254786950, 'centromansos@dgs.mimimi.com').
centro_saude(5, 'Centro Falência'                 ,  interdito                  , 253014687, 'fomos@falencia.pt').
% Conhecimento negativo de centro de saude
-centro_saude(I,N,M,T,E) :- nao(centro_saude(I,N,M,T,E)),
                            nao(excecao(centro_saude(I,N,M,T,E))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado staff: #IdStaff, #Idcentro, Nome, Email, Fase -> {V,F}
staff(0 , 0, 'Lewis Hamilton', 'lewishamilton@dgs.cbc.com').
staff(1 , 0, 'Batmene'       , 'batmene@dgs.cbc.com').
staff(2 , 0, 'Andre Andre'   , 'andretwice@dgs.cbc.com').
staff(3 , 0, 'Marcus Edwards', 'edwards.marcus@dgs.cbc.com').

staff(4 , 1, 'Bruno Varela'  , 'brunovarela@dgs.cowboys.pt').
staff(6 , 1, 'Pedro Henrique', 'pedrohenrique@dgs.cowboys.pt').
staff(7 , 1, 'Lyle Foster'   , 'lylefoster@dgs.cowboys.pt').

staff(8 , 2, 'Joao Carlos Teixeira', 'jct@dgs.pc.pt').
staff(9 , 2, 'Max Verstappen'      , 'verstappen@dgs.pc.pt').
staff(10, 2, 'Daniel Ricciardo'    , 'danielric@dgs.pc.pt').
staff(11, 2, 'Cristina Ferreira'   , 'criscris@dgs.pc.pt').

staff(12, 3, 'Claire Williams', 'williams.claire@dgs.mansos.com').
staff(13, 3, 'Serena Williams', 'serenawilliams@dgs.mansos.com').
staff(14, 3, 'Maria Sharapova', 'sharapova.m@dgs.mansos.com').
staff(15, 3, 'Dulce Felix'    , 'dulcefelix@dgs.mansos.com').
% Conhecimento negativo de staff
-staff(S,C,N,E,F) :- nao(staff(S,C,N,E,F)),
                     nao(excecao(staff(S,C,N,E,F))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado vacinacao_Covid: #Staff, #utente, Data, Vacina, Toma, Fase -> {V,F}
vacinacao_Covid(0 , 1, '20-01-2021', 'Oxford'     , 1, 1).
vacinacao_Covid(2 , 1, '20-01-2021', 'Oxford'     , 2, 1).
vacinacao_Covid(4 , 2,  null       , 'Pfizer'     , 1, 3).
vacinacao_Covid(6 , 2, '20-01-2021', 'Pfizer'     , 2, 3).
vacinacao_Covid(8 , 3, '08-02-2021', 'BioNYech'   , 1, 3).
vacinacao_Covid(12, 7, '01-02-2021', 'AstraZeneca', 1, 2).
% Conhecimento negativo de vacinacao_covid
-vacinacao_Covid(S,U,D,V,T,F) :- nao(vacinacao_Covid(S,U,D,V,T,F)),
                                 nao(excecao(vacinacao_Covid(S,U,D,V,T,F))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado consulta: #Id, #Staff, #utente, Data, Tipo de Consulta -> {V,F}
consulta(0, 0 , 0, '12-03-2021', 'Pedologia').
consulta(1, 4 , 1, '14-02-2021', 'Clinica Geral').
consulta(2, 4 , 1, '15-02-2021',  null).
consulta(3, 12, 6, '11-01-2021', 'Cardiologia').
consulta(4, 12, interdito, '25-01-2021', 'Ginecologia').
consulta(5, 13, 6, '29-01-2021', 'Podologia').
consulta(6, 15, 7, '06-03-2021', 'Exames medicos').
% Conhecimento negativo de consulta
-consulta(I,S,U,D,T) :- nao(consulta(I,S,U,D,T)),
                        nao(excecao(consulta(I,S,U,D,T))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado tratamento: #Id, #Staff, #utente, Data, Tipo de tratamento -> {V,F}
tratamento(0,  0, 0, '22-03-2021', 'Tratamento pedologico').
tratamento(1,  5, 2, '20-02-2021', 'Exames desportivos').
tratamento(2,  4, 1, '17-03-2021', 'Exames desportivos').
tratamento(3, 12, 6, '14-01-2021', 'Exame de resistencia').
tratamento(4, 13, 6, '12-02-2021', 'Tratamento pedologico').
tratamento(5, 10, 4, '23-01-2021', 'Raio X').
% Conhecimento negativo de tratamento
-tratamento(I,S,U,D,T) :- nao(tratamento(I,S,U,D,T)),
                          nao(excecao(tratamento(I,S,U,D,T))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Povoamento do predicado receita: #Id, #Staff, #utente, Data, Nome -> {V,F}
receita(0, 4 , 1, '14-02-2021', 'Ben-u-ron').
receita(1, 12, 6, '11-01-2021', 'Brufen').
receita(2, 13, 6, '12-02-2021', 'Montelucaste').
receita(3, 0,  0, '22-03-2021', 'Kestine').
% Conhecimento negativo de receita
-receita(Id, Staff, Utente, Data, Nome) :- nao(receita(Id, Staff, Utente, Data, Nome)),
                                           nao(excecao(receita(Id, Staff, Utente, Data, Nome))).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Exceções para o conhecimento imperfeito (Incerto, Interdito e Impreciso)
excecao(utente(I,_,N,D,E,Nr,M,P,D,C)) :- utente(I,null,N,D,E,Nr,M,P,D,C).
excecao(utente(4 ,'925775811','Timoteo Mata','25-12-1970','titeomata@outlook.com','990247466','Rua da Mata', 'Coveiro', ['Osteoporose'], 2)).
excecao(utente(4 ,'925775811','Timoteo Mata','25-12-1970','titeomata@outlook.com','990247466','Rua da Mata', 'Coveiro', ['Osteoporose'], 3)).
excecao(utente(I,S,N,D,E,Nr,_,P,D,C)) :- utente(I,S,N,D,E,Nr,interdito,P,D,C).
excecao(centro_saude(I,N,M,T,_)) :- centro_saude(I,N,M,T,null).
excecao(centro_saude(4, 'Clinica Aveiro', 'Aveiro', 963852741, 'aveiro@aveiro.pt')).
excecao(centro_saude(4, 'Clinica Aveiro', 'Aveiro', 253698147, 'aveiro@aveiro.pt')).
excecao(centro_saude(I,N,_,T,E)) :- centro_saude(I,N,interdito,T,E).
excecao(staff(5, 1,'Rochinha','rochinhaM@dgs.cowboys.pt')).
excecao(staff(5, 2,'Rochinha','rochinhaM@dgs.cowboys.pt')).
excecao(vacinacao_Covid(S,U,_,V,T,F)) :- vacinacao_Covid(S,U,null,V,T,F).
excecao(vacinacao_Covid(12, 8, '01-02-2021', 'AstraZeneca', 1, 2)).
excecao(vacinacao_Covid(12, 8, '01-02-2021', 'Oxford'    , 1, 2)).
excecao(consulta(I,S,U,D,_)) :- consulta(I,S,U,D,null).
excecao(consulta(7, 1, 2, '29-01-2021', 'Cirurgia')).
excecao(consulta(7, 2, 2, '29-01-2021', 'Cirurgia')).
excecao(consulta(I,S,_,D,T)) :- consulta(I,S,interdito,D,T).
excecao(tratamento(0, 4, 1, '21-02-2021', 'Picalm')).
excecao(tratamento(0, 4, 1, '21-02-2021', 'Fenistil')).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Valores nulos, para exceções interditas
nulo(interdito).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado que calcula o risco das Profissões (Se for de risco devolve 2, caso contrario 0)
riscoProfissao('Dentista', 2) :- !.
riscoProfissao('Medico', 2) :- !. riscoProfissao('Medica', 2) :- !.
riscoProfissao('Enfermeiro', 2) :- !. riscoProfissao('Enfermeira', 2) :- !.
riscoProfissao('Professor', 2) :- !. riscoProfissao('Professora', 2) :- !.
riscoProfissao('Educador de Infancia', 2) :- !. riscoProfissao('Educadora de Infancia', 2) :- !.
riscoProfissao('Bombeiro', 2) :- !. riscoProfissao('Bombeira', 2) :- !.
riscoProfissao(_, 0).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado fasesVacinacao: {Fases, Vacinas, X} -> {V,F}
fasesVacinacao(Fases, Vacinas, X) :- -vacinados(NaoVacinados),
                                     associaRiscoUtente(NaoVacinados, [], L),
                                     distribui(Fases, Vacinas, Vacinas, L, [], [],X),!.

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado fasesVacinacaoSegundaToma: {Fases, Vacinas, X} -> {V,F}
fasesVacinacaoSegundaToma(Fases, Vacinas, X) :- segundaToma(NaoVacinados),
                                                associaRiscoUtente(NaoVacinados, [], L),
                                                distribui(Fases, Vacinas, Vacinas, L, [], [],X),!.

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Distribui uma lista de utentes por várias fases, as fases estão limitadas a um numero de vacinas
distribui(0,_,_,_,_,R,R).
distribui(_,_,_,[],L2,LF,R) :- append([L2],LF,R).
distribui(F,V,0,L1,L2,LF,R) :- F2 is F - 1, distribui(F2,V,V,L1,[],[L2|LF],R).
distribui(F,V1,V2,[(_,N)|T],L,LF,R) :- V3 is V2 - 1, distribui(F,V1,V3,T,[N|L],LF,R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Cria uma lista (Utente, Risco), ordenada decrescentemente pelo Risco
associaRiscoUtente([],L,R) :- ordenaDec(L,R).
associaRiscoUtente([H|T],X,R) :- calculaRiscoUtente(H,Risco), associaRiscoUtente(T, [(Risco,H)|X],R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Calcula o nivel de risco de um utente
% Idade > 65 = +1 de Risco
% Profissao de Risco = +2 de Risco
% Doenças = +2 de Risco p/doença
calculaRiscoUtente(Nome, Risco) :- utente(_,_,Nome,DataNascimento,_,_,_,Profissao,_,_),
                                   riscoProfissao(Profissao, R), quantasDoencas(Nome, Doencas),
                                   calculaIdade(DataNascimento, Idade), idoso(Idade, V),
                                   Risco is R + V + Doencas * 2.

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado idoso: {Idade, Risco} -> {V,F}
idoso(Idade, 1) :- Idade > 64, !.
idoso(_, 0).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado que calcula o número de doencas crónicas de um utente
quantasDoencas(Nome, N) :- utente(_,_,Nome,_,_,_,_,_,L,_),
                           comprimento(L, N).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado calculaIdade: {Data, Idade} -> {V,F}
calculaIdade([], _).
calculaIdade(D, I) :- separaLista(D, "-", "", L1),
                      string2listInt(L1, [], L2),
                      idade(L2,I).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Converte uma lista de strings numa lista de inteiros
string2listInt([], L, L).
string2listInt([H|T], L, L1) :- atom_number(H, R),
                                string2listInt(T, [R|L], L1).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado que calcula a idade
idade([Ano,Mes,Dia], I) :- hoje([Hdia,Hmes,Hano]), Mes == Hmes, Dia > Hdia, I is Hano - Ano - 1,!.
idade([Ano,Mes,_], I)   :- hoje([_,Hmes,Hano]), Mes > Hmes, I is Hano - Ano - 1,!.
idade([Ano,_,_], I)     :- hoje([_,_,Hano]), I is Hano - Ano.

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Calcular o dia de hoje e pôr num array no formato [Dia, Mes, Ano]
hoje(L) :- dia(D), mes(M), ano(A), append([D,M], [A], L).

dia(Dia) :- get_time(Stamp), stamp_date_time(Stamp, DateTime, local),
            date_time_value(day, DateTime, Dia).

mes(Mes) :- get_time(Stamp), stamp_date_time(Stamp, DateTime, local),
            date_time_value(month, DateTime, Mes).

ano(Ano) :- get_time(Stamp), stamp_date_time(Stamp, DateTime, local),
            date_time_value(year, DateTime, Ano).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensao do predicado vacinados: [Utente] -> {V,F}
vacinados(X) :- solucoes(N , vacinado(N), L),
                ordena(L,X).

% Extensao do predicado vacinado: {NomeUtente} -> {V,F}
vacinado(X) :- utente(N,_,X,_,_,_,_,_,_,_),
               vacinacao_Covid(_,N,_,_,_,_).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Identificar pessoas não vacinadas
-vacinados(X) :- solucoes(N, -vacinado(N), L),
                 ordena(L,X).

-vacinado(X) :- utente(N,_,X,_,_,_,_,_,_,_),
                nao(vacinacao_Covid(_,N,_,_,_,_)).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Identificar pessoas vacinadas indevidamente
vacInd(Fase,Lista) :- vacinadosFase(L2, Fase),
                      ordena(L2, L),
                      vacIndAux(L, Fase, [], Lista),!.

vacinadoFase(Nome, Fase) :- utente(Id,_,Nome,_,_,_,_,_,_,_),
                         vacinacao_Covid(_,Id,_,_,_,Fase).
vacinadosFase(L, Fase) :- solucoes(N, vacinadoFase(N, Fase), X),
                          ordena(X, L).

vacIndAux([],_, R, R).
vacIndAux([H|T], 1, L2, R) :- calculaRiscoUtente(H, Risco),
                              Risco < 2, !,
                              vacIndAux(T, 1, [H|L2], R).
vacIndAux([_|T], 1, L2, R) :- vacIndAux(T, 1, L2, R).
vacIndAux([H|T], 2, L2, R) :- calculaRiscoUtente(H, Risco),
                              Risco \= 1, !,
                              vacIndAux(T, 2, [H|L2], R).
vacIndAux([_|T], 2, L2, R) :- vacIndAux(T, 2, L2, R).
vacIndAux([H|T], 3, L2, R) :- calculaRiscoUtente(H, Risco),
                              Risco \= 0, !,
                              vacIndAux(T, 3, [H|L2], R).
vacIndAux([_|T], 3, L2, R) :- vacIndAux(T, 3, L2, R).
vacIndAux(_,_,[],[]).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Identificar pessoas não vacinadas e que são candidatas a vacinação
candidatos(X) :- -vacinados(Lista), candidatosAux(Lista, [], X).

candidatosAux([],L,L).
candidatosAux([H|T], L, R) :- calculaRiscoUtente(H, Risco),
                              Risco > 0,
                              candidatosAux(T,[H|L],R),!.
candidatosAux([_|T], L, R) :- candidatosAux(T,L,R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Identificar pessoas a quem falta a segunda toma da vacina
segundaToma(X) :- solucoes(U, segundaTomaAux(U), X).

segundaTomaAux(X) :- utente(N,_,X,_,_,_,_,_,_,_),
                  vacinacao_Covid(_,N,_,_,1,_),
                  nao(vacinacao_Covid(_,N,_,_,2,_)).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Listagem do staff de um centro de saude [(Staff,Vacinado)] -> {V,F}
listagemStaff(Centro, L) :- solucoes(Nome, staff(_,Centro,Nome,_), Staff),
                            staffVacinado(Staff, [], L).

staffVacinado([],L,L).
staffVacinado([H|T],L,R) :- vacinado(H), staffVacinado(T, [[H,1]|L],R), !.
staffVacinado([H|T],L,R) :- staffVacinado(T, [[H,0]|L],R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Listagem da idade dos utentes por ordem decrescente
listagemDecrescente(X) :- solucoes((Nome, DataNascimento),
                          utente(_,_,Nome,DataNascimento,_,_,_,_,_,_), S),
                          ld(S, [], L1),
                          ordenaDec(L1,X).

ld([],L,L).
ld([(N,D)|T],L,R) :- calculaIdade(D,I), ld(T,[(I,N)|L],R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Ver o registo das receitas médicas de um utente
registo(Utente, X) :- solucoes(Medicamento, receita(_,_,Utente,_,Medicamento), X).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado que verifica os utentes que efetuaram um tratamento sem consulta prévia
tratamentoSemConsulta(L) :- solucoes(IdU , tratamento(_,_,IdU,_,_), L2),
                            tscAux(L2, [], L),!.

tscAux([], L, L).
tscAux([Id|T], L, R) :- solucoes(Id, consulta(_,_,Id,_,_), L2),
                        comprimento(L2,N), N == 0,
                        utente(Id,_,Nome,_,_,_,_,_,_,_),
                        tscAux(T, [Nome|L], R).
tscAux([Id|T], L, R) :- solucoes(Id, consulta(_,_,Id,_,_), L2),
                        comprimento(L2,N), N > 0,
                        tscAux(T, L, R).


doenca(Doenca, L) :- solucoes(Nome, (utente(_,_,Nome,_,_,_,_,_,Doencas,_), membro(Doenca, Doencas)), L).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensao do meta-predicado demo (Sistema de inferencia)
demo(Questao, verdadeiro) :- Questao.
demo(Questao, falso) :- -Questao.
demo(Questao, desconhecido) :- nao(Questao), nao(-Questao).
%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensao do meta-predicado demoConj (Conjunção entre duas questoes)
demoConj(Questao1, Questao2, R) :- demo(Questao1, Q1),
                                   demo(Questao2, Q2),
                                   conjuncao(Q1, Q2, R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do meta-predicado conjunção
% verdadeiro e desconhecido => desconhecido
conjuncao(verdadeiro, desconhecido, desconhecido).
conjuncao(desconhecido, verdadeiro, desconhecido).
% verdadeiro e falso => falso
conjuncao(verdadeiro, falso, falso).
conjuncao(falso, verdadeiro, falso).
% falso e desconhecido => falso
conjuncao(falso, desconhecido, falso).
conjuncao(desconhecido, falso, falso).
% algo e algo => algo
conjuncao(verdadeiro, verdadeiro, verdadeiro).
conjuncao(desconhecido, desconhecido, desconhecido).
conjuncao(falso, falso, falso).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensao do meta-predicado demoDisj (Disjunção entre duas questoes)
demoDisj(Questao1, Questao2, R) :- demo(Questao1, Q1),
                                   demo(Questao2, Q2),
                                   disjuncao(Q1, Q2, R).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do meta-predicado disjunção
% verdadeiro ou desconhecido => verdadeiro
disjuncao(verdadeiro, desconhecido, verdadeiro).
disjuncao(desconhecido, verdadeiro, verdadeiro).
% verdadeiro ou falso => verdadeiro
disjuncao(verdadeiro, falso, verdadeiro).
disjuncao(falso, verdadeiro, verdadeiro).
% falso ou desconhecido => desconhecido
disjuncao(falso, desconhecido, desconhecido).
disjuncao(desconhecido, falso, desconhecido).
% algo ou algo => algo
disjuncao(verdadeiro, verdadeiro, verdadeiro).
disjuncao(desconhecido, desconhecido, desconhecido).
disjuncao(falso, falso, falso).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensao do meta-predicado nao: Questao -> {verdadeiro, falso, desconhecido}
nao(Questao) :- Questao, !, fail.
nao(_).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Estrutural:  nao permitir a insercao de conhecimento repetido do predicado utente (Id repetido)
+utente(Id,_,_,_,_,_,_,_,_,_) :: (solucoes(Id, utente(Id,_,_,_,_,_,_,_,_,_), S),
                                 comprimento(S,N),
                                 N == 1).
% Invariante Estrutural:  nao permitir a insercao de conhecimento repetido do predicado utente
%                                                                       (Número de Segurança Social repetido)
+utente(_,Ss,_,_,_,_,_,_,_,_) :: (solucoes(Ss, utente(_,Ss,_,_,_,_,_,_,_,_), S),
                                 comprimento(S,N),
                                 N == 1).
% Invariante Referencial: garante que o identificador do Centro existe
+utente(_,_,_,_,_,_,_,_,_,Centro) :: (existeCentroId(Centro)).
% Invariante Referencial: garante que não colocamos conhecimento interdito
+utente(I,S,N,D,E,Nr,Morada,P,D,C) :: (solucoes(Morada, (utente(I,S,N,D,E,Nr,Morada,P,D,C), nao(nulo(Morada)), S),
                                      comprimento(S,N),
                                      N == 0)).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Estrutural:  nao permitir a insercao de conhecimento repetido do predicado centro_saude (Id repetido)
+centro_saude(Id,_,_,_,_) :: (solucoes(Id, centro_saude(Id,_,_,_,_), S),
                             comprimento(S,N),
                             N == 1).
% Invariante Estrutural:  nao permitir a insercao de conhecimento repetido do predicado centro_saude
%                                                                               (Numero de telemovel repetido)
+centro_saude(_,_,_,Nr,_) :: (solucoes(Nr, centro_saude(_,_,_,Nr,_), S),
                             comprimento(S,N),
                             N == 1).
% Invariante Referencial: nao permitir remover um centro se ainda houver staff na base de conhecimento
-centro_saude(Centro,_,_,_,_) :: (solucoes(Centro, staff(_,Centro,_,_), S),
                                 comprimento(S,N),
                                 N == 0).
% Invariante Referencial: garante que não colocamos conhecimento interdito
+centro_saude(I,N,Morada,T,E) :: (solucoes(Morada, (centro_saude(I,N,Morada,T,E), nao(nulo(Morada))), S),
                                 comprimento(S,N),
                                 N == 0).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Estrutural:  nao permitir a insercao de conhecimento repetido do predicado staff (Id repetido)
+staff(Id,_,_,_) :: (solucoes(Id, staff(Id,_,_,_), S),
                    comprimento(S,N), N==1).
% Invariante Referencial: garante que o centro onde o staff inserido trabalha existe
+staff(_,Centro,_,_) :: (existeCentroId(Centro)).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Referencial: garante que o utente não levou esta Toma da vacina
+vacinacao_Covid(_,Utente,_,_,Toma,_) :: (solucoes((Utente, Toma), 
                                         vacinacao_Covid(_,Utente,_,_,Toma,_), S),
                                         comprimento(S,N), N==1).
% Invariante Referencial: garante que o identificador do Staff existe
+vacinacao_Covid(Staff,_,_,_,_,_) :: (existeStaffId(Staff)).
% Invariante Referencial: garante que o identificador do Utente existe
+vacinacao_Covid(_,Utente,_,_,_,_) :: (existeUtenteID(Utente)).
% Invariante Referencial: garante que a Fase da vacinação é inferior a 4
+vacinacao_Covid(_,_,_,_,_,Fase) :: (Fase < 4).
% Invariante Referencial: garante que o utente levou a primeira Toma antes de levar a segunda
+vacinacao_Covid(_,Utente,_,_,2,_) :: (vacinacao_Covid(_,Utente,_,_,1,_)).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Estrutural: não permite a inserção de um identificador já existente
+consulta(Id,_,_,_,_) :: (solucoes(Id, consulta(Id,_,_,_,_), S),
                         comprimento(S,N),
                         N==1).
% Invariante Referencial: garante que o identificador do Staff existe
+consulta(_,Staff,_,_,_) :: (existeStaffId(Staff)).
% Invariante Referencial: garante que o identificador do Utente existe
+consulta(_,_,Utente,_,_) :: (existeUtenteID(Utente)).
% Invariante Referencial: não permite a inserção de conhecimento interdito
+consulta(I,S,Utente,D,T) :- (solucoes(Utente, (consulta(I,S,Utente,D,T), nao(nulo(Utente))), S),
                             comprimento(S,N),
                             N == 0).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Estrutural: garante que não existem tratamentos com o mesmo identificador
+tratamento(Id,_,_,_,_) :: (solucoes(Id, tratamento(Id,_,_,_,_), L),
							comprimento(L,N),
							N == 1).
% Invariante Referencial: garante que o membro do staff está presente na base de conhecimento
+tratamento(_,Id,_,_,_) :: existeStaffId(Id).
% Invariante Referencial: garante que o utente está presente na base de conhecimento
+tratamento(_,_,Id,_,_) :: existeUtenteID(Id).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Invariante Estrutural: não permite a inserção de um identificador já existente
+receita(Id,_,_,_,_) :: (solucoes(Id, receita(Id,_,_,_,_), S),
                        comprimento(S, N),
                        N == 1).
% Invariante Referencial: garante que o identificador do Staff existe
+receita(_,Staff,_,_,_) :: (existeStaffId(Staff)).
% Invariante Referencial: garante que o identificador do Utente existe
+receita(_,_,Utente,_,_) :: (existeUtenteID(Utente)).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Verificam se o id existe
existeCentroId(X) :- centro_saude(X,_,_,_,_).
existeStaffId(X) :- staff(X,_,_,_).
existeUtenteID(X) :- utente(X,_,_,_,_,_,_,_,_,_).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado que permite a evolucao do conhecimento
evolucao(Termo) :- solucoes(Invariante, +Termo::Invariante, Lista),
	               insercao(Termo),
	               teste(Lista).

insercao(Termo) :- assert(Termo).
insercao(Termo) :- retract(Termo), !, fail.

teste([]).
teste([R|LR]) :- R, teste(LR).

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Extensão do predicado que permite a involucao do conhecimento
involucao(Termo) :- solucoes(Invariante, -Termo::Invariante, Lista),
	                remocao(Termo),
	                teste(Lista).

remocao(Termo) :- retract(Termo).
remocao(Termo) :- assert(Termo), !, fail.

%--------------------------------- - - - - - - - - - -  -  -  -  -   -   -   -    -    -    -    -     -     -
% Tradução de alguns predicados já definidos em inglês
solucoes(X,Y,Z) :- findall(X,Y,Z).
comprimento(S,N) :- length(S,N).
ordena(L,X) :- sort(L,X).
separaLista(S, C, W, L) :- split_string(S, C, W, L).
ordenaDec(L,X) :- sort(L,X1), reverse(X1,X).
membro(X,Xs) :- member(X,Xs).