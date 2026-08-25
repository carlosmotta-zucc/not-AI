# Sprint 2 - Frente A: resultados dos experimentos

Gerado por `experimentos/sprint02/frente_a_experimentos.py`. Nao editar a mao: rode o script novamente.

Corpus: The Verdict, 20 479 caracteres.

## E1 - Contagem de tokens em textos de tamanhos diferentes

**Pergunta:** a contagem de tokens e propriedade do texto ou do par texto + tokenizador? E o vocabulario cresce junto com o corpus?

**Configuracao:** prefixos de The Verdict com 500, 1 000, 2 500, 5 000, 10 000, 20 479 caracteres, tokenizados pelo splitter proprio e pelo BPE do GPT-2.

| caracteres | tokens (proprio) | tokens distintos | distintos/total | tokens (BPE) | car./token proprio | car./token BPE |
| ---------- | ---------------- | ---------------- | --------------- | ------------ | ------------------ | -------------- |
| 500        | 110              | 73               | 0.66            | 123          | 4.55               | 4.07           |
| 1 000      | 236              | 142              | 0.60            | 259          | 4.24               | 3.86           |
| 2 500      | 570              | 256              | 0.45            | 623          | 4.39               | 4.01           |
| 5 000      | 1 106            | 426              | 0.39            | 1 223        | 4.52               | 4.09           |
| 10 000     | 2 222            | 751              | 0.34            | 2 476        | 4.50               | 4.04           |
| 20 479     | 4 730            | 1 149            | 0.24            | 5 145        | 4.33               | 3.98           |

## E2 - Comportamento diante de palavra fora do vocabulario

**Pergunta:** qual e o custo de o vocabulario ser fechado, e o que muda quando <|unk|> entra ou quando o tokenizador e subword?

**Configuracao:** vocabulario construido sobre The Verdict inteiro, em duas versoes -- com e sem <|unk|>. As frases de teste usam palavras que nao aparecem no conto.

Vocabulario sem especiais: 1149 entradas. Com <|endoftext|> e <|unk|>: 1151.

| caso                           | tokens | ausentes | sem <|unk|>               | viram <|unk|> | tokens BPE | BPE reconstroi |
| ------------------------------ | ------ | -------- | ------------------------- | ------------- | ---------- | -------------- |
| palavra comum ausente do conto | 7      | 1        | KeyError em 'Hello'       | 1             | 7          | sim            |
| nome proprio inexistente       | 2      | 2        | KeyError em 'Akwirw'      | 2             | 6          | sim            |
| termo tecnico moderno          | 6      | 3        | KeyError em 'transformer' | 3             | 6          | sim            |
| duas palavras novas distintas  | 2      | 2        | KeyError em 'quantum'     | 2             | 3          | sim            |

Colapso de informacao com <|unk|>, entrada `quantum blockchain`:

```
tokens        : ['quantum', 'blockchain']
Token IDs     : [1150, 1150]
decode        : <|unk|> <|unk|>
BPE tokens    : ['quant', 'um', ' blockchain']
BPE decode    : quantum blockchain
```

## E3 - Portugues contra ingles

**Pergunta:** o BPE do GPT-2, treinado majoritariamente em ingles, gasta mais tokens para dizer a mesma coisa em portugues?

**Configuracao:** 5 pares de frases equivalentes em conteudo, tokenizadas pelo splitter proprio e pelo BPE.

| par   | PT proprio | EN proprio | PT BPE | EN BPE | custo BPE PT/EN |
| ----- | ---------- | ---------- | ------ | ------ | --------------- |
| 1     | 10         | 10         | 19     | 10     | 1.90x           |
| 2     | 9          | 8          | 21     | 8      | 2.62x           |
| 3     | 9          | 9          | 19     | 9      | 2.11x           |
| 4     | 8          | 9          | 18     | 9      | 2.00x           |
| 5     | 9          | 10         | 22     | 10     | 2.20x           |
| total | 45         | 46         | 99     | 46     | 2.15x           |

Caracteres por token no BPE: portugues 2.74, ingles 5.85.

Fragmentacao de palavras acentuadas pelo BPE:

Os `�` na saida nao sao erro de codificacao: o BPE opera sobre bytes, e um caractere acentuado ocupa dois bytes em UTF-8. Quando a fronteira entre dois tokens cai no meio desse par, nenhum dos dois tokens isolados forma um caractere valido -- so a concatenacao forma. E a prova de que a unidade do BPE e o byte, nao a letra.

```
implementação    -> 5 tokens: ['im', 'plement', 'a', 'ç', 'ão']
atenção          -> 3 tokens: ['aten', 'ç', 'ão']
informações      -> 7 tokens: ['in', 'form', 'a', 'ç', '�', '�', 'es']
não              -> 2 tokens: ['n', 'ão']
implementation   -> 2 tokens: ['im', 'plementation']
attention        -> 2 tokens: ['att', 'ention']
```

## E4 - Comparativo: tokenizador proprio contra BPE

**Pergunta:** quais propriedades separam os dois tokenizadores, e qual delas justifica levar o BPE para as sprints seguintes?

**Configuracao:** The Verdict inteiro (20 479 caracteres).

| propriedade                   | proprio (por palavras) | BPE do GPT-2 (subword) |
| ----------------------------- | ---------------------- | ---------------------- |
| tamanho do vocabulario        | 1 151                  | 50 257                 |
| tokens no corpus              | 4 730                  | 5 145                  |
| caracteres por token          | 4.33                   | 3.98                   |
| palavra fora do vocabulario   | <|unk|> ou KeyError    | fatiada em subwords    |
| reconstroi o texto original   | nao                    | sim                    |
| espaco                        | descartado             | parte do token         |
| precisa de <|unk|>            | sim                    | nao                    |
| vocabulario depende do corpus | sim                    | nao (ja treinado)      |
| tempo de encode do corpus     | 1.5 ms                 | 2.0 ms                 |

Primeiros e ultimos pares (token, ID) do vocabulario proprio:

```
inicio : [('!', 0), ('"', 1), ("'", 2), ('(', 3), (')', 4)]
fim    : [('younger', 1146), ('your', 1147), ('yourself', 1148), ('<|endoftext|>', 1149), ('<|unk|>', 1150)]
```

Mesma frase nos dois tokenizadores, entrada `It's the last he painted, you know.`:

```
proprio tokens : ['It', "'", 's', 'the', 'last', 'he', 'painted', ',', 'you', 'know', '.']
proprio IDs    : [57, 2, 868, 1007, 616, 547, 761, 5, 1145, 610, 8]
proprio decode : It's the last he painted, you know.
BPE tokens     : ['It', "'s", ' the', ' last', ' he', ' painted', ',', ' you', ' know', '.']
BPE IDs        : [1026, 338, 262, 938, 339, 13055, 11, 345, 760, 13]
BPE decode     : It's the last he painted, you know.
```

