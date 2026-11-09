# The 30 most frequent template strings of F's FLAN rows

Over the 1,357,022 FLAN rows of F's 4 host mixes: labels and first sentences at the start of a line (and
right after such a label), found in at least 3 tasks; rows = FLAN rows that have the string there. The
renderings are how 27B and 12B translated the string inside a segment that starts a line with it (the
segment starts with it, or its line has the same index in source and output); most frequent first, with
counts. 27B: the pilot and its measurement shards; 12B: the pilot and the run so far.

| English | rows | share | 27B | 12B |
|---|---|---|---|---|
| `A:` | 364,801 | 26.9% | `A:` (1597); `Antwoord:` (57); `V:` (2) | `Antwoord:` (68864); `A:` (21817); `Biografie:` (233) |
| `Q:` | 294,104 | 21.7% | `V:` (1824); `Vraag:` (81) | `Vraag:` (95622); `V:` (2168); `Q:` (100) |
| `Output:` | 234,868 | 17.3% | `Uitvoer:` (918); `Antwoord:` (219); `Resultaat:` (83) | `Uitvoer:` (43254); `Output:` (14127); `Antwoord:` (3184) |
| `Answer:` | 187,473 | 13.8% | `Antwoord:` (659) | `Antwoord:` (38465); `Biografie:` (85); `ANTWOORD:` (3) |
| `Input:` | 156,674 | 11.5% | `Invoer:` (1042); `Gegeven:` (2); `Tekst:` (2) | `Invoer:` (44824); `Input:` (1387); `Context:` (556) |
| `Detailed Instructions:` | 156,574 | 11.5% | `Gedetailleerde instructies:` (1268) | `Gedetailleerde instructies:` (46879) |
| `Question:` | 131,533 | 9.7% | `Vraag:` (910) | `Vraag:` (42305); `Antwoordmogelijkheden:` (3); `Context:` (3) |
| `Problem:` | 84,763 | 6.2% | `Probleem:` (380); `Opdracht:` (21); `Opgave:` (17) | `Probleem:` (25787); `Vraag:` (1208); `Opgave:` (823) |
| `Student:` | 78,658 | 5.8% | `Student:` (366); `Leerling:` (90); `CONCLUSIE:` (1) | `Student:` (21889); `Leerling:` (5) |
| `Teacher:` | 78,581 | 5.8% | `Leraar:` (553); `Docent:` (29) | `Docent:` (23006) |
| `Now, understand the problem?` | 78,451 | 5.8% | `Begrijp je nu het probleem?` (158); `Begrijpt u de opdracht?` (113); `Begrijpt u het probleem nu?` (69) | `Begrijpt u het probleem nu?` (14580); `Begrijpt u het probleem?` (3774); `Begrijpt u nu het probleem?` (2930) |
| `Instructions:` | 78,349 | 5.8% | `Instructies:` (580) | `Instructies:` (23073) |
| `Definition:` | 78,324 | 5.8% | `Definitie:` (662) | `Definitie:` (23491); `Vertaling:` (3) |
| `You will be given a definition of a task first, then some input of the task.` | 78,195 | 5.8% | `Eerst krijgt u een definitie van een taak, daarna een aantal gegevens die bij die taak horen.` (73); `U krijgt eerst een definitie van een taak, gevolgd door een beschrijving van de taak.` (69); `Eerst krijgt u een definitie van een taak, daarna een aantal gegevens die bij de taak horen.` (62) | `U krijgt eerst een definitie van een taak, gevolgd door de input voor die taak.` (15557); `U krijgt eerst een definitie van een taak, gevolgd door de invoer voor die taak.` (1181); `U krijgt eerst een definitie van de taak, gevolgd door de input voor de taak.` (928) |
| `Solution:` | 78,044 | 5.8% | `Oplossing:` (389); `Antwoord:` (8) | `Oplossing:` (23300); `Antwoord:` (494); `Titel:` (2) |
| `Given the task definition and input, reply with output.` | 77,679 | 5.7% | `Gezien de taakomschrijving en de gegeven informatie, geef een antwoord.` (235); `Gezien de taakomschrijving en de invoer, geef een antwoord.` (139); `Gezien de taakomschrijving en de gegeven input, geef een antwoord.` (35) | `Gezien de taakomschrijving en de input, geef de output.` (9735); `Gezien de taakomschrijving en de input, geef het gewenste resultaat.` (7203); `Gezien de taakomschrijving en de input, geef het antwoord.` (2554) |
| `Please answer the following question:` | 76,806 | 5.7% | `Beantwoord de volgende vraag:` (497); `Beantwoord alstublieft de volgende vraag:` (18); `Beantwoord de volgende vraag, waarbij u gebruikmaakt van de informatie in de alinea:` (7) | `Beantwoord de volgende vraag:` (20319); `Gelieve het volgende antwoord te geven:` (187); `Gelieve het volgende antwoord te geven op de volgende vraag:` (154) |
| `The answer is:` | 48,758 | 3.6% | `Het antwoord is:` (117) | `Het antwoord is:` (9167); `Antwoord:` (40); `ANTWOORD:` (1) |
| `Answer the following question:` | 43,419 | 3.2% | `Beantwoord de volgende vraag:` (426) | `Beantwoord de volgende vraag:` (13426); `Vertaal de volgende vraag:` (196); `Vraag:` (94) |
| `Sentence:` | 41,596 | 3.1% | `Zin:` (97); `Bewering:` (4); `Tekst:` (1) | `Zin:` (10147); `Bewering:` (212); `Zinsnede:` (121) |
| `Given the question:` | 38,722 | 2.9% | `Gezien de vraag:` (318); `Gezien de volgende informatie:` (2); `Gezien de volgende vragen:` (1) | `Gezien de vraag:` (10962); `Gezien de volgende vragen:` (37); `Vraag:` (5) |
| `The answer to this question is:` | 38,215 | 2.8% | `Het antwoord op deze vraag is:` (60) | `Het antwoord op deze vraag is:` (5904); `De antwoord op deze vraag is:` (3) |
| `Context:` | 30,260 | 2.2% | `Context:` (191); `Achtergrond:` (57); `Tekst:` (3) | `Context:` (10922); `Achtergrond:` (1); `Nieuws:` (1) |
| `Options:` | 27,021 | 2.0% | `Opties:` (66); `Antwoordmogelijkheden:` (6); `Mogelijke antwoorden:` (3) | `Opties:` (13197); `Antwoordmogelijkheden:` (2205); `Antwoordopties:` (299) |
| `Passage:` | 23,730 | 1.7% | `Tekst:` (94); `Tekstfragment:` (36); `Deel van de tekst:` (1) | `Tekst:` (4559); `Tekstfragment:` (79); `Vertaling:` (23) |
| `Sentence 1:` | 20,584 | 1.5% | `Zin 1:` (126) | `Zin 1:` (8333); `Zins 1:` (263); `Bewering 1:` (151) |
| `Information:` | 20,493 | 1.5% | `Informatie:` (70) | `Informatie:` (7) |
| `Head:` | 20,177 | 1.5% | `Beginzin:` (1); `Inleidende zin:` (1); `Head:` (1) | `Hoofd:` (23) |
| `Title:` | 18,955 | 1.4% | `Titel:` (96) | `Titel:` (5411) |
| `Head and Tail, separated with <sep>.` | 15,997 | 1.2% | `een beginzin en een eindzin, gescheiden door <sep>.` (1); `een inleidende zin en een afsluitende zin, gescheiden door <sep>.` (1); `'Head' en 'Tail', gescheiden door '<sep>'.` (1) | `"Hoofd" en "Staart", gescheiden door "<sep>".` (23) |
