# MusicSheetsRecognition

Projekat iz predmeta Soft kompjuting

<ul>
  <li>Članovi tima: Danijel Radulović, Dragan Ćulibrk</li>
  <br>
  <li>Asistent: Dragan Vidaković</li>
  <br>
  <li>Problem:<br>
    Čitanje muzičkih zapisa, prepoznavanje muzičkih simbola, generisanje njihove tekstualne predstave i melodije na osnovu toga.
  </li>
</ul>

## Skup podataka
Za trening i validacioni skup korišćen The Printed Images of Music Staves (PrIMuS) skup podataka https://grfia.dlsi.ua.es/primus/

## Metodologija
Korišćen End-to-end Optical Music Recognition algoritam, koji radi na principu prepoznavanja muzičkih simbola jedne linije notnog zapisa. Detaljnije u radu https://www.researchgate.net/publication/324460067_End-to-End_Neural_Optical_Music_Recognition_of_Monophonic_Scores i na Github repozitorijumu https://github.com/OMR-Research/tf-end-to-end)
<br>
<br>
Treniranu arhitekturu mreže čine:
<ul>
  <li>4 konvolutivna bloka, koji se sastoje od konvolutivnog sloja (početni broj filtera 32, u svakom narednom bloku se duplira, veličina kernela 3x3), batch normalizacije, leaky relu aktivacione funkcije i pooling sloja (pool_size 2x2)</li>
  <br>
  <li>2 bidirekciona LSTM sloja sa 256 jedinica i dropout sa koeficijentom 0.5</li>
  <br>
  <li>potpuno povezani sloj i Softmax
  </li><br>
  <li>Connectionist Temporal Classification (https://en.wikipedia.org/wiki/Connectionist_temporal_classification) loss, koji se koristi za treniranje rekurentnih neuronskih mreža, za rešavanje sequence-to-sequence problema gde postoji vremenska zavisnost između svakog dela ulazne sekvence.
  </li>
</ul>

Pre puštanja testne slike u neuronsku mrežu, izdvajaju se pojedinačne linije notnog zapisa uz pomoć computer vision tehnika. 

## Potrebne biblioteke i alati
Da bi se rešenje pokrenulo, potrebno je instalirati sledeće:
 <ul>
  <li>Python3</li>
  <li>Keras and Tensorflow</li>
  <li>Numpy</li>
  <li>OpenCV</li>
  <li>Music21</li>
 </ul>
 
 Rešenje je moguće pokrenuti:
 <ul>
  <li>Kao veb aplikaciju - potrebno je instalirati i Flask</li>
  <li>Iz komandne linije - python ctc_predict.py -image ../data/Example/000051652-1_2_1.png -model ../semantic_model/semantic_model.meta -vocabulary ../data/vocabulary_semantic.txt</li>
 </ul>
