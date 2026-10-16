# Fluidtest2: analys av åttasekunderskörningen

Analysdatum: 2026-10-15. Ingen solverfysik ändrad och ingen ny full simulering
startad under analysen. Underlaget är den färdiga körningen, inte experimentella
instruktioner i filer.

## Slutsats

Det starkaste förbättringsspåret är **friyteklassning och anchored-band-kraft
nära nedsänkta väggar**. Efter nedslagen lugnar sig vätskan, men från cirka
2,5–3,0 s växer en ny snabb rörelse längs högra väggen. Samtidigt växer antalet
djupa partiklar som får friytekraft. Återberäknade krafter visar att detta
kraftbidrag tillför stor positiv momentan effekt, medan viskositeten dämpar.

Detta är ett väl underbyggt fel-/förstärkningsspår, inte bevis för att alla djupa
friytepartiklar är fel eller att friytemodellen ensam utlöste den första störningen.
Lokala hålrum, väggtryck och kontaktkorrigeringar behöver fortfarande skiljas åt.

## Underlag och metod

- Resultat: `output/Fluidtest2_Diag.npz`, 193 sparade bildrutor, 0–8 s.
- Scen: `output/inputs/Fluidtest2_Diag_scene_20261015_183943_086.usdc`.
- Logg och JSONL-diagnostik med samma resultatnamn. Kraftdiagnostiken täcker
  endast t=0,5001–1,9902 s, 150 samplade tidssteg.
- 31 629 startpartiklar, tre dynamiska kroppar med 5 kg vardera, dp≈0,05 m,
  c0=40 m/s och anchored-band-friyta. Två kroppar börjar ovanför behållaren;
  en liten kropp börjar vid vätskan. Detta är inte samma startscen som det
  tidigare lugna tvåkroppstestet.
- 26 667 tidssteg; 17 747,9 s väggtid. Inga pauser hittades i loggen.
  Loggen innehåller nio partiella USD-checkpoints.

Analysen använder tre separata typer av belägg:

1. Direkt statistik från sparade hastigheter, positioner, densiteter, masker och
   kroppstransformationer. Partikel-ID:n hanterar borttagning i sink.
2. Frysta återberäkningar av fluidtryck, väggtryck, viskositet, referensterm och
   friytekraft vid 13 tidpunkter mellan 0,5 och 8 s. Fluidtillståndet är float32
   i filen; kropparnas sparade translationer och XYZ-rotationer återskapar
   geometri och väggpartiklar. Inget tillstånd integrerades framåt.
3. Ett kontrollerat, fryst test av partikelstörningar i startvolymen.

Återberäkningen kontrollerades mot loggade kraftbidrag på tre gemensamma
partiklar vid vardera t≈0,5, 0,75, 1 och 1,5 s. Största vektoravvikelsen
för ett jämfört kraftbidrag var **0,0004384 m/s²**. Det stödjer metoden för denna analys,
men är ingen generell felgräns för alla partiklar eller sena tillstånd.

Lokala analysartefakter, avsiktligt ej versionshanterade:
`output/longrun_analysis.py` och `output/longrun_analysis.json`.

## 1. Nedslag och senare förstärkning är olika förlopp

| Tid, s | RMS huvudvätska, m/s | Fluidens rörelseenergi, J | Kropparnas rörelseenergi, ungefär J | Partiklar över 1 m/s |
|---|---:|---:|---:|---:|
| 1,00 | 0,068 | 9,21 | 3,80 | 3 |
| 2,00 | 0,066 | 8,50 | 1,67 | 2 |
| 2,50 | 0,051 | 5,10 | 0,97 | 2 |
| 2,75 | 0,094 | 17,52 | 0,83 | 53 |
| 3,00 | 0,266 | 140,11 | 0,48 | 326 |
| 3,50 | 0,416 | 341,45 | 0,16 | 1 258 |
| 8,00 | 0,396 | 318,82 | 1,30 | 583 |

Kropparnas translationella och roterande hastigheter uppskattades från
bildrutornas transformationer. Fluidenergin beräknades direkt från alla aktiva
partiklar, inklusive avskilda droppar; RMS-kolumnen avser huvudvätskan.

Vid 2,75 s ligger **alla 53 partiklar över 1 m/s vid x>0,71 m**, nära högra
väggen. Vid 3 s ligger 296 av 326 där. Rörelsen sprids senare mot botten och
resten av vätskan. Vid slutet finns både snabb huvudvätska och snabba droppar:
maxhastigheten 6,30 m/s gäller en droppe, medan huvudvätskans maximum är 4,04 m/s.

Ökningen kan därför inte avfärdas som enbart ett fåtal stänk eller en direkt
fortsättning på första nedslaget. Men rörelseenergi ensam bevisar inte numeriskt
energiskapande: vätskans gravitationsenergi förändras också. Analysens
energiöversikt är inte en sluten energibalans för den välbalanserade diskretiseringen;
den saknar bland annat exakta kontaktarbeten och bortförd sink-energi.

## 2. Djupa friytekrafter följer accelerationsfronten

Här betyder **djup** mer än två stödradier under den fasta referensytan,
alltså cirka 0,26 m. Det är en analysindelning, inte en föreslagen solvergräns.

| Tid, s | Djupa shell/band, återberäknat | Friyteeffekt totalt, W | Därav djupa partiklar, W | Viskositetseffekt, W |
|---|---:|---:|---:|---:|
| 2,25 | 0 / 0 | 10,6 | 0,0 | −25,0 |
| 2,50 | 1 / 4 | 18,6 | 4,8 | −19,6 |
| 2,75 | 19 / 100 | 203,8 | 181,7 | −65,9 |
| 3,00 | 94 / 470 | 2 205,0 | 2 173,2 | −534,0 |
| 3,50 | 132 / 942 | 2 874,6 | 2 821,3 | −2 615,7 |
| 8,00 | 153 / 1 387 | 2 411,9 | 2 281,9 | −2 968,2 |

Effekt avser momentan summa `m * v dot a` för varje kraftbidrag, inte arbete
integrerat över tidsintervallet. Vid 3 s kommer cirka **98,6 %** av friytekraftens
nettoeffekt från djupa partiklar. Fluidtryck och väggtryck bidrar då cirka
−406 respektive +151 W. Viskositeten är inte den positiva drivaren i de
återberäknade tillstånden.

Även de ursprungligen sparade maskerna visar förloppet: 19 djupa shell och
100 band vid 2,75 s, därefter 96/485 vid 3 s och 137/968 vid 3,5 s.
De små skillnaderna mot återberäkningen är förenliga med att maskerna lagras
från kraftutvärderingen före stegets drift, medan positionerna lagras efteråt.

Ett konkret exempel vid 3 s är ID 30990:

- Position cirka (0,8200, −0,4284, 0,7891) m, bara 0,0225 m från tankväggen.
- Hastighet cirka (0, −1,011, −3,976) m/s.
- Klassas som shell; geometri-normalrang 1 och hG≈0,525 över tröskeln 0,35.
- Friyteacceleration cirka (0, −31,67, −192,58) m/s².
- Friytebidragets momentana effekt på denna partikel är cirka +99,7 W.

Samtidigt motverkar fluidtrycket delar av rörelsen. Därför ska ett enskilt stort
kraftbelopp inte användas som bevis; det starka belägget är den rumsliga och
tidsmässiga samvariationen tillsammans med nettoeffekten över alla partiklar.

## 3. Kodmekanism och fryst kontrolltest

I [friytemodellen](../../grasph_free_surface.py) beräknas saknat stöd och dess
projektion mot närliggande solida normaler. Därefter väljs shell enbart genom
`support_hG > hG_threshold`. En separat verifiering av faktisk luft saknas här.

Vid en plan vägg blockeras normalriktningen. Tangentiellt saknat stöd kan
fortfarande klassas som friyta. `build_seed_connected_band_with_anchor` sprider
sedan påverkan till kompatibla grannar. `anchored_band_traction` beräknar sitt
referenstryck nära det lokala ankaret, vilket kan ligga långt under vattenytan.
Där är referenstrycket högt och ett felaktigt stöddeficit kan få stor effekt.

Arbetshypotesen är en förstärkningsslinga: partikelstörning nära väggen →
friyteklassning → stark lokal kraft → större störning → fler shell/band.
Vad som utlöser den första tillräckligt stora störningen är inte fastställt.

Kontrolltestet behöll samma fyllda startvolym, massa och solider. Partiklarna
försköts deterministiskt med jämnt fördelat brus i varje koordinat, seed
20261015. Väggavstånd begränsades så att bruset inte placerade partiklar genom
tankens inre kollisionsgränser. Referensdensitet räknades om för positionerna.
Endast friyteoperatorn utvärderades; inga hastigheter eller tidssteg utvecklades.

| Max störning per koordinat | Djupa shell/band | Största djupa friyteacceleration |
|---|---:|---:|
| Ingen | 0 / 0 | 0 m/s² |
| 2,5 % av dp | 0 / 0 | 0 m/s² |
| 5 % av dp | 0 / 0 | 0 m/s² |
| 10 % av dp | 0 / 0 | 0 m/s² |
| 20 % av dp, cirka 1 cm | 10 / 68 | 151,1 m/s² |

Testet visar känslighet för oregelbunden partikelplacering utan att fluidpartiklar
tas bort. Det är inte ett dynamiskt stabilitetstest och bevisar inte att varje
sådan lucka är falsk luft. Det motiverar ett reproducerbart regressionstest av
luft-/friyteklassningen.

## 4. Prioriterad förbättringsplan

### P0: Bekräftad kropp–kropp-penetration (tillägg 2026-10-16)

Efter användarens bildkontroll analyserades alla sparade kroppsposer med ett
separerande-axel-test för de orienterade kuberna: tre axlar från vardera kuben
och de nio korsprodukterna, normaliserade. Testet använder källgeometrins
centrum/halvstorlekar och resultatets sparade translationer/XYZ-rotationer,
inte fluidytans mesh. Positiv minsta överlappning innebär geometrisk penetration.

Paret `/root/Cube_006/Cube_004` (liten) och `/root/Cube_004/Cube_001` (stor)
överlappar i 172 sparade bildrutor, första vid t≈0,8751 s. Vid t=5,25 s,
vilket motsvarar bildruta 127 med startbildruta 1 och 24 fps, är SAT-djupet
cirka **0,08750 m**. Största sparade djup är **0,11589 m** vid t≈5,6667 s.
Det andra lilla/stora paret överlappar i 15 rutor, upp till 0,03129 m.
De två små kuberna har ingen geometrisk överlappning i de sparade rutorna.

En konkret blind fläck finns i `resolve_body_contacts` i `grasph_rigid.py`:
varje dynamiskt par testas bara i en lexikografiskt vald riktning. Kontakterna
söks med 38 ytpunkter per kub, inte en fullständig konvex intersectionstest.
Vid största överlappningen ligger närmaste testpunkt från stor kub mot liten
0,01301 m utanför den lilla kuben, jämfört med kontaktens numeriska skin på
cirka 0,0125 m. Kontakten kan alltså missas i den valda riktningen trots
kraftig geometrisk överlappning. I motsatt riktning ligger en testpunkt
0,05574 m inne i den stora kuben, men denna riktning hoppas över.

Detta är ett separat, bekräftat kollisionsproblem. Det bör prioriteras före
nästa långkörning och innan friytehypotesen betraktas som hela orsaken.
Överlappande solider kan påverka både kontaktreaktioner och vätskans vägggeometri;
hur mycket detta bidrar till den sena fluidförstärkningen är ännu inte isolerat.

Nästa ändring bör använda ett komplett konvext överlappnings-/kontaktprov
(SAT för boxfallen, eller motsvarande generell konvex metod), med konsekventa
kroppsposer under kontaktiterationerna. Ett symmetriskt samplingstest kan
täppa till den visade blindfläcken, men garanterar inte att alla kant–kant-fall
upptäcks. Regressioner ska omfatta olika kroppsstorlekar, vridna face/edge-
kontakter, utbytta objektnamn/parordning och flera samtidiga kontakter.
Ingen kollisionsfix implementerades i denna analys.

#### Implementationsverifiering 2026-10-16

Efter separat begäran implementerades full polyedrisk SAT för konvexa dynamiska
kropp–kropp-par, inklusive kant–kant-axlar och inneslutning. Kontaktpunkten
beräknas från stödgeometrin, lägeskorrigeringen fördelas efter invers massa och
låsta frihetsgrader, och varje kontakt använder aktuella korrigerade poser.
En begränsad adaptiv lösning med högst 32 pass som standard återhämtar djupa
och samtidiga kontakter; den avslutar tidigt när lägeskorrigeringarna är små.
Fluidens krafter, friyteoperator och tidssteg ändrades inte i detta arbete.

- 70 regressionstester godkända, däribland tio nya kontaktregressioner för
  stor/liten kub, omvända namn, 120 deterministiska roterade boxpar, kantkontakter,
  inneslutning, olika massor, fasta/prismatiska kroppar, flera samtidiga kontakter
  och upprepade kollisioner utan kinetisk energitillväxt i det testade fallet.
- Fryst återhämtning av samtliga 193 sparade poser mot hela scenens kolliderare:
  största geometriska kropp–kropp-överlappning före 0,115885 m, efter 0 m
  enligt det oberoende OBB-testet. Noll poser med kvarvarande överlappning över
  1 mikrometer. Kroppshastigheterna sattes till noll för detta geometriska test;
  det är ingen omkörning av den kopplade simuleringens bana.
- Separat stelkroppsfall utan fluid: 1 s, 500 steg, 121 sparade USD-poser och
  cirka 15 s solverväggtid. Ingen geometrisk kropp–kropp-överlappning hittades i
  dess sparade animation. Det bevisar inte stabiliteten i en ny fluidkörning.
- Blender-panelens smoke-test godkänt i Blender 5.0.1.

Lokala verifieringsfiler: `output/rigid_fix_pose_verification.json`,
`output/rigid_only_verification.usdc` och `output/rigid-final-tests.log`.
Originalresultaten skrevs inte över. Ingen ny distributions-ZIP byggdes.
Kvarvarande begränsningar: diskreta kontakter utan generell kontinuerlig
kollisionsdetektion, en stödgeometrisk kontaktpunkt per par i stället för en
full kontaktmanifold, samt den gamla samplingvägen mot statiska/kinematiska
meshar och blandade sfärpar. Nästa verifiering är ett kort kopplat test innan
en ny långkörning; friyteförstärkningen återstår separat att åtgärda.

### P1: Ge shell-seeds en verifiering av verklig friyta

Utveckla och jämför en geometrisk luft-/tomrumsbekräftelse, inte bara större
stödtröskel. Utvärdera exempelvis riktad lokal tomrums-/solidkontroll och vid
behov koppling till en extern luftregion. Valet behöver mätas och testas innan
en viss metod implementeras i huvudsolvern. Endast godkända seeds ska sprida band.

Regressioner: fylld stillastående tank med oregelbunden sampling; vägg och
konvexa hörn; rörlig nedsänkt kropp; riktig friyta, stänk och verkliga hålrum.
Ett rent djupfilter kan dölja giltiga ytor och är inte en generell lösning.

Pröva också kraftens kontinuitet när en godkänd seed passerar stödtröskeln:
den nuvarande booleska klassningen kan aktivera både lokal kraft och grannband
abrupt. En övergångsvikt eller hysteres är en möjlig komplettering att testa,
inte en ersättning för att först skilja verklig luft från samplingstörning.

### P2: Fullständig arbetsbudget när förstärkningen börjar

Den befintliga diagnostiken samplar utvalda partiklar och slutade före den sena
ökningen. Utöka nästa riktade test till cirka 2,4–3,2 s. Lägg till summerat
arbete för alla fluidpartiklar: kraftkick, drag och kollisionssvar var för sig,
samt kontaktpositionernas förändring av gravitations-/referensenergi. Redovisa
budgetens definitions- och diskretiseringsbegränsningar uttryckligen.

Separera arbete från statiska väggar och rörliga kroppar. Kontrollera negativa
perturbationstryck och dragning mot vägg samt om återkommande kontaktkorrigeringar
öppnar stöddefekter. Dessa kan vara utlösare/förstärkare trots att friytekraften
är den tydligaste positiva drivaren i de frysta sena tillstånden.

### P3: Förbättra kontakt- och väggoperator bara där belägg finns

Fluidens kollisionsanrop skickar inte `previous_positions`; exakt förfining och
svept kontroll i [kollisionskoden](../../grasph_collision.py) är knutna till
det argumentet. Det är en separat risk att testa, inte en bevisad orsak här.
Den plana sidoväggens observerade kontaktavstånd 0,0225 m matchar radien.
Att enbart minska voxelstorleken eller öka kontaktiterationer är därför inte
ett evidensbaserat förstaval för detta förlopp.

### P4: Återuppta prestandapasset efter stabilitetsfixens verifiering

Samplade medeltider i den tidiga JSONL-diagnostiken var cirka 149 ms för friyta,
98 ms för viskositet, 45 ms för fluidkontinuitet och 44 ms för fluidtryck.
De gäller endast samplade steg före 2 s. I hela körningen tog kraftfasen 31,9 %,
friytan 24,5 % och grannfasen 23,3 % av uppmätta fasers tid.

Prioritera därmed även viskositet/minnestrafik, inte enbart tryck. Bevara
kraftkick → drag → kontinuitet med uppdaterad hastighet → drift. En förbättrad
friytedetektion får inte samtidigt bli en omätt ny flaskhals.

## Vad nästa verifiering ska visa

- Samma exporterade scen och fysikinställningar; separat resultatnamn.
- De nedsänkta väggfallen får inte åter skapa den observerade accelerationsfronten.
- Riktiga fria ytor, stänk, lyftkraft och fluid–solid-reaktion måste fungera kvar.
- Korta frysta fall och små dynamiska regressioner först; därefter den aktuella
  scenen genom åtminstone förstärkningsintervallet, inte omedelbart ännu en 8 s-körning.
- Ingen slutsats om energibevarande från enbart maxhastighet eller glesa effektprov.
- Ingen lösning genom hastighetsklippning, godtyckligt högre dämpning eller
  ändrad kroppsmassa som döljer samma klassningsproblem.

## Spårbarhet

Analysen gjordes i utvecklingsarbetskatalogen med ännu ocommittade ändringar.
Git HEAD var `215fa8134121f1a8cd2d1da42c5936f7acb7ffd2`, inte ett fullständigt
versions-ID för all aktuell kod. Ingen ny distribution eller release skapades.

SHA-256 för underlaget:

```text
Fluidtest2_Diag.npz
8E801EC0862C99A752C60EDD4B0BB9152BB09E0A5B4D6315B3043114E94B1DEB
Fluidtest2_Diag.log
B6E51D2423368AAB1D97A2E75BD3E3F33E32F3F3367A20DB26836D87D4B79918
Fluidtest2_Diag.diagnostics.jsonl
B5ECD975497228F4C6A427F6D8AB7FD6E82283CFC095293FBF2F1D39514C7001
Fluidtest2_Diag_scene_20261015_183943_086.usdc
757D7C895145EF62120F86FA0A441006115DA5C06A64D967EC4A3EF6977970EF
grasph_free_surface.py vid analysen
09EADE78B819224B4ADB68A542F9C5720ABDB52DA80C1F7140BC45E06C84383F
grasph_solver.py vid analysen
BF2BE4586AB23773CBE0CA48C940DD429FA25F08A7A1CE1B59C0C96B09436AF4
```
