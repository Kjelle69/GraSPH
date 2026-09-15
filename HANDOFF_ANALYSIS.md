GraSPH-handoff analyserad 2026-09-15. Arbetskatalogen innehåller v0.54 FIX6,
inte FIX5. Ingen zip eller ursprunglig FIX5-logg var bifogad. Solverns fysik
och parametrar har inte ändrats.

Det sparade `output/TankAnim_grasph.npz` innehåller 432 partiklar, dp=0,03 m,
91 snapshots och 3 s simulering. Maskerna är förenliga med FIX6:s klusterregel:
vid 1,20 s är fem partiklar detached trots att ingen saknar fluidgrannar.
NPZ lagrar inte versionsnummer eller droplet-mode, så dess exakta ursprung
kan inte fastställas enbart från filen.

En ny jämförelsekörning använder nuvarande FIX6-kod med `--droplet-mode zero`
och cutoff 0. Den isolerar klassificeringsregeln från FIX6:s CPU-förändringar;
den är inte en exakt återkörning av ursprunglig FIX5-kod.

| Mått | Sparat resultat, klusterliknande mask | Ny körning, strikt nollgrannarsregel |
| --- | ---: | ---: |
| Första registrerade detached | 1,20 s | Aldrig |
| Max samtidigt registrerade detached | 42 | 0 |
| Unika registrerade detached | 73 | 0 |
| Frikopplade partiklar enligt positionsgraf vid 3 s | 0 | 27 |
| Komponenter enligt positionsgraf vid 3 s | 1 | 4 |
| Partiklar i största komponent vid 3 s | 432 | 405 |
| Slutlig maximal hastighet | 1,352 m/s | 1,779 m/s |

Det sparade resultatet har 80 detached-perioder: 72 ger nettoförflyttning nedåt
och alla slutar med återanslutning i den registrerade masken. Totalt 200 av 242
detached-partikel/snapshot-observationer har negativ vertikal hastighet; 157
ligger inom dp från en ändlig collider-face enligt planprojektion. Detta stöder
att frikopplade kluster faller och återansluter. Det är inte ett direkt mått på
hur många partiklar som sitter fast: face-närhet omfattar även botten och Gate,
och metoden mäter inte avstånd till trianglarnas kanter. Vid 3 s har båda
körningarna noll face-nära partiklar ovanför den ursprungliga vattenytan.
En minskning av kvarhängande wall/ceiling-partiklar är därför inte bevisad av
detta grova test, trots den tydliga skillnaden i återanslutning.

Det sparade resultatets snapshot-densitet ligger mellan 925,38 och 1072,65
kg/m³ och maximal snapshot-hastighet är 3,277 m/s. Nollgrannarskörningens
snapshot-densitet ligger mellan 908,66 och 1057,51 kg/m³ med samma högsta
snapshot-hastighet. Alla sparade tillstånd är
finita. Den nya körningen slutför samtliga 10000 steg utan solverns
instabilitetsfel. Det finns ingen tätare FIX5-baslinje för att bedöma nya
instabiliteter. Huvudmassan kan inte antas vara oförändrad efter fragmentering:
återanslutande kluster kan ändra resten av vätskans rörelse. Positionerna är
exakt lika i samtliga snapshots före 1,17 s, vilket stöder en gemensam baslinje
före fragmenteringen. Vid 3 s skiljer sig tyngdpunkterna med 27,4 mm och RMS
av avståndet mellan motsvarande partikel-ID är 325 mm. Det senare inkluderar
partiklarnas omfördelning och är inte ett mått på skillnad i fri ytform.

Den nya 3 s-körningen tar 198,9 s inklusive output, ungefär 50,3 steg/s.
Av de uppmätta inner-loop-faserna går 53,1 % till wall interactions,
25,4 % till free surface, 8,9 % till SPH forces, 6,6 % till fluid neighbors
och 5,9 % till collision. Det gäller 432 partiklar och FIX6:s sökimplementation;
det är inte en hastighetsjämförelse med den ursprungliga FIX5-körningen.
En separat cProfile-körning över de första 100 stegen visar stor kostnad i
cross-neighbor search, front-facing-filtrering och finite-face covariance.
Det korta testet representerar stillastående vatten före Gate-rörelsen.

Rena CPU-optimeringar att testa separat, i denna ordning:

1. Återanvänd sökträd för statiska väggar. För translation-only kinematics kan
   separata träd ligga i respektive referenspose och frågepunkterna förskjutas.
   Verifiera exakt samma accepterade par och strikt support-gräns.
2. Förberäkna triangel-AABB, barycentriska konstanter och normalernas
   ytterprodukter. Translation ändrar bara läget, inte triangelns form.
3. Profilera och minska temporära arrayer i front-facing-filtreringen.
   Behåll normalriktning, facing positions och samma epsilon.

En fysikfråga finns kvar: `anchored_band_traction` hämtar fortfarande
hydrostatisk referenstryck från den ursprungliga vattenytan även för detached
kluster. Därför är beskrivningen "ordinary WCSPH" inte fullständig för hela
kraftsumman. Detta bör studeras i ett separat kraftdiagnostiktest innan någon
ytterligare fysikändring införs. Största-komponent-regeln behöver också prövas
för flera separata fluidvolymer; den klassar alla utom den största som detached.

USD-preflight är godkänd: Gate drivs av `/root/Gate`, maximal förflyttning är
0,192274 m. Det befintliga resultatet sublayerar `../TankAnim.usdc`, bevarar
Z-up och metersPerUnit=1 och har 91 Points-samples.

Analysen kan reproduceras med `py analysis_handoff.py`. Jämförelsen körs med
`py grasph.py TankAnim.usdc --time 3.0 --dp 0.03 --droplet-mode zero --name handoff_zero_3s`.
Detaljer finns i `output/handoff_analysis.json`, `output/handoff_zero_analysis.json`,
`output/handoff_zero_3s.log` och `output/handoff_zero_profile.pstats`.
Snapshots lagrar post-drift-positioner men mask och free-surface-state från
pre-drift-steget. Första detached-tiden är därför snapshot-upplöst, inte exakt
första solversteget. Continuum-meshing är ett senare steg enligt handoffen;
fallbacken behöver fortfarande tätare och fysikaliskt avgränsad validering.
