# Beta cerrada — runbook

Objetivo: 10–30 testers jugando con el overlay en partidas reales, 2–4 semanas,
antes de cualquier lanzamiento público. La beta recoge evidencia de decisiones
de tienda explícitas; no demuestra que una compra cause una victoria ni que el
modelo sea "óptimo". El pipeline nocturno genera candidatos, pero nunca cambia
el modelo servido sin una evaluación y promoción manual.

## Qué necesita un tester (y qué no)

- **NO necesita Riot API key** ni Python: el overlay habla solo con la API
  local del cliente de League (`https://127.0.0.1:2999`) y con el CDN público
  de Data Dragon. La key solo hace falta en la máquina que ingesta y entrena.
- Windows 10/11 con WebView2 (viene de serie en Win11; el launcher cae a
  navegador si falta).
- League en **Borderless o Windowed** — sobre Fullscreen exclusivo no se puede
  dibujar (limitación de todos los overlays).

Instrucciones de tester: descomprimir `BuildAdvisor-beta.zip`, ejecutar
`BuildAdvisor.exe`, entrar a partida (Practice Tool vale para probar). El
widget aparece arriba a la derecha; se arrastra desde cualquier punto. **No da
consejo mientras juegas normalmente.** Cuando la tienda está abierta y vas a
decidir, pulsa `I’m deciding in shop`. Esa sesión dura 90 s: registra localmente
el oro exacto, las opciones que mostró y la siguiente **adición de inventario**
observada, o un no-buy si no compras. Un cambio que solo quite un ítem (por
ejemplo, un consumible usado) se excluye de métricas: no se inventa una compra.
`Cancel session` marca abandono, nunca un no-buy.

Los registros viven solo en `data\beta_telemetry\` dentro de la carpeta del
tester. No incluyen Riot ID, nombre de invocador, snapshot bruto ni inventario
de rivales. El tester puede entregar esa carpeta al evaluador; en la máquina de
desarrollo, copiar los JSONL bajo `data\beta_telemetry\` y revisar evidencia:

```powershell
python scripts\eval_shop_telemetry.py
```

Un solo basket mostrado que supere el oro exacto es **hard stop** de la beta.
No expandir ni promocionar el modelo mientras ese contador no sea cero.

## Build del bundle

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_beta.ps1        # release
powershell -ExecutionPolicy Bypass -File scripts\build_beta.ps1 -Console  # debug
```

Produce `dist\BuildAdvisor\` y `dist\BuildAdvisor-beta.zip`. Usa `.venv-beta`
(torch CPU: el modelo son 4MB y puntúa en milisegundos; CUDA engordaría el zip
~2GB). El bundle congela `app/advisor.py` (uvicorn en hilo + pywebview) y
copia `web/`, `scripts/train_prefix.py` (featurización), el modelo, su
`deployment_manifest.json` y la caché
de Data Dragon junto al exe (`app/config.py` ancla ROOT al exe cuando va
congelado). Actualizar el modelo de una beta ya distribuida = reemplazar
`data\ml\prefix_model.pt` **y** su `deployment_manifest.json` en la carpeta
del tester. El builder se niega a crear una beta si el par no está promovido y
ligado por hash.

## Personal API Key de Riot (bloqueante, pedir YA — tarda días)

Solo para la máquina de ingesta/entrenamiento (la dev key caduca cada 24h y
aborta el pull nocturno). Pedir en <https://developer.riotgames.com> → Register
Product → **Personal API Key**. Borrador de solicitud:

> **Product name:** Build Advisor
> **Description:** A desktop companion that suggests what a Challenger-level
> player would buy at each shop visit, live in-game. An ML model is trained on
> Match-v5 timelines of Challenger/Grandmaster ranked games (KR/EUW); at play
> time the app reads only the sanctioned Live Client Data API on localhost and
> static Data Dragon files — no scraping, no automation, no gameplay actions.
> The API key is used server-side only, for a nightly ingest of ranked match
> timelines within rate limits.
> **APIs used:** MATCH-V5 (matches + timelines), LEAGUE-V4 (ladder), ACCOUNT-V1.

Cuando llegue: reemplazar `RIOT_API_KEY` en `.env` — nada más que cambiar.

## Runbook de día de parche

1. Añadir la fecha go-live a `PATCH_STARTS_UTC` en `app/config.py` (la ingesta
   avisa con WARNING si falta; con fallback de 16 días funciona pero
   sobre-pide match lists).
2. Data Dragon se auto-descarga para la versión nueva (nada que hacer), pero
   ítems nuevos/cambiados no tienen datos de entrenamiento hasta que el
   pipeline acumule partidas del parche — esperar 1–2 noches antes de empujar
   modelo nuevo a testers.

## Checklist hacia la beta

- [ ] Personal API key solicitada / concedida
- [ ] `build_beta.ps1` produce bundle que arranca en una máquina limpia
- [ ] Validación en partida real propia (jugar con `python -m app.live --dump`
      y probar una sesión de tienda: compra, no-buy y cancelación)
- [ ] `scripts\eval_shop_telemetry.py` sin baskets por encima del oro exacto
- [ ] Candidato evaluado con `scripts\eval_policy.py` y
      `scripts\eval_conditional_policy.py` sobre validation, antes de promoción
- [ ] Candidato copiado como `data\ml\prefix_model.pt` y promovido con
      `scripts\promote_served_model.py --report policy_eval_val_<candidato>.json`
- [ ] Canal de feedback (Discord) + 10–30 testers
- [ ] Nombre/branding mínimo del widget
- [ ] Decidir cadencia de actualización del modelo hacia testers
