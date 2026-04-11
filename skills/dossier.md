---
name: dossier
description: Genera o actualiza el dossier de una oportunidad
args: [nombre_entidad]
---

Genera un dossier completo para: **{0}**

Usa la herramienta `search_web` para buscar:
- Qué hace {0}, sector, tamaño
- Programas de funding o colaboración disponibles en 2025-2026
- Contactos clave (directores, program officers)
- Casos de uso o proyectos similares al de David (sensores ultrasónicos, IIoT, cleantech)

Luego devuelve un análisis estructurado con:
- **Descripción**: qué es y qué ofrece
- **Fit con David**: por qué encaja (o no) con su perfil IIoT/cleantech NYC
- **Score estimado**: 0-100 basado en profile_match, timing, effort_vs_reward
- **Próximos pasos**: 2-3 acciones concretas con plazo
- **Por qué no**: obstáculos reales

Responde en español.
