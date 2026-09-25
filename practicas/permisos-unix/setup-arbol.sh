#!/usr/bin/env bash
set -euo pipefail

ANALYST_USER="maria"
PROGRAMMER_USER="javi"

ROOT="/home/arbol"
GESTION="${ROOT}/Gestion"
BIN="${ROOT}/bin"
PROGRAMAS="${ROOT}/Programas"

AUD_EST="${GESTION}/estudiante"
AUD_WAT="${GESTION}/watson"
AUD_INF="${AUD_EST}/informe.txt"
AUD_DOC="${AUD_WAT}/documento_informe.txtx"

PROG_EST="${PROGRAMAS}/estudiante"
PROG_ADA="${PROGRAMAS}/ada"
PROG_MAT="${PROG_EST}/matriz.c"
PROG_SIMPLE="${PROG_ADA}/simple.ada"

sudo chown "${PROGRAMMER_USER}" "${ROOT}"

sudo chown john "${GESTION}"
sudo chgrp administration "${GESTION}"

sudo chown joanna "${BIN}"
sudo chgrp administration "${BIN}"

sudo chown joanna "${PROGRAMAS}"
sudo chgrp administration "${PROGRAMAS}"

sudo chown "${ANALYST_USER}" "${AUD_EST}"
sudo chgrp auditor "${AUD_EST}"
[[ -f "${AUD_INF}" ]] && sudo chown "${ANALYST_USER}" "${AUD_INF}" && sudo chgrp auditor "${AUD_INF}"

sudo chown watson "${AUD_WAT}"
sudo chgrp auditor "${AUD_WAT}"
[[ -f "${AUD_DOC}" ]] && sudo chown watson "${AUD_DOC}" && sudo chgrp auditor "${AUD_DOC}"

sudo chown "${PROGRAMMER_USER}" "${PROG_EST}"
sudo chgrp programmer "${PROG_EST}"
[[ -f "${PROG_MAT}" ]] && sudo chown "${PROGRAMMER_USER}" "${PROG_MAT}" && sudo chgrp programmer "${PROG_MAT}"

sudo chown ada "${PROG_ADA}"
sudo chgrp programmer "${PROG_ADA}"
[[ -f "${PROG_SIMPLE}" ]] && sudo chown ada "${PROG_SIMPLE}" && sudo chgrp programmer "${PROG_SIMPLE}"
